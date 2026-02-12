#!/usr/bin/env python3
"""
基于递归模型预测控制 (Recursive MPC) 的全局轨迹优化器
=======================================================

将 Hybrid A* 生成的粗糙全局轨迹进行平滑优化，使其满足：
  - 路径平滑性（最小化二阶/三阶导数）
  - 曲率连续性（避免急转弯）
  - 障碍物安全裕度
  - 起点/终点位姿精确保持

算法核心：
  采用滑动窗口策略，将长轨迹分段，在每个窗口内
  构造二次规划 (QP) 问题并通过求解线性方程组得到最优解，
  窗口之间通过重叠区域保证全局连续性（递归 MPC）。

依赖：仅 numpy（无需 scipy / cvxpy）。
"""

import math
import numpy as np


class RecursiveMPCOptimizer:
    """递归 MPC 全局轨迹优化器"""

    def __init__(self, params=None):
        """
        初始化优化器参数。

        Parameters
        ----------
        params : dict, optional
            可配置参数字典，任何未指定项使用默认值。
            - horizon       : int   滑动窗口长度（点数），默认 40
            - overlap        : int   相邻窗口重叠点数，默认 15
            - w_reference    : float 参考轨迹跟踪权重，默认 1.0
            - w_smooth       : float 二阶平滑权重（曲率最小化），默认 60.0
            - w_jerk         : float 三阶平滑权重（曲率变化率最小化），默认 80.0
            - w_obstacle     : float 障碍物惩罚权重，默认 500.0
            - obstacle_margin: float 安全距离（米），默认 2.0
            - max_iter       : int   障碍物迭代优化次数，默认 8
            - fix_start_n    : int   起点固定点数，默认 5
            - fix_end_n      : int   终点固定点数，默认 5
        """
        p = params or {}
        self.horizon = p.get('horizon', 40)
        self.overlap = p.get('overlap', 15)
        self.w_reference = p.get('w_reference', 1.0)
        self.w_smooth = p.get('w_smooth', 60.0)
        self.w_jerk = p.get('w_jerk', 80.0)
        self.w_obstacle = p.get('w_obstacle', 500.0)
        self.obstacle_margin = p.get('obstacle_margin', 2.0)
        self.max_iter = p.get('max_iter', 8)
        self.fix_start_n = p.get('fix_start_n', 5)
        self.fix_end_n = p.get('fix_end_n', 5)

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------

    def optimize(self, path_x, path_y, path_yaw, directions,
                 grid_map, map_cfg):
        """
        优化完整轨迹。

        Parameters
        ----------
        path_x, path_y : list[float]
            原始 x/y 坐标序列。
        path_yaw : list[float]
            原始航向角序列（弧度）。
        directions : list[bool]
            每个点的行驶方向 (True=前进, False=倒车)。
        grid_map : np.ndarray
            栅格地图 (0=障碍, 1=空闲)。
        map_cfg : dict
            地图参数，包含 x_min, y_min, resolution 等。

        Returns
        -------
        opt_x, opt_y, opt_yaw : list[float]
            优化后的轨迹坐标和航向角。
        """
        n = len(path_x)
        if n < 4:
            return list(path_x), list(path_y), list(path_yaw)

        opt_x = np.array(path_x, dtype=np.float64)
        opt_y = np.array(path_y, dtype=np.float64)
        ref_x = opt_x.copy()
        ref_y = opt_y.copy()

        # ── 滑动窗口递归优化 ──
        start = 0
        window_count = 0
        while start < n - 2:
            end = min(start + self.horizon, n)
            if end - start < 4:
                break

            # 确定该窗口内的固定点：首窗口固定头部，末窗口固定尾部
            fix_head = self.fix_start_n if start == 0 else 2
            fix_tail = self.fix_end_n if end == n else 0

            opt_x[start:end], opt_y[start:end] = self._optimize_window(
                opt_x, opt_y, ref_x, ref_y,
                start, end, fix_head, fix_tail,
                grid_map, map_cfg
            )
            window_count += 1

            # 步进（减去重叠）
            step = max(1, self.horizon - self.overlap)
            start += step

        # ── 重算航向角 ──
        opt_yaw = self._recalculate_yaw(opt_x, opt_y, path_yaw, directions)

        print(f"  递归MPC优化完成: {window_count} 个窗口, "
              f"路径点数: {n}")

        return opt_x.tolist(), opt_y.tolist(), opt_yaw

    # ------------------------------------------------------------------
    #  窗口优化核心
    # ------------------------------------------------------------------

    def _optimize_window(self, opt_x, opt_y, ref_x, ref_y,
                         start, end, fix_head, fix_tail,
                         grid_map, map_cfg):
        """
        优化单个窗口内的轨迹。

        采用迭代线性化策略：
          1) 构建 QP 基础矩阵（参考跟踪 + 平滑性）
          2) 迭代检查障碍物近距离点，添加推力修正
          3) 求解线性方程组得到优化位置
        """
        n = end - start
        wx = opt_x[start:end].copy()
        wy = opt_y[start:end].copy()
        rx = ref_x[start:end]
        ry = ref_y[start:end]

        # 确定自由变量索引（非固定点）
        fixed = set()
        for i in range(fix_head):
            fixed.add(i)
        for i in range(n - fix_tail, n):
            fixed.add(i)
        free_idx = [i for i in range(n) if i not in fixed]

        if len(free_idx) < 2:
            return wx, wy

        # ── 构建 QP 基础矩阵 (仅自由变量) ──
        nf = len(free_idx)
        H_base = self._build_qp_matrix(n, free_idx)

        # ── 迭代障碍物修正 ──
        for iteration in range(self.max_iter):
            # 构造右端向量（参考跟踪项）
            bx = self.w_reference * rx[free_idx]
            by = self.w_reference * ry[free_idx]

            # 减去固定点对平滑项的贡献
            bx_corr, by_corr = self._fixed_point_correction(
                n, free_idx, fixed, wx, wy)
            bx -= bx_corr
            by -= by_corr

            # 障碍物推力
            H_obs = np.zeros((nf, nf))
            bx_obs = np.zeros(nf)
            by_obs = np.zeros(nf)
            has_obs = self._add_obstacle_forces(
                wx, wy, free_idx, grid_map, map_cfg,
                H_obs, bx_obs, by_obs
            )

            # 求解
            H = H_base + H_obs
            # 添加正则化确保正定
            H += np.eye(nf) * 1e-6

            try:
                new_x = np.linalg.solve(H, bx + bx_obs)
                new_y = np.linalg.solve(H, by + by_obs)
            except np.linalg.LinAlgError:
                break

            # 更新自由变量
            for k, idx in enumerate(free_idx):
                wx[idx] = new_x[k]
                wy[idx] = new_y[k]

            if not has_obs:
                break  # 无障碍物冲突，无需继续迭代

        return wx, wy

    # ------------------------------------------------------------------
    #  QP 矩阵构建
    # ------------------------------------------------------------------

    def _build_qp_matrix(self, n, free_idx):
        """
        构建二次规划的 Hessian 矩阵 H。

        Cost = w_ref * ||p - p_ref||²
             + w_smooth * ||D2 * p||²
             + w_jerk * ||D3 * p||²

        其中 D2 是二阶差分矩阵 (p_{i-1} - 2*p_i + p_{i+1})，
             D3 是三阶差分矩阵 (p_{i+2} - 3*p_{i+1} + 3*p_i - p_{i-1})。

        H = w_ref * I + w_smooth * D2^T D2 + w_jerk * D3^T D3
        """
        nf = len(free_idx)
        free_set = set(free_idx)
        # 建立 free index -> 矩阵行列索引 映射
        idx_map = {v: k for k, v in enumerate(free_idx)}

        H = np.eye(nf) * self.w_reference  # 参考跟踪项

        # ── 二阶差分平滑 ──
        for i in range(1, n - 1):
            # 差分涉及 i-1, i, i+1
            triplet = [i - 1, i, i + 1]
            coeffs = [1.0, -2.0, 1.0]
            involved_free = [(j, c) for j, c in zip(triplet, coeffs)
                             if j in free_set]
            if not involved_free:
                continue
            for ja, ca in involved_free:
                for jb, cb in involved_free:
                    H[idx_map[ja], idx_map[jb]] += self.w_smooth * ca * cb

        # ── 三阶差分（jerk 最小化）──
        for i in range(1, n - 2):
            # 差分涉及 i-1, i, i+1, i+2
            quartet = [i - 1, i, i + 1, i + 2]
            coeffs = [-1.0, 3.0, -3.0, 1.0]
            involved_free = [(j, c) for j, c in zip(quartet, coeffs)
                             if j in free_set]
            if not involved_free:
                continue
            for ja, ca in involved_free:
                for jb, cb in involved_free:
                    H[idx_map[ja], idx_map[jb]] += self.w_jerk * ca * cb

        return H

    def _fixed_point_correction(self, n, free_idx, fixed, wx, wy):
        """
        计算固定点对平滑项右端向量的贡献。

        当差分涉及固定点时，其值需要移到方程右端。
        """
        nf = len(free_idx)
        free_set = set(free_idx)
        idx_map = {v: k for k, v in enumerate(free_idx)}

        bx_corr = np.zeros(nf)
        by_corr = np.zeros(nf)

        # ── 二阶差分中固定点的贡献 ──
        for i in range(1, n - 1):
            triplet = [i - 1, i, i + 1]
            coeffs = [1.0, -2.0, 1.0]
            involved_free = [(j, c) for j, c in zip(triplet, coeffs)
                             if j in free_set]
            involved_fixed = [(j, c) for j, c in zip(triplet, coeffs)
                              if j in fixed]
            if not involved_free or not involved_fixed:
                continue
            # 固定点值移到右端: -w * c_free * c_fixed * p_fixed
            for jf, cf in involved_free:
                for jx, cx in involved_fixed:
                    bx_corr[idx_map[jf]] += self.w_smooth * cf * cx * wx[jx]
                    by_corr[idx_map[jf]] += self.w_smooth * cf * cx * wy[jx]

        # ── 三阶差分中固定点的贡献 ──
        for i in range(1, n - 2):
            quartet = [i - 1, i, i + 1, i + 2]
            coeffs = [-1.0, 3.0, -3.0, 1.0]
            involved_free = [(j, c) for j, c in zip(quartet, coeffs)
                             if j in free_set]
            involved_fixed = [(j, c) for j, c in zip(quartet, coeffs)
                              if j in fixed]
            if not involved_free or not involved_fixed:
                continue
            for jf, cf in involved_free:
                for jx, cx in involved_fixed:
                    bx_corr[idx_map[jf]] += self.w_jerk * cf * cx * wx[jx]
                    by_corr[idx_map[jf]] += self.w_jerk * cf * cx * wy[jx]

        return bx_corr, by_corr

    # ------------------------------------------------------------------
    #  障碍物检测与推力
    # ------------------------------------------------------------------

    def _add_obstacle_forces(self, wx, wy, free_idx, grid_map, map_cfg,
                             H_obs, bx_obs, by_obs):
        """
        检测轨迹点的障碍物接近情况，添加推离力。

        对于距离障碍物小于 obstacle_margin 的点：
          - 计算到最近障碍物的方向
          - 在 QP 中添加惩罚项，将点推向安全方向
        """
        res = map_cfg['resolution']
        x_min = map_cfg['x_min']
        y_min = map_cfg['y_min']
        h, w = grid_map.shape
        idx_map = {v: k for k, v in enumerate(free_idx)}
        free_set = set(free_idx)
        margin_cells = int(self.obstacle_margin / res)
        has_any = False

        for idx in free_idx:
            x, y = wx[idx], wy[idx]
            gx = int(round((x - x_min) / res))
            gy = int(round((y - y_min) / res))

            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                continue

            # 在局部区域搜索最近障碍物
            min_dist_sq = float('inf')
            nearest_ox, nearest_oy = x, y
            found = False

            search_r = margin_cells
            for dy in range(-search_r, search_r + 1):
                for dx in range(-search_r, search_r + 1):
                    cx, cy = gx + dx, gy + dy
                    if 0 <= cx < w and 0 <= cy < h and grid_map[cy, cx] == 0:
                        # 障碍物栅格
                        ox = x_min + cx * res
                        oy = y_min + cy * res
                        dsq = (x - ox) ** 2 + (y - oy) ** 2
                        if dsq < min_dist_sq:
                            min_dist_sq = dsq
                            nearest_ox, nearest_oy = ox, oy
                            found = True

            if not found:
                continue

            dist = math.sqrt(min_dist_sq)
            if dist >= self.obstacle_margin:
                continue

            has_any = True
            # 推力方向：从障碍物指向轨迹点
            if dist < 1e-6:
                # 点在障碍物内，用与参考方向的法向推出
                push_x, push_y = 1.0, 0.0
            else:
                push_x = (x - nearest_ox) / dist
                push_y = (y - nearest_oy) / dist

            # 惩罚强度随距离衰减（越近越强）
            strength = self.w_obstacle * (1.0 - dist / self.obstacle_margin) ** 2

            k = idx_map[idx]
            # 添加到 Hessian 对角线
            H_obs[k, k] += strength
            # 目标位置 = 当前位置 + push_dir * margin
            target_x = nearest_ox + push_x * self.obstacle_margin
            target_y = nearest_oy + push_y * self.obstacle_margin
            bx_obs[k] += strength * target_x
            by_obs[k] += strength * target_y

        return has_any

    # ------------------------------------------------------------------
    #  航向角重算
    # ------------------------------------------------------------------

    def _recalculate_yaw(self, opt_x, opt_y, orig_yaw, directions):
        """
        根据优化后的位置序列重新计算航向角。

        前进时航向 = atan2(dy, dx)
        倒车时航向 = atan2(dy, dx) + π（车头朝反方向）
        最后一个点保持原始航向。
        """
        n = len(opt_x)
        yaw = np.array(orig_yaw, dtype=np.float64)

        for i in range(n - 1):
            dx = opt_x[i + 1] - opt_x[i]
            dy = opt_y[i + 1] - opt_y[i]
            if abs(dx) < 1e-8 and abs(dy) < 1e-8:
                continue  # 保持原始航向

            theta = math.atan2(dy, dx)
            if directions[i]:
                yaw[i] = theta
            else:
                # 倒车：车头朝行驶反方向
                yaw[i] = self._normalize_angle(theta + math.pi)

        # 最后一个点保持与原始一致
        yaw[-1] = orig_yaw[-1]

        return yaw.tolist()

    @staticmethod
    def _normalize_angle(angle):
        """将角度归一化到 [-π, π]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
