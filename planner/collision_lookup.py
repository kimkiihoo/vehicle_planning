import math
import numpy as np


class CollisionLookup:
    def __init__(self, mask, resolution, x_min, y_min):
        """
        mask       : 0/1 栅格地图 (0=障碍, 1=空闲)
        resolution : 栅格边长 (米/格)
        x_min/y_min: 世界坐标系下，栅格(0,0)对应的实际坐标
        """
        self.mask = mask
        self.res = resolution
        self.x0 = x_min
        self.y0 = y_min

        # 车辆几何
        self.W  = 3.2          # 宽
        self.LF = 4.51          # 后轴→车头
        self.LB = 1.01         # 后轴→车尾
        self.vehicle_corners = self._calc_vehicle_corners()

    # ---------------------------------------------------------------------
    #  基础工具
    # ---------------------------------------------------------------------
    def _calc_vehicle_corners(self):
        h = self.W / 2.0
        return np.array(
            [[ self.LF,  h],   # 左前
             [ self.LF, -h],   # 右前
             [-self.LB, -h],   # 右后
             [-self.LB,  h]]   # 左后
        )

    def _w2g(self, x, y):
        """世界坐标→栅格索引"""
        return ((x - self.x0) / self.res).astype(int), ((y - self.y0) / self.res).astype(int)

    def _g2w_center(self, ix, iy):
        """栅格索引→世界坐标(中心)"""
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def _point_collision(self, x, y):
        ix, iy = self._w2g(np.asarray(x), np.asarray(y))
        # 越界
        if ix < 0 or ix >= self.mask.shape[1] or iy < 0 or iy >= self.mask.shape[0]:
            return True
        # 0 表示障碍
        return self.mask[iy, ix] == 0

    # ---------------------------------------------------------------------
    #  对外主接口
    # ---------------------------------------------------------------------
    def collision_detection(self, x, y, yaw):
        """
        x, y : 后轴中心 (m)
        yaw  : 航向角 (rad)
        True = 碰撞
        """
        # ① 先跑一次快速轮廓检测，能提前返回时就少一次精确检测
        if self._contour_check(x, y, yaw):
            return True

        # ② 跑精确 NumPy 向量化检测
        return self._grid_check_vectorized(x, y, yaw)

    # ---------------------------------------------------------------------
    #  极速轮廓检测（与旧版一致）
    # ---------------------------------------------------------------------
    def _contour_check(self, x, y, yaw):
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        # 四角
        for cx, cy in self.vehicle_corners:
            wx = x + cx * cos_y - cy * sin_y
            wy = y + cx * sin_y + cy * cos_y
            if self._point_collision(wx, wy):
                return True

        # 车前后边线 5 点抽样
        for t in np.linspace(-self.W / 2, self.W / 2, 5):
            # 前
            if self._point_collision(x + self.LF * cos_y - t * sin_y,
                                     y + self.LF * sin_y + t * cos_y):
                return True
            # 后
            if self._point_collision(x - self.LB * cos_y - t * sin_y,
                                     y - self.LB * sin_y + t * cos_y):
                return True
        return False

    # ---------------------------------------------------------------------
    #  精确栅格检测（NumPy 向量化，无 python for 双循环）
    # ---------------------------------------------------------------------
    def _grid_check_vectorized(self, x, y, yaw):
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        # 四个角的世界坐标
        rot = np.array([[ cos_y, -sin_y],
                        [ sin_y,  cos_y]])
        world_corners = (rot @ self.vehicle_corners.T).T + np.array([x, y])

        # 计算外接 AABB 的栅格范围
        ix_min, iy_min = self._w2g(world_corners[:, 0].min(), world_corners[:, 1].min())
        ix_max, iy_max = self._w2g(world_corners[:, 0].max(), world_corners[:, 1].max())

        # 裁剪到地图范围
        ix_min = max(ix_min, 0)
        iy_min = max(iy_min, 0)
        ix_max = min(ix_max, self.mask.shape[1] - 1)
        iy_max = min(iy_max, self.mask.shape[0] - 1)

        # AABB 内若全为空闲，直接安全返回
        submask = self.mask[iy_min:iy_max + 1, ix_min:ix_max + 1]
        if submask.all():          # 全为 1 → 无障碍
            return False

        # 找到子块中所有障碍栅格的索引
        obs_iy, obs_ix = np.nonzero(submask == 0)
        if obs_ix.size == 0:       # 理论不会走到这，但留个保险
            return False

        # 转回全局索引
        obs_ix += ix_min
        obs_iy += iy_min

        # 障碍格中心的世界坐标
        gx, gy = self._g2w_center(obs_ix, obs_iy)

        # 转到车辆坐标系 (向量化)
        dx = gx - x
        dy = gy - y
        local_x =  dx * cos_y + dy * sin_y
        local_y = -dx * sin_y + dy * cos_y

        # 判断是否在车矩形内
        hit = ((-self.LB <= local_x) & (local_x <= self.LF) &
               (np.abs(local_y) <= self.W / 2))
        return hit.any()
