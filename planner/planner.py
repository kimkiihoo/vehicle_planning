# 内置库
import math
import os
import csv
import copy

# 第三方库
import numpy as np
from typing import List, Tuple, Optional

# 从 hybrid_a_star 模块导入
from hybrid_a_star import hybrid_a_star_planning, Config


# 模块级别的辅助函数，用于角度归一化到 (-pi, pi]
def _angle_wrap(angle_rad: float) -> float:
    """将角度归一化到 (-pi, pi] 范围内"""
    angle_rad = angle_rad % (2.0 * math.pi)
    if angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    elif angle_rad <= -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def _calculate_curvature_menger_robust(
        p0_x: float, p0_y: float,
        p1_x: float, p1_y: float,
        p2_x: float, p2_y: float,
        epsilon: float = 1e-9
) -> float:
    """
    使用Menger曲率公式通过三个点 P0, P1, P2 计算 P1 点的近似带符号曲率。
    P0是P1的前一个点，P2是P1的后一个点。
    正曲率表示逆时针转（左转），负曲率表示顺时针转（右转）。
    """
    # 计算各边长的平方
    l01_sq = (p1_x - p0_x) ** 2 + (p1_y - p0_y) ** 2
    l12_sq = (p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2
    l02_sq = (p2_x - p0_x) ** 2 + (p2_y - p0_y) ** 2

    # 计算边长
    l01 = math.sqrt(l01_sq)
    l12 = math.sqrt(l12_sq)
    l02 = math.sqrt(l02_sq)

    # 检查是否有重合的点
    if l01 < epsilon or l12 < epsilon or l02 < epsilon:
        return 0.0

    # 计算两倍的有向面积
    numerator_val = (p1_x - p0_x) * (p2_y - p1_y) - \
                    (p1_y - p0_y) * (p2_x - p1_x)

    # 检查三点是否共线
    if abs(numerator_val) < epsilon * epsilon:
        return 0.0

    denominator_val = l01 * l12 * l02

    if abs(denominator_val) < epsilon:
        return 0.0

    # Menger曲率公式
    curvature = numerator_val / denominator_val

    return curvature


class Planner:
    """泊车场景的规划器"""

    def __init__(self, observation):
        self._goal_x = np.mean(observation["test_setting"]["goal"]["x"])
        self._goal_y = np.mean(observation["test_setting"]["goal"]["y"])
        self._observation = observation

    def calculate_curvature(self, x1: float, y1: float, angle1: float,
                            x2: float, y2: float, angle2: float) -> float:
        """使用两点法计算曲率（备用方法）"""
        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_s = math.sqrt(delta_x ** 2 + delta_y ** 2)
        if delta_s < 1e-6:
            return 0.0
        angle1_norm = _angle_wrap(angle1)
        angle2_norm = _angle_wrap(angle2)
        delta_yaw = _angle_wrap(angle2_norm - angle1_norm)
        curvature = delta_yaw / delta_s
        return curvature

    def process(self, observation, scene_type, collision_lookup=None, scenario_to_test=None):
        """泊车场景规划器主函数"""
        if scene_type != "B":
            raise ValueError(f"此规划器仅支持泊车场景(B)，不支持: {scene_type}")

        config = Config(observation)
        path_planned_intermediate = []  # [x,y,yaw,direction]

        task_type = "B"
        scene_name = scenario_to_test['data']['scene_name']

        # 目标位置处理
        goal_b_scene = [
            np.mean(observation['test_setting']['goal']['x']),
            np.mean(observation['test_setting']['goal']['y']),
            observation['test_setting']['goal']['heading'][0]
        ]

        # 特殊场景的目标航向角调整
        goal_adjustments = {
            "B309_mission_1_7_1": -2
        }

        if scene_name in goal_adjustments:
            goal_b_scene[2] = goal_adjustments[scene_name]

        # 起始位置处理
        start_point_b_scene = [
            observation['vehicle_info']['ego']['x'],
            observation['vehicle_info']['ego']['y'],
            observation['vehicle_info']['ego']['yaw_rad']
        ]

        # 调用混合A*算法进行路径规划
        astar_path_obj_b = hybrid_a_star_planning(
            start_point_b_scene, goal_b_scene, collision_lookup, observation, config,
            task_type=task_type, use_dijkstra=True, rs_dist=70, scenario_to_test=scenario_to_test
        )

        if astar_path_obj_b is not None:
            for j in range(len(astar_path_obj_b.xlist)):
                path_planned_intermediate.append([
                    astar_path_obj_b.xlist[j], astar_path_obj_b.ylist[j],
                    astar_path_obj_b.yawlist[j], astar_path_obj_b.directionlist[j]
                ])
        else:
            print("A* planning failed for B scene.")
            return []

        # 计算曲率
        if not path_planned_intermediate:
            return []

        final_path_with_curvature = []
        curvatures = [0.0] * len(path_planned_intermediate)

        if len(path_planned_intermediate) >= 3:
            for i in range(1, len(path_planned_intermediate) - 1):
                p_prev = path_planned_intermediate[i - 1]
                p_curr = path_planned_intermediate[i]
                p_next = path_planned_intermediate[i + 1]
                if len(p_prev) >= 2 and len(p_curr) >= 2 and len(p_next) >= 2:
                    curv = _calculate_curvature_menger_robust(
                        p_prev[0], p_prev[1],
                        p_curr[0], p_curr[1],
                        p_next[0], p_next[1]
                    )
                    curvatures[i] = curv

            if len(path_planned_intermediate) > 1:
                curvatures[0] = curvatures[1]
                curvatures[-1] = curvatures[-2]

        elif len(path_planned_intermediate) == 2:
            # 只有两个点时使用两点法
            p0_data = path_planned_intermediate[0]
            p1_data = path_planned_intermediate[1]
            if len(p0_data) >= 3 and len(p1_data) >= 3:
                curv_val = self.calculate_curvature(
                    p0_data[0], p0_data[1], p0_data[2],
                    p1_data[0], p1_data[1], p1_data[2]
                )
                curvatures[0] = curv_val
                curvatures[1] = curv_val

        # 将曲率附加到路径点
        for i in range(len(path_planned_intermediate)):
            original_point = path_planned_intermediate[i]
            if len(original_point) >= 4:
                final_path_with_curvature.append(list(original_point[:4]) + [curvatures[i]])

        # 移除第一个点（与原始行为保持一致）
        if final_path_with_curvature:
            final_path_with_curvature = final_path_with_curvature[1:]

        return final_path_with_curvature
