#!/usr/bin/env python3
"""
非结构道路车辆轨迹规划与泊车 Demo
===================================
基于混合A*算法，从 input/ 加载地图、障碍物和场景配置，
规划车辆从起点到停车位的完整轨迹，
输出 output.json / output.jpg / output.gif。
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon as ShapelyPolygon
import imageio.v2 as imageio

# ── 把项目根目录加入 sys.path ─────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from planner.hybrid_a_star import hybrid_a_star_planning, Config
from planner.collision_lookup import CollisionLookup
from optimizer.recursive_mpc import RecursiveMPCOptimizer

# ── 中文字体 ──────────────────────────────────────
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.path.exists(FONT_PATH):
    # 备用
    FONT_PATH = "/usr/share/fonts/truetype/arphic/uming.ttc"
FONT_PROP = FontProperties(fname=FONT_PATH, size=12)
FONT_PROP_TITLE = FontProperties(fname=FONT_PATH, size=16)
FONT_PROP_SMALL = FontProperties(fname=FONT_PATH, size=9)
FONT_PROP_LEGEND = FontProperties(fname=FONT_PATH, size=10)

# ── 车辆几何参数 ─────────────────────────────────
VEHICLE_W = 3.2
VEHICLE_LF = 4.51
VEHICLE_LB = 1.01

# ── 障碍物膨胀半径 ─────────────────────────────────
OBSTACLE_INFLATION = 0.15  # 米


# ===========================================================================
#  1. 加载 JSON 输入
# ===========================================================================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_inputs(input_dir):
    map_cfg = load_json(os.path.join(input_dir, 'map.json'))
    obstacles = load_json(os.path.join(input_dir, 'obstacles.json'))
    scenario = load_json(os.path.join(input_dir, 'scenario.json'))
    return map_cfg, obstacles, scenario


# ===========================================================================
#  2. 构建栅格地图
# ===========================================================================

def build_grid_map(map_cfg, obstacles):
    """将多边形障碍物光栅化到二值栅格地图 (0=障碍, 1=空闲)

    对每个障碍物施加 OBSTACLE_INFLATION 膨胀半径，避免碰撞漏检。
    """
    from shapely.geometry import Point

    res = map_cfg['resolution']
    x_min, y_min = map_cfg['x_min'], map_cfg['y_min']
    x_max, y_max = map_cfg['x_max'], map_cfg['y_max']

    width = int(round((x_max - x_min) / res))
    height = int(round((y_max - y_min) / res))
    grid = np.ones((height, width), dtype=np.uint8)  # 全为空闲

    for obs in obstacles['obstacles']:
        verts = obs['vertices']
        poly = ShapelyPolygon(verts)
        # 施加膨胀半径
        if OBSTACLE_INFLATION > 0:
            poly = poly.buffer(OBSTACLE_INFLATION)
        # 获取轴对齐包围盒
        minx_o, miny_o, maxx_o, maxy_o = poly.bounds
        ix_min = max(0, int(math.floor((minx_o - x_min) / res)))
        iy_min = max(0, int(math.floor((miny_o - y_min) / res)))
        ix_max = min(width - 1, int(math.ceil((maxx_o - x_min) / res)))
        iy_max = min(height - 1, int(math.ceil((maxy_o - y_min) / res)))

        for iy in range(iy_min, iy_max + 1):
            for ix in range(ix_min, ix_max + 1):
                wx = x_min + ix * res
                wy = y_min + iy * res
                if poly.contains(Point(wx, wy)):
                    grid[iy, ix] = 0

    print(f"栅格地图构建完成: {width}x{height}, 障碍像素占比: "
          f"{(grid == 0).sum() / grid.size * 100:.1f}%"
          f" (膨胀半径: {OBSTACLE_INFLATION}m)")
    return grid


# ===========================================================================
#  3. 计算曲率
# ===========================================================================

def calculate_curvatures(path_x, path_y, path_yaw):
    """计算路径每一点的曲率"""
    n = len(path_x)
    curvatures = [0.0] * n
    if n >= 3:
        for i in range(1, n - 1):
            x0, y0 = path_x[i - 1], path_y[i - 1]
            x1, y1 = path_x[i], path_y[i]
            x2, y2 = path_x[i + 1], path_y[i + 1]
            # Menger 曲率
            num = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
            l01 = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            l12 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            l02 = math.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)
            den = l01 * l12 * l02
            if den > 1e-9:
                curvatures[i] = num / den
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
    return curvatures


# ===========================================================================
#  4. 输出 JSON
# ===========================================================================

def save_output_json(path, path_x, path_y, path_yaw, directions, curvatures,
                     iter_num, search_time):
    data = {
        "planning_info": {
            "algorithm": "Hybrid A* + Reeds-Shepp",
            "iterations": iter_num,
            "search_time_sec": round(search_time, 3),
            "path_length": len(path_x),
        },
        "trajectory": []
    }
    for i in range(len(path_x)):
        data["trajectory"].append({
            "index": i,
            "x": round(path_x[i], 4),
            "y": round(path_y[i], 4),
            "yaw_rad": round(path_yaw[i], 4),
            "yaw_deg": round(math.degrees(path_yaw[i]), 2),
            "direction": "前进" if directions[i] else "倒车",
            "curvature": round(curvatures[i], 6)
        })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"轨迹数据已保存至: {path}")


# ===========================================================================
#  5. 绘图工具
# ===========================================================================

def draw_vehicle(ax, x, y, yaw, color='royalblue', alpha=0.7, label=None):
    """绘制车辆矩形"""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    corners_local = np.array([
        [VEHICLE_LF, VEHICLE_W / 2],
        [VEHICLE_LF, -VEHICLE_W / 2],
        [-VEHICLE_LB, -VEHICLE_W / 2],
        [-VEHICLE_LB, VEHICLE_W / 2],
    ])
    corners_world = np.zeros_like(corners_local)
    for i, (cx, cy) in enumerate(corners_local):
        corners_world[i, 0] = x + cx * cos_y - cy * sin_y
        corners_world[i, 1] = y + cx * sin_y + cy * cos_y

    poly = plt.Polygon(corners_world, closed=True, facecolor=color, edgecolor='black',
                        alpha=alpha, linewidth=1.2, label=label, zorder=5)
    ax.add_patch(poly)

    # 车头方向箭头
    arrow_len = 2.5
    ax.annotate('', xy=(x + arrow_len * cos_y, y + arrow_len * sin_y), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.8), zorder=6)




def setup_plot(ax, map_cfg, obstacles, scenario, title, grid_map=None):
    """设置基础地图绘图"""
    x_min, y_min = map_cfg['x_min'], map_cfg['y_min']
    x_max, y_max = map_cfg['x_max'], map_cfg['y_max']
    res = map_cfg['resolution']

    # 地图背景
    if grid_map is not None:
        extent = [x_min, x_max, y_min, y_max]
        ax.imshow(grid_map, cmap='gray', origin='lower', extent=extent,
                  alpha=0.3, zorder=0)

    # 障碍物
    obstacle_colors = {
        '路边护栏': '#8B4513',
        '碎石堆': '#A0522D',
        '施工区域': '#FF8C00',
        '植被区域': '#228B22',
        '停放车辆A': '#4682B4',
        '废弃物堆': '#696969',
        '临时围栏': '#CD853F',
        '车位左栏杆': '#708090',
        '车位右栏杆': '#708090',
        '车位后栏杆': '#708090',
    }
    for obs in obstacles['obstacles']:
        verts = obs['vertices']
        label = obs.get('label', '')
        if label in ('下边界', '上边界', '左边界', '右边界'):
            color = '#555555'
            alpha = 0.5
        else:
            color = obstacle_colors.get(label, '#888888')
            alpha = 0.75
        poly = plt.Polygon(verts, closed=True, facecolor=color,
                           edgecolor='black', alpha=alpha, linewidth=0.8, zorder=2)
        ax.add_patch(poly)
        # 标注
        if label not in ('下边界', '上边界', '左边界', '右边界'):
            cx = np.mean([v[0] for v in verts])
            cy = np.mean([v[1] for v in verts])
            ax.text(cx, cy, label, fontsize=7, color='white',
                    ha='center', va='center', fontproperties=FONT_PROP_SMALL,
                    fontweight='bold', zorder=3)

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (米)', fontproperties=FONT_PROP)
    ax.set_ylabel('Y (米)', fontproperties=FONT_PROP)
    ax.set_title(title, fontproperties=FONT_PROP_TITLE, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.2, linestyle='--')


# ===========================================================================
#  6. 输出静态图片
# ===========================================================================

def save_output_jpg(path, map_cfg, obstacles, scenario, grid_map,
                    path_x, path_y, path_yaw, directions,
                    orig_path=None):
    """保存规划结果静态图（orig_path 为优化前轨迹，用于对比显示）"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    setup_plot(ax, map_cfg, obstacles, scenario,
               '非结构道路车辆轨迹规划与泊车演示', grid_map)

    # 轨迹 — 按方向分色
    for i in range(len(path_x) - 1):
        color = '#00ff88' if directions[i] else '#ff6b6b'
        ax.plot(path_x[i:i + 2], path_y[i:i + 2], color=color,
                linewidth=2.0, alpha=0.9, zorder=3)

    # 起点车辆
    sx, sy = scenario['start']['x'], scenario['start']['y']
    syaw = math.radians(scenario['start']['yaw_deg'])
    draw_vehicle(ax, sx, sy, syaw, color='#00b4d8', alpha=0.9, label='起始位置')

    # 终点车辆
    gx, gy = scenario['goal']['x'], scenario['goal']['y']
    gyaw = math.radians(scenario['goal']['yaw_deg'])
    draw_vehicle(ax, gx, gy, gyaw, color='#e63946', alpha=0.9, label='目标位置')

    # 方向图例
    ax.plot([], [], color='#00ff88', linewidth=2.5, label='前进轨迹')
    ax.plot([], [], color='#ff6b6b', linewidth=2.5, label='倒车轨迹')

    # 优化前轨迹（白色虚线对比）
    if orig_path is not None:
        ox, oy = orig_path
        ax.plot(ox, oy, color='#ffffff', linewidth=1.0, alpha=0.35,
                linestyle='--', zorder=2, label='优化前轨迹')

    legend = ax.legend(loc='upper left', prop=FONT_PROP_LEGEND,
                       facecolor='#1a1a2e', edgecolor='white',
                       labelcolor='white', framealpha=0.8)

    # 信息框
    import planner.hybrid_a_star as ha
    info_text = (f"算法: 混合A* + Reeds-Shepp\n"
                 f"迭代次数: {ha.LAST_ITER_NUM}\n"
                 f"搜索用时: {ha.LAST_SEARCH_TIME:.2f}秒\n"
                 f"路径点数: {len(path_x)}")
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontproperties=FONT_PROP_SMALL, color='white',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460',
                      edgecolor='white', alpha=0.85),
            fontsize=9, zorder=10)

    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"静态图已保存至: {path}")


# ===========================================================================
#  7. 输出 GIF 动画
# ===========================================================================

def save_output_gif(path, map_cfg, obstacles, scenario, grid_map,
                    path_x, path_y, path_yaw, directions):
    """保存规划过程动画 GIF"""
    n = len(path_x)
    # 每隔几帧采样以控制帧数
    step = max(1, n // 80)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)

    frames = []
    tmp_dir = os.path.join(ROOT, 'output', '_tmp_frames')
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"正在生成动画帧 ({len(indices)} 帧)...")

    for frame_idx, i in enumerate(indices):
        fig, ax = plt.subplots(1, 1, figsize=(14, 9), dpi=100)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        setup_plot(ax, map_cfg, obstacles, scenario,
                   f'车辆轨迹规划动画 — 步骤 {i}/{n - 1}', grid_map)

        # 已走过的轨迹
        for j in range(i):
            color = '#00ff88' if directions[j] else '#ff6b6b'
            ax.plot(path_x[j:j + 2], path_y[j:j + 2], color=color,
                    linewidth=1.8, alpha=0.7, zorder=3)

        # 起点标记
        sx, sy = scenario['start']['x'], scenario['start']['y']
        syaw = math.radians(scenario['start']['yaw_deg'])
        ax.plot(sx, sy, 'o', color='#00b4d8', markersize=10, zorder=4)
        ax.text(sx + 1.5, sy + 1.5, '起点', fontproperties=FONT_PROP_SMALL,
                color='#00b4d8', fontsize=10, zorder=4)

        # 当前车辆
        draw_vehicle(ax, path_x[i], path_y[i], path_yaw[i],
                     color='#f72585' if not directions[i] else '#4cc9f0',
                     alpha=0.95)

        # 方向提示
        dir_text = '前进中' if directions[i] else '倒车中'
        dir_color = '#00ff88' if directions[i] else '#ff6b6b'
        ax.text(0.02, 0.02, dir_text, transform=ax.transAxes,
                fontproperties=FONT_PROP, color=dir_color,
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460',
                          edgecolor=dir_color, alpha=0.9),
                zorder=10)

        # 进度条
        progress = (i + 1) / n
        ax.text(0.98, 0.02, f'进度: {progress * 100:.0f}%',
                transform=ax.transAxes,
                fontproperties=FONT_PROP_SMALL, color='white',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.8),
                zorder=10)

        plt.tight_layout()
        frame_path = os.path.join(tmp_dir, f'frame_{frame_idx:04d}.png')
        fig.savefig(frame_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    # 最后一帧多停留
    for _ in range(10):
        frames.append(frames[-1])

    imageio.mimsave(path, frames, duration=0.12, loop=0)
    print(f"动画已保存至: {path}")

    # 清理临时帧
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ===========================================================================
#  主流程
# ===========================================================================

def main():
    print("=" * 60)
    print("  非结构道路车辆轨迹规划与泊车 Demo")
    print("  算法: 混合A* + Reeds-Shepp")
    print("=" * 60)

    input_dir = os.path.join(ROOT, 'input')
    output_dir = os.path.join(ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载输入
    print("\n[1/7] 加载输入数据...")
    map_cfg, obstacles, scenario = load_inputs(input_dir)

    start = [scenario['start']['x'],
             scenario['start']['y'],
             math.radians(scenario['start']['yaw_deg'])]
    goal = [scenario['goal']['x'],
            scenario['goal']['y'],
            math.radians(scenario['goal']['yaw_deg'])]

    print(f"  起点: ({start[0]:.1f}, {start[1]:.1f}), "
          f"航向: {scenario['start']['yaw_deg']}°")
    print(f"  终点: ({goal[0]:.1f}, {goal[1]:.1f}), "
          f"航向: {scenario['goal']['yaw_deg']}°")
    print(f"  停车位: {scenario['parking_slot']['type']}")

    # 2. 构建栅格地图
    print("\n[2/7] 构建栅格地图...")
    grid_map = build_grid_map(map_cfg, obstacles)

    # 3. 构建碰撞查找表
    print("\n[3/7] 构建碰撞查找表...")
    collision_lookup = CollisionLookup(
        mask=grid_map,
        resolution=map_cfg['resolution'],
        x_min=map_cfg['x_min'],
        y_min=map_cfg['y_min']
    )

    # 4. 规划
    print("\n[4/7] 运行混合A*路径规划...")
    config = Config(
        x_min=map_cfg['x_min'],
        y_min=map_cfg['y_min'],
        x_max=map_cfg['x_max'],
        y_max=map_cfg['y_max'],
        xy_resolution=2.0,
        yaw_resolution=np.deg2rad(5),
        motion_resolution=0.5,
        n_steer=15,
        grid_resolution=map_cfg['resolution']
    )

    path = hybrid_a_star_planning(
        start=start, goal=goal,
        collision_lookup=collision_lookup,
        config=config,
        grid_map=grid_map,
        grid_resolution=map_cfg['resolution'],
        x_min=map_cfg['x_min'],
        y_min=map_cfg['y_min'],
        use_dijkstra=True,
        rs_dist=15
    )

    if path is None:
        print("\n❌ 规划失败！无法找到从起点到终点的路径。")
        sys.exit(1)

    print(f"\n✅ 规划成功！路径点数: {len(path.xlist)}")

    # 保存原始轨迹用于对比
    orig_x = list(path.xlist)
    orig_y = list(path.ylist)

    # 5. 递归MPC轨迹优化
    print("\n[5/7] 递归MPC轨迹优化...")
    optimizer = RecursiveMPCOptimizer({
        'horizon': 40,
        'overlap': 15,
        'w_reference': 1.0,
        'w_smooth': 60.0,
        'w_jerk': 80.0,
        'w_obstacle': 500.0,
        'obstacle_margin': 2.0,
        'max_iter': 8,
        'fix_start_n': 5,
        'fix_end_n': 5,
    })
    opt_x, opt_y, opt_yaw = optimizer.optimize(
        path.xlist, path.ylist, path.yawlist, path.directionlist,
        grid_map, map_cfg
    )

    # 6. 计算曲率 & 输出
    print("\n[6/7] 保存结果...")
    curvatures = calculate_curvatures(opt_x, opt_y, opt_yaw)

    import planner.hybrid_a_star as ha
    save_output_json(
        os.path.join(output_dir, 'output.json'),
        opt_x, opt_y, opt_yaw, path.directionlist, curvatures,
        ha.LAST_ITER_NUM, ha.LAST_SEARCH_TIME
    )

    save_output_jpg(
        os.path.join(output_dir, 'output.jpg'),
        map_cfg, obstacles, scenario, grid_map,
        opt_x, opt_y, opt_yaw, path.directionlist,
        orig_path=(orig_x, orig_y)
    )

    # 7. GIF
    print("\n[7/7] 生成动画...")
    save_output_gif(
        os.path.join(output_dir, 'output.gif'),
        map_cfg, obstacles, scenario, grid_map,
        opt_x, opt_y, opt_yaw, path.directionlist
    )

    print("\n" + "=" * 60)
    print("  所有输出已保存至 output/ 目录:")
    print("    output/output.json  — 轨迹数据（优化后）")
    print("    output/output.jpg   — 静态规划图（含优化前后对比）")
    print("    output/output.gif   — 规划动画（优化后轨迹）")
    print("=" * 60)


if __name__ == '__main__':
    main()
