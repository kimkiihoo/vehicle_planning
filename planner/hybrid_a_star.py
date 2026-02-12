# 内置库
import heapq
import math
from math import sqrt, cos, sin, tan, pi
from collections import deque
import time

# 第三方库
import numpy as np

try:
    from . import reeds_shepp_path_planning as rs
except ImportError:
    import reeds_shepp_path_planning as rs

# 车辆参数
WB = 3.5  # 轴距
W = 3.2  # 车辆宽度
LF = 4.51  # 后轴到车头的距离
LB = 1.01  # 后轴到车尾的距离
MAX_STEER = np.deg2rad(34)  # 最大转向角 [rad]

# 规划成本参数
SB_COST = 150.0  # 切换方向惩罚（高值避免频繁换向）
BACK_COST = 500.0  # 倒车惩罚（高值使搜索阶段尽量避免倒车）
STEER_CHANGE_COST = 10.0  # 转向角变化惩罚
STEER_COST = 25.0  # 转向角惩罚
H_COST = 250.0  # 启发式成本（泊车场景）
MAX_OBSTACLE_COST = 1000.0  # 障碍物最大代价
DECAY_RATE = 6.0  # 代价衰减率（米）
OBSTACLE_COST = 20.0  # 障碍物代价权重

# 记录最近一次搜索的统计信息（供外部读取）
LAST_ITER_NUM = 0
LAST_SEARCH_TIME = 0.0


class Config:
    """规划器配置（直接接受地图边界参数）"""
    def __init__(self, x_min, y_min, x_max, y_max,
                 xy_resolution=1.0, yaw_resolution=np.deg2rad(5),
                 motion_resolution=0.3, n_steer=11, grid_resolution=0.1):
        self.xy_resolution = xy_resolution
        self.yaw_resolution = yaw_resolution
        self.motion_resolution = motion_resolution
        self.n_steer = n_steer
        self.grid_resolution = grid_resolution

        self.minx = round(x_min / xy_resolution)
        self.miny = round(y_min / xy_resolution)
        self.maxx = round(x_max / xy_resolution)
        self.maxy = round(y_max / xy_resolution)

        self.xw = round(self.maxx - self.minx)
        self.yw = round(self.maxy - self.miny)

        self.minyaw = round(-180 / yaw_resolution) - 1
        self.maxyaw = round(180 / yaw_resolution)
        self.yaww = round(self.maxyaw - self.minyaw)


class Node:
    def __init__(self, xind, yind, yawind, direction, xlist, ylist, yawlist, directions,
                 steer=0.0, pind=None, cost=None):
        if len(xlist) != len(directions):
            raise ValueError(f"Directions length {len(directions)} does not match xlist length {len(xlist)}")
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = bool(direction)
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directions = [bool(d) for d in directions]
        self.steer = steer
        self.pind = pind
        self.cost = cost if cost is not None else float('inf')


class Path:
    def __init__(self, xlist, ylist, yawlist, directionlist, cost):
        if len(xlist) != len(directionlist):
            raise ValueError(f"Directionlist length {len(directionlist)} does not match xlist length {len(xlist)}")
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directionlist = [bool(d) for d in directionlist]
        self.cost = cost


def generate_obstacle_cost_map(grid_map, config):
    """生成障碍物代价地图
    
    Args:
        grid_map: numpy 数组, 0=障碍, 1=空闲
        config: Config 对象
    """
    if grid_map is None:
        return None

    height, width = grid_map.shape
    cost_map = np.zeros((height, width), dtype=np.float32)

    queue = deque()
    visited = set()

    # 标记所有障碍物像素
    for py in range(height):
        for px in range(width):
            if grid_map[py, px] == 0:
                cost_map[py, px] = MAX_OBSTACLE_COST
                queue.append((px, py, 0))
                visited.add((px, py))

    # 使用BFS计算距离衰减的代价
    while queue:
        px, py, dist = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if grid_map[ny, nx] != 0:
                    distance_m = (dist + 1) * config.grid_resolution
                    calculated_cost = MAX_OBSTACLE_COST * math.exp(-distance_m / DECAY_RATE)
                    cost_map[ny, nx] = calculated_cost
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))

    return cost_map


def calc_motion_inputs(config):
    """生成运动输入（转向角），包含前进和倒车"""
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER, config.n_steer), [0.0])):
        yield [steer, True]   # 前进
        yield [steer, False]  # 倒车


def check_car_collision(xlist, ylist, yawlist, collision_lookup):
    """
    对连续轨迹点做碰撞检测，返回 True 表示有碰撞
    """
    for x, y, yaw in zip(xlist, ylist, yawlist):
        if collision_lookup.collision_detection(x, y, yaw):
            return True
    return False


def precompute_collision_map(config, grid_map, grid_resolution, x_min, y_min):
    """预计算碰撞地图"""
    width = config.maxx - config.minx + 1
    height = config.maxy - config.miny + 1
    collision_map = np.zeros((width, height), dtype=np.bool_)

    obstacle_map = (grid_map == 0)

    for xind in range(config.minx, config.maxx + 1):
        for yind in range(config.miny, config.maxy + 1):
            x = xind * config.xy_resolution
            y = yind * config.xy_resolution
            px = math.floor((x - x_min) / grid_resolution)
            py = math.floor((y - y_min) / grid_resolution)
            if 0 <= px < grid_map.shape[1] and 0 <= py < grid_map.shape[0]:
                collision_map[xind - config.minx, yind - config.miny] = obstacle_map[py, px]

    return collision_map


def dijkstra_distance_map(goal, config, grid_map, grid_resolution, x_min, y_min):
    """使用Dijkstra算法计算到目标的距离地图"""
    start_time = time.time()
    gx = round(goal[0] / config.xy_resolution)
    gy = round(goal[1] / config.xy_resolution)

    collision_map = precompute_collision_map(config, grid_map, grid_resolution, x_min, y_min)
    if collision_map[gx - config.minx, gy - config.miny]:
        print("[Dijkstra] 警告：目标点在障碍物上！")
        return {}

    width = config.maxx - config.minx + 1
    height = config.maxy - config.miny + 1
    dist_map_array = np.full((width, height), np.inf)
    dist_map_array[gx - config.minx, gy - config.miny] = 0

    max_dist = 600.0
    pq = [(0, (gx, gy))]
    visited = set()

    while pq:
        dist, (xind, yind) = heapq.heappop(pq)
        if (xind, yind) in visited:
            continue
        visited.add((xind, yind))
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = xind + dx, yind + dy
            if not (config.minx <= nx <= config.maxx and config.miny <= ny <= config.maxy):
                continue
            if collision_map[nx - config.minx, ny - config.miny]:
                continue
            new_dist = dist + config.xy_resolution
            if new_dist > max_dist:
                continue
            if new_dist < dist_map_array[nx - config.minx, ny - config.miny]:
                dist_map_array[nx - config.minx, ny - config.miny] = new_dist
                heapq.heappush(pq, (new_dist, (nx, ny)))

    dist_map = {}
    for xind in range(config.minx, config.maxx + 1):
        for yind in range(config.miny, config.maxy + 1):
            dist = dist_map_array[xind - config.minx, yind - config.miny]
            if dist != np.inf:
                dist_map[(xind, yind)] = dist

    end_time = time.time()
    print(f"[Dijkstra] 计算用时: {end_time - start_time:.2f}秒")
    return dist_map


def pi_2_pi(angle):
    """将角度归一化到[-pi, pi]"""
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    """车辆运动模型"""
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)
    return x, y, yaw


def get_neighbors(current, config, collision_lookup):
    """获取当前节点的邻居节点"""
    for steer, d in calc_motion_inputs(config):
        node = calc_next_node(current, steer, d, config, collision_lookup)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, collision_lookup):
    """计算下一个节点"""
    x, y, yaw = current.xlist[-1], current.ylist[-1], current.yawlist[-1]
    arc_l = config.xy_resolution * 1.5
    xlist, ylist, yawlist, directions = [], [], [], []

    for dist in np.arange(0, arc_l, config.motion_resolution):
        nx, ny, nyaw = move(x, y, yaw, config.motion_resolution * (1 if direction else -1), steer)
        xlist.append(nx)
        ylist.append(ny)
        yawlist.append(nyaw)
        directions.append(bool(direction))
        x, y, yaw = nx, ny, nyaw

    if check_car_collision(xlist, ylist, yawlist, collision_lookup):
        return None

    xind = round(x / config.xy_resolution)
    yind = round(y / config.xy_resolution)
    yawind = round(yaw / config.yaw_resolution)

    addedcost = 0.0
    if direction != current.direction:
        addedcost += SB_COST
    if not direction:
        addedcost += BACK_COST
    addedcost += STEER_COST * abs(steer)
    addedcost += STEER_CHANGE_COST * abs(current.steer - steer)
    cost = current.cost + addedcost + arc_l

    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, directions,
                pind=calc_index(current, config), cost=cost, steer=steer)
    return node


def is_same_grid(n1, n2):
    """检查两个节点是否在同一网格"""
    return n1.xind == n2.xind and n1.yind == n2.yind and n1.yawind == n2.yawind


def analytic_expansion(current, goal, config, collision_lookup):
    """使用Reeds-Shepp曲线进行解析扩展"""
    sx, sy, syaw = current.xlist[-1], current.ylist[-1], current.yawlist[-1]
    gx, gy, gyaw = goal.xlist[-1], goal.ylist[-1], goal.yawlist[-1]

    # 泊车场景使用最大曲率
    max_curvature = math.tan(MAX_STEER) / WB

    paths = rs.calc_paths(sx, sy, syaw, gx, gy, gyaw, max_curvature, step_size=config.motion_resolution)
    if not paths:
        return None

    best_path, best = None, None
    for path in paths:
        if not check_car_collision(path.x, path.y, path.yaw, collision_lookup):
            # 泊车场景的路径选择标准
            l_back = sum(abs(l) for l in path.lengths if l < 0)
            b_num = sum(1 for l in path.lengths if l < 0)

            cost = calc_rs_path_cost(path) + b_num * 100000
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analystic_expantion(current, goal, config, collision_lookup):
    """使用解析扩展更新节点"""
    apath = analytic_expansion(current, goal, config, collision_lookup)
    if apath:
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw = apath.yaw[1:]
        fcost = current.cost + calc_rs_path_cost(apath)
        fpind = calc_index(current, config)
        fd = [bool(d) for d in apath.directions[1:]]
        if len(fd) != len(fx):
            fd = fd[:len(fx)]
        fsteer = 0.0
        fpath = Node(current.xind, current.yind, current.yawind,
                     current.direction, fx, fy, fyaw, fd,
                     cost=fcost, pind=fpind, steer=fsteer)
        return True, fpath
    return False, None


def calc_rs_path_cost(rspath):
    """计算Reeds-Shepp路径的代价"""
    cost = 0.0
    # 基本路径长度代价
    for l, d in zip(rspath.lengths, rspath.directions):
        if d:
            cost += l
        else:
            # 泊车场景的倒车惩罚
            cost += abs(l) * (BACK_COST + 5.0)

    # 方向切换代价
    for i in range(len(rspath.lengths) - 1):
        if rspath.directions[i] != rspath.directions[i + 1]:
            cost += SB_COST

    # 转向代价
    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += STEER_COST * abs(MAX_STEER)

    # 转向变化代价
    nctypes = len(rspath.ctypes)
    ulist = [0.0] * nctypes
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def hybrid_a_star_planning(start, goal, collision_lookup, config,
                           grid_map=None, grid_resolution=0.1,
                           x_min=0.0, y_min=0.0,
                           use_dijkstra=True, rs_dist=44):
    """混合A*路径规划主函数
    
    Args:
        start: [x, y, yaw] 起点
        goal: [x, y, yaw] 终点
        collision_lookup: CollisionLookup 对象
        config: Config 对象
        grid_map: 栅格地图 numpy 数组 (0=障碍, 1=空闲)
        grid_resolution: 栅格分辨率(米)
        x_min, y_min: 栅格原点对应的世界坐标
        use_dijkstra: 是否使用Dijkstra启发式
        rs_dist: Reeds-Shepp连接距离阈值
    
    Returns:
        Path 对象或 None
    """
    print("开始混合A*路径规划!")
    from time import time as _time
    t0 = _time()

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])

    cost_map = None
    dist_map = {}
    if use_dijkstra and grid_map is not None:
        cost_map = generate_obstacle_cost_map(grid_map, config)
        dist_map = dijkstra_distance_map(goal, config, grid_map, grid_resolution, x_min, y_min)

    # 初始化起始和目标节点
    nstart = Node(round(start[0] / config.xy_resolution), round(start[1] / config.xy_resolution),
                  round(start[2] / config.yaw_resolution), True, [start[0]], [start[1]], [start[2]], [True], cost=0)
    ngoal = Node(round(goal[0] / config.xy_resolution), round(goal[1] / config.xy_resolution),
                 round(goal[2] / config.yaw_resolution), True, [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}
    pq = []
    openList[calc_index(nstart, config)] = nstart
    heapq.heappush(pq, (calc_cost(nstart, goal, dist_map, config, cost_map, grid_resolution, x_min, y_min),
                        calc_index(nstart, config)))

    iter_num = 0
    fpath = None
    while True:
        iter_num += 1

        if not openList:
            print("无法找到路径，开放列表为空!")
            return None

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        # 计算到目标的距离
        if use_dijkstra and dist_map:
            dist = dist_map.get((current.xind, current.yind), float('inf'))
        else:
            dist = sqrt((current.xlist[-1] - goal[0]) ** 2 + (current.ylist[-1] - goal[1]) ** 2)

        # 尝试使用Reeds-Shepp曲线连接到目标
        if dist < rs_dist:
            isupdated, fpath = update_node_with_analystic_expantion(
                current, ngoal, config, collision_lookup)
            if isupdated:
                print("成功使用Reeds-Shepp曲线连接到目标!")
                break

        # 扩展邻居节点
        for neighbor in get_neighbors(current, config, collision_lookup):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor_index not in openList or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(pq,
                               (calc_cost(neighbor, goal, dist_map, config, cost_map, grid_resolution, x_min, y_min),
                                neighbor_index))
                openList[neighbor_index] = neighbor

        if iter_num > 120000:
            print("无法找到路径，超过迭代限制!")
            return None

    path = get_final_path(closedList, fpath, nstart, config)
    search_time = _time() - t0
    global LAST_ITER_NUM, LAST_SEARCH_TIME
    LAST_ITER_NUM = iter_num
    LAST_SEARCH_TIME = search_time
    print(f"混合A*迭代次数：{iter_num}次，用时：{search_time:.2f}秒")

    return path


def calc_cost(n, goal, dist_map, config, cost_map=None,
              grid_resolution=0.1, x_min=0.0, y_min=0.0):
    """计算节点的总代价"""
    xind, yind = n.xind, n.yind

    # 启发式代价
    if dist_map:
        h_cost = dist_map.get((xind, yind), sqrt((n.xlist[-1] - goal[0]) ** 2 + (n.ylist[-1] - goal[1]) ** 2))
    else:
        h_cost = sqrt((n.xlist[-1] - goal[0]) ** 2 + (n.ylist[-1] - goal[1]) ** 2)

    # 障碍物代价
    obstacle_cost = 0.0
    if cost_map is not None:
        x = xind * config.xy_resolution
        y = yind * config.xy_resolution
        px = math.floor((x - x_min) / grid_resolution)
        py = math.floor((y - y_min) / grid_resolution)
        if 0 <= px < cost_map.shape[1] and 0 <= py < cost_map.shape[0]:
            obstacle_cost = cost_map[py, px]

    return n.cost + H_COST * h_cost + OBSTACLE_COST * obstacle_cost


def get_final_path(closed, ngoal, nstart, config):
    """从闭集中重构最终路径"""
    rx, ry, ryaw = list(reversed(ngoal.xlist)), list(reversed(ngoal.ylist)), list(reversed(ngoal.yawlist))
    direction = list(reversed(ngoal.directions))
    nid = ngoal.pind
    finalcost = ngoal.cost

    while nid:
        n = closed[nid]
        if len(n.xlist) != len(n.directions):
            n.directions = n.directions[:len(n.xlist)]
        rx.extend(list(reversed(n.xlist)))
        ry.extend(list(reversed(n.ylist)))
        ryaw.extend(list(reversed(n.yawlist)))
        direction.extend(list(reversed(n.directions)))
        nid = n.pind

    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    direction = list(reversed(direction))

    if len(direction) != len(rx):
        direction = direction[:len(rx)]
    if len(direction) > 0:
        direction[0] = nstart.directions[0]

    path = Path(rx, ry, ryaw, direction, finalcost)
    return path


def verify_index(node, config):
    """验证节点索引是否在有效范围内"""
    xind, yind, yawind = node.xind, node.yind, node.yawind
    return (config.minx <= xind <= config.maxx and
            config.miny <= yind <= config.maxy and
            config.minyaw <= yawind <= config.maxyaw)


def calc_index(node, config):
    """计算节点的唯一索引"""
    ind = ((node.yawind - config.minyaw) * config.xw * config.yw) + \
          ((node.yind - config.miny) * config.xw) + \
          (node.xind - config.minx)
    if ind < 0:
        return 0
    return ind