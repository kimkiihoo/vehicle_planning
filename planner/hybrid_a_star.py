
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
SB_COST = 300.0        # 切换方向惩罚
BACK_COST = 50.0       # 倒车惩罚
STEER_CHANGE_COST = 50.0  # 转向角变化惩罚
STEER_COST = 15.0      # 转向角惩罚
H_COST = 18.0          # 启发式权重
MAX_OBSTACLE_COST = 1000.0
DECAY_RATE = 6.0
OBSTACLE_COST = 20.0
STRAIGHT_COST = 5.0    # 偏离直线惩罚

# 记录最近一次搜索的统计信息
LAST_ITER_NUM = 0
LAST_SEARCH_TIME = 0.0
LAST_PATH_LENGTH = 0.0

# 优化参数
COARSE_GRID_RES = 1.0 # 粗栅格分辨率 (1.0m) 用于Dijkstra启发式
RS_HEURISTIC_DIST = 20.0 # 只有距离目标小于此值时才启用RS启发式

class Config:
    """规划器配置"""
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
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directionlist = [bool(d) for d in directionlist]
        self.cost = cost


def generate_obstacle_cost_map(grid_map, config):
    """
    生成障碍物代价地图 (Fine Grid)
    """
    if grid_map is None:
        return None

    height, width = grid_map.shape
    cost_map = np.zeros((height, width), dtype=np.float32)

    queue = deque()
    visited = set()

    # 标记障碍物
    obstacles = np.where(grid_map == 0)
    cost_map[obstacles] = MAX_OBSTACLE_COST
    for r, c in zip(obstacles[0], obstacles[1]):
        queue.append((c, r, 0))
        visited.add((c, r))

    # BFS
    while queue:
        px, py, dist = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if grid_map[ny, nx] != 0:
                    distance_m = (dist + 1) * config.grid_resolution
                    if distance_m > 4.0: 
                        continue
                    calculated_cost = MAX_OBSTACLE_COST * math.exp(-distance_m / DECAY_RATE)
                    cost_map[ny, nx] = calculated_cost
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
    return cost_map


def precompute_collision_map(config, grid_map, grid_resolution, x_min, y_min):
    """预计算碰撞地图 (Hybrid A* Grid Resolution)"""
    width = config.maxx - config.minx + 1
    height = config.maxy - config.miny + 1
    collision_map = np.zeros((width, height), dtype=np.bool_)
    
    x_indices = np.arange(config.minx, config.maxx + 1)
    y_indices = np.arange(config.miny, config.maxy + 1)
    
    for xi, xind in enumerate(x_indices):
        for yi, yind in enumerate(y_indices):
            x = xind * config.xy_resolution
            y = yind * config.xy_resolution
            px = int((x - x_min) / grid_resolution)
            py = int((y - y_min) / grid_resolution)
            if 0 <= px < grid_map.shape[1] and 0 <= py < grid_map.shape[0]:
                if grid_map[py, px] == 0:
                     collision_map[xi, yi] = True
    return collision_map


def dijkstra_distance_map(goal, config, grid_map, grid_resolution, x_min, y_min):
    """
    使用粗栅格 (Coarse Grid) 计算Dijkstra启发式
    """
    start_time = time.time()
    
    # 1. 降采样地图
    scale = COARSE_GRID_RES / grid_resolution
    coarse_h = int(grid_map.shape[0] / scale)
    coarse_w = int(grid_map.shape[1] / scale)
    
    coarse_map = np.zeros((coarse_h, coarse_w), dtype=np.int8) 
    
    obs_y, obs_x = np.where(grid_map == 0)
    coarse_obs_y = (obs_y / scale).astype(int)
    coarse_obs_x = (obs_x / scale).astype(int)
    
    valid_mask = (coarse_obs_x >= 0) & (coarse_obs_x < coarse_w) & \
                 (coarse_obs_y >= 0) & (coarse_obs_y < coarse_h)
    
    coarse_map.fill(1) 
    coarse_map[coarse_obs_y[valid_mask], coarse_obs_x[valid_mask]] = 0 
    
    # 2. 运行 Dijkstra
    cg_goal_x = int((goal[0] - x_min) / COARSE_GRID_RES)
    cg_goal_y = int((goal[1] - y_min) / COARSE_GRID_RES)
    
    if 0 <= cg_goal_x < coarse_w and 0 <= cg_goal_y < coarse_h:
         if coarse_map[cg_goal_y, cg_goal_x] == 0:
             print("[Dijkstra] 警告：目标点在粗栅格障碍物上，尝试寻找最近空闲点...")
             found = False
             for dx in range(-2, 3):
                 for dy in range(-2, 3):
                     nx, ny = cg_goal_x + dx, cg_goal_y + dy
                     if 0 <= nx < coarse_w and 0 <= ny < coarse_h and coarse_map[ny, nx] == 1:
                         cg_goal_x, cg_goal_y = nx, ny
                         found = True
                         break
                 if found: break
    
    dist_map = np.full((coarse_h, coarse_w), np.inf)
    dist_map[cg_goal_y, cg_goal_x] = 0
    
    pq = [(0, cg_goal_x, cg_goal_y)]
    
    motions = [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
               (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    
    while pq:
        d, cx, cy = heapq.heappop(pq)
        if d > dist_map[cy, cx]: continue
        
        for dx, dy, cost_mult in motions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < coarse_w and 0 <= ny < coarse_h:
                if coarse_map[ny, nx] == 1:
                    new_dist = d + COARSE_GRID_RES * cost_mult
                    if new_dist < dist_map[ny, nx]:
                        dist_map[ny, nx] = new_dist
                        heapq.heappush(pq, (new_dist, nx, ny))
    
    end_time = time.time()
    print(f"[Dijkstra] Coarse Grid ({coarse_w}x{coarse_h}) 计算用时: {end_time - start_time:.2f}秒")
    return dist_map


def calc_motion_inputs(config):
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER, config.n_steer), [0.0])):
        yield [steer, True]
        yield [steer, False]


def check_car_collision(xlist, ylist, yawlist, collision_lookup):
    for x, y, yaw in zip(xlist, ylist, yawlist):
        if collision_lookup.collision_detection(x, y, yaw):
            return True
    return False


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)
    return x, y, yaw


def get_neighbors(current, config, collision_lookup, cost_map, x_min, y_min, grid_res):
    """Adaptive Step Size"""
    current_obs_cost = 0
    if cost_map is not None:
        px = int((current.xlist[-1] - x_min) / grid_res)
        py = int((current.ylist[-1] - y_min) / grid_res)
        if 0 <= px < cost_map.shape[1] and 0 <= py < cost_map.shape[0]:
            current_obs_cost = cost_map[py, px]
            
    step_scale = 1.0
    if current_obs_cost < 50.0: 
        step_scale = 1.6
    elif current_obs_cost < 200.0:
        step_scale = 1.2
        
    for steer, d in calc_motion_inputs(config):
        node = calc_next_node(current, steer, d, config, collision_lookup, step_scale)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, collision_lookup, step_scale=1.0):
    x, y, yaw = current.xlist[-1], current.ylist[-1], current.yawlist[-1]
    
    arc_l = config.xy_resolution * 1.5 * step_scale
    
    xlist, ylist, yawlist, directions = [], [], [], []
    steps = max(2, int(arc_l / config.motion_resolution))
    dt = arc_l / steps
    
    for _ in range(steps):
        nx, ny, nyaw = move(x, y, yaw, dt * (1 if direction else -1), steer)
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
    
    dist_cost = arc_l if direction else arc_l * 1.2
    
    cost = current.cost + addedcost + dist_cost

    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, directions,
                pind=calc_index(current, config), cost=cost, steer=steer)
    return node


def is_same_grid(n1, n2):
    return n1.xind == n2.xind and n1.yind == n2.yind and n1.yawind == n2.yawind


def analytic_expansion(current, goal, config, collision_lookup):
    sx, sy, syaw = current.xlist[-1], current.ylist[-1], current.yawlist[-1]
    gx, gy, gyaw = goal.xlist[-1], goal.ylist[-1], goal.yawlist[-1]
    max_curvature = math.tan(MAX_STEER) / WB
    
    dist = math.hypot(sx - gx, sy - gy)
    if dist > 30.0: return None

    paths = rs.calc_paths(sx, sy, syaw, gx, gy, gyaw, max_curvature, step_size=config.motion_resolution)
    if not paths: return None

    best_path, best = None, None
    for path in paths:
        if not check_car_collision(path.x, path.y, path.yaw, collision_lookup):
            l_back = sum(abs(l) for l in path.lengths if l < 0)
            b_num = sum(1 for l in path.lengths if l < 0)
            cost = calc_rs_path_cost(path) + b_num * 100000
            if not best or best > cost:
                best = cost
                best_path = path
    return best_path


def update_node_with_analystic_expantion(current, goal, config, collision_lookup):
    apath = analytic_expansion(current, goal, config, collision_lookup)
    if apath:
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw = apath.yaw[1:]
        fcost = current.cost + calc_rs_path_cost(apath)
        fpind = calc_index(current, config)
        fd = [bool(d) for d in apath.directions[1:]]
        if len(fd) != len(fx): fd = fd[:len(fx)]
        fsteer = 0.0
        fpath = Node(current.xind, current.yind, current.yawind,
                     current.direction, fx, fy, fyaw, fd,
                     cost=fcost, pind=fpind, steer=fsteer)
        return True, fpath
    return False, None


def calc_rs_path_cost(rspath):
    cost = 0.0
    for l, d in zip(rspath.lengths, rspath.directions):
        if d: cost += l
        else: cost += abs(l) * (BACK_COST + 5.0)
    for i in range(len(rspath.lengths) - 1):
        if rspath.directions[i] != rspath.directions[i + 1]: cost += SB_COST
    for ctype in rspath.ctypes:
        if ctype != "S": cost += STEER_COST * abs(MAX_STEER)
    nctypes = len(rspath.ctypes)
    ulist = [0.0] * nctypes
    for i in range(nctypes):
        if rspath.ctypes[i] == "R": ulist[i] = -MAX_STEER
        elif rspath.ctypes[i] == "L": ulist[i] = MAX_STEER
    for i in range(len(rspath.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])
    return cost


def hybrid_a_star_planning(start, goal, collision_lookup, config,
                           grid_map=None, grid_resolution=0.1,
                           x_min=0.0, y_min=0.0,
                           use_dijkstra=True, rs_dist=44):
    print("开始混合A*路径规划!")
    t0 = time.time()

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])

    cost_map = None
    dist_map = {} # Coarse grid dijkstra map
    
    if use_dijkstra and grid_map is not None:
        cost_map = generate_obstacle_cost_map(grid_map, config)
        dist_map = dijkstra_distance_map(goal, config, grid_map, grid_resolution, x_min, y_min)

    nstart = Node(round(start[0] / config.xy_resolution), round(start[1] / config.xy_resolution),
                  round(start[2] / config.yaw_resolution), True, [start[0]], [start[1]], [start[2]], [True], cost=0)
    ngoal = Node(round(goal[0] / config.xy_resolution), round(goal[1] / config.xy_resolution),
                 round(goal[2] / config.yaw_resolution), True, [goal[0]], [goal[1]], [goal[2]], [True])
    
    goal_pose = goal
    openList, closedList = {}, {}
    pq = []
    openList[calc_index(nstart, config)] = nstart
    
    heapq.heappush(pq, (calc_cost(nstart, goal_pose, dist_map, config, cost_map, grid_resolution, x_min, y_min),
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

        dist = math.hypot(current.xlist[-1] - goal[0], current.ylist[-1] - goal[1])

        if dist < rs_dist:
            isupdated, fpath = update_node_with_analystic_expantion(
                current, ngoal, config, collision_lookup)
            if isupdated:
                print("成功使用Reeds-Shepp曲线连接到目标!")
                break

        for neighbor in get_neighbors(current, config, collision_lookup, cost_map, x_min, y_min, grid_resolution):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor_index not in openList or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(pq,
                               (calc_cost(neighbor, goal_pose, dist_map, config, cost_map, grid_resolution, x_min, y_min),
                                neighbor_index))
                openList[neighbor_index] = neighbor

        if iter_num > 100000:
            print("无法找到路径，超过迭代限制!")
            return None

    path = get_final_path(closedList, fpath, nstart, config)
    search_time = time.time() - t0
    
    path_len = 0.0
    for i in range(len(path.xlist) - 1):
        path_len += math.hypot(path.xlist[i+1]-path.xlist[i], path.ylist[i+1]-path.ylist[i])
    
    global LAST_ITER_NUM, LAST_SEARCH_TIME, LAST_PATH_LENGTH
    LAST_ITER_NUM = iter_num
    LAST_SEARCH_TIME = search_time
    LAST_PATH_LENGTH = path_len
    
    print(f"混合A*迭代次数：{iter_num}次，用时：{search_time:.2f}秒")
    print(f"规划后路径长度: {path_len:.2f} m")

    return path


def calc_cost(n, goal, dist_map, config, cost_map=None,
              grid_resolution=0.1, x_min=0.0, y_min=0.0, is_start=False):
    current_x = n.xlist[-1]
    current_y = n.ylist[-1]
    current_yaw = n.yawlist[-1]

    h_dijkstra = 0.0
    if dist_map is not None:
        cg_x = int((current_x - x_min) / COARSE_GRID_RES)
        cg_y = int((current_y - y_min) / COARSE_GRID_RES)
        if 0 <= cg_y < dist_map.shape[0] and 0 <= cg_x < dist_map.shape[1]:
             val = dist_map[cg_y, cg_x]
             if val != np.inf: h_dijkstra = val
             else: h_dijkstra = math.hypot(current_x - goal[0], current_y - goal[1])
        else: h_dijkstra = math.hypot(current_x - goal[0], current_y - goal[1])
    else: h_dijkstra = math.hypot(current_x - goal[0], current_y - goal[1])
        
    h_rs = 0.0
    if h_dijkstra < RS_HEURISTIC_DIST:
        max_curvature = math.tan(MAX_STEER) / WB
        qs = [current_x, current_y, current_yaw]
        qg = [goal[0], goal[1], goal[2]]
        rs_paths = rs.generate_path(qs, qg, max_curvature)
        if rs_paths: h_rs = min([p.L for p in rs_paths])
        else: h_rs = h_dijkstra
    else:
        h_rs = h_dijkstra

    h_cost = max(h_dijkstra, h_rs)
    
    obstacle_cost = 0.0
    if cost_map is not None:
        px = int((current_x - x_min) / grid_resolution)
        py = int((current_y - y_min) / grid_resolution)
        if 0 <= px < cost_map.shape[1] and 0 <= py < cost_map.shape[0]:
            obstacle_cost = cost_map[py, px]

    goal_angle = math.atan2(goal[1] - current_y, goal[0] - current_x)
    angle_diff = abs(rs.pi_2_pi(goal_angle - current_yaw))
    straight_cost = STRAIGHT_COST * angle_diff

    return n.cost + H_COST * h_cost + OBSTACLE_COST * obstacle_cost + straight_cost

def get_final_path(closed, ngoal, nstart, config):
    rx, ry, ryaw = list(reversed(ngoal.xlist)), list(reversed(ngoal.ylist)), list(reversed(ngoal.yawlist))
    direction = list(reversed(ngoal.directions))
    nid = ngoal.pind
    finalcost = ngoal.cost
    while nid:
        n = closed[nid]
        if len(n.xlist) != len(n.directions): n.directions = n.directions[:len(n.xlist)]
        rx.extend(list(reversed(n.xlist)))
        ry.extend(list(reversed(n.ylist)))
        ryaw.extend(list(reversed(n.yawlist)))
        direction.extend(list(reversed(n.directions)))
        nid = n.pind
    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    direction = list(reversed(direction))
    if len(direction) != len(rx): direction = direction[:len(rx)]
    if len(direction) > 0: direction[0] = nstart.directions[0]
    path = Path(rx, ry, ryaw, direction, finalcost)
    return path

def verify_index(node, config):
    xind, yind, yawind = node.xind, node.yind, node.yawind
    return (config.minx <= xind <= config.maxx and
            config.miny <= yind <= config.maxy and
            config.minyaw <= yawind <= config.maxyaw)

def calc_index(node, config):
    ind = ((node.yawind - config.minyaw) * config.xw * config.yw) + \
          ((node.yind - config.miny) * config.xw) + \
          (node.xind - config.minx)
    if ind < 0: return 0
    return ind