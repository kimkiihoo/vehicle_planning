# 算法原理介绍

本项目实现了一个非结构化环境下的车辆轨迹规划与优化系统。主要包含两个阶段：
1.  **初值规划**：使用 **改进的混合 A* (Hybrid A*)** 算法生成一条无碰撞、符合车辆运动学的粗糙路径。
2.  **后处理优化**：使用 DL-IAPS 和 PJSO 算法对路径进行平滑和速度规划，生成高质量的轨迹。

## 1. 改进的混合 A* (Hybrid A*)

混合 A* 是一种结合了传统 A* 搜索和车辆运动学模型的路径规划算法。针对传统方法容易产生的“绕路”和“蛇形”问题，我们引入了多项改进：

### 1.1 精细化代价函数 (Cost Function)
为了让路径平滑且趋向直行，我们将总代价设计为以下几部分的加权和：
*   **路径长度代价**：分离了几何距离与运动学代价。
*   **方向切换代价 (Gear Switch Cost)**：大幅增加换挡惩罚，抑制不必要的倒车。
*   **转向惩罚 (Steering Cost)**：惩罚大角度转向和转向角的剧烈变化，促进平滑过渡。
*   **直线偏离代价 (Straight Deviation Cost)**：新增项。惩罚车辆航向与目标方向的偏差，引导车辆尽量沿直指目标的走廊行驶，减少由于搜索空间过大导致的“漫游”。

### 1.2 增强启发式函数 (Enhanced Heuristic)
采用了组合启发式函数 `max(h_holonomic, h_non_holonomic)`：
*   **Holonomic Heuristic (2D Dijkstra)**：考虑障碍物的最短路距离，引导避障。
*   **Non-holonomic Heuristic (Reeds-Shepp)**：引入 Reeds-Shepp 曲线长度作为启发值。相比传统的欧氏距离，它显式考虑了车辆的最小转弯半径，提供了更紧致（Tighter）的下界估计，大幅减少了无效节点的扩展，引导车辆更早调整姿态对准目标。

### 1.3 解析扩展 (Analytic Expansion)
在搜索过程中，定期尝试使用 Reeds-Shepp 曲线直接连接当前点和终点，一旦无碰撞即提前终止搜索，提高效率并直接获得平滑的末端路径。

## 2. 轨迹优化 (Trajectory Optimization)

为了让生成的轨迹更加平滑且舒适，我们引入了基于数值优化的后处理模块 `TrajectoryOptimizer`。

### 2.1 路径平滑 (DL-IAPS)

**DL-IAPS (Dual-Loop Iterative Anchoring Path Smoothing)** 是一种双层循环的路径平滑算法。
-   **内层循环**：构建 QP 问题最小化路径曲率，约束在安全框 (Safety Box) 内。
-   **外层循环**：碰撞检测与安全框调整。

### 2.2 速度规划 (PJSO)

**PJSO (Piece-wise Jerk Speed Optimization)** 用于在已知几何路径的基础上生成最优速度曲线。
-   **目标**：最小化加加速度 (Jerk) 和行程时间。
-   **约束**：动力学约束、最大速度/加速度、曲率限速。
