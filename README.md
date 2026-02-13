# 车辆轨迹规划与优化 Demo (Trajectory Planning & Optimization)

本项目演示了 **混合 A* (Hybrid A*)** 结合 **轨迹优化 (Trajectory Optimization)** 算法在非结构化场景下的应用。

## 核心算法

1.  **路径规划**：Hybrid A* + Reeds-Shepp Curves + 膨胀障碍物碰撞检测。
2.  **后处理优化**：
    *   **DL-IAPS**: 双层迭代锚点路径平滑 (基于 SCP)。
    *   **PJSO**: 分段 Jerk 速度规划 (优化 $s, v, a$)。

## 环境配置

### 1. 激活虚拟环境

项目已预配置 `.venv` 虚拟环境，请先激活：

```bash
# Linux / macOS
source .venv/bin/activate
```

### 2. 安装依赖

如果首次运行或依赖有更新，请安装：

```bash
pip install -r requirements.txt
pip install cvxpy  # 优化器核心依赖
```

## 运行 Demo

直接运行主程序即可：

```bash
python3 run_demo.py
```

程序将输出整个规划过程的日志信息。

## 结果查看

运行成功后，结果将保存在 `output/` 目录下：

*   `output.json`: 包含优化后的轨迹点坐标 $(x, y, \theta)$ 、速度 $v$、加速度 $a$ 及曲率 $\kappa$。
*   `output.jpg`: 静态可视化图，展示优化前后的轨迹对比以及速度/加速度曲线。
*   `output.gif`: 动态生成过程动画。

## 详细文档

*   [算法原理介绍](document/introduction.md)
*   [文件清单](document/文件清单.md)
