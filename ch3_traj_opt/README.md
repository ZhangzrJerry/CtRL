# Chapter 3: 轨迹优化

项目中包括以下文件

- `main.py`：主程序，设置初始和目标状态，调用优化器并可视化结果。
- `visualization.py`：可视化模块，用于显示机器人轨迹和障碍物。
- `rrt.py`：你需要完成的基于 RRT 的路径规划模块，实现 `rrt_planner` 函数。
- `traj_opt.py`：你需要完成的轨迹优化模块，实现 `optimize_trajectory` 函数。

## 运行项目

```bash
cd ch4_traj_opt
pip install -r requirements.txt
python main.py --debug
```

如果不带 `--debug` 参数，程序会尝试将动画保存为 `traj_viz.mp4`（需要 `ffmpeg`）。

## 完成效果

[![Trajectory Optimization Visualization](./viz.mp4)](./viz.mp4)
