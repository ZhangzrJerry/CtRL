# Chapter 4 — 移动机器人最优轨迹生成与可视化

这是第 4 章的示例：一个简单的基于直接离散化的轨迹优化器（差分驱动移动机器人）以及可视化工具。

文件说明：

- `traj_opt.py`：轨迹优化实现（使用 `scipy.optimize.minimize` 对离散控制序列优化）。
- `visualization.py`：动画可视化，显示规划路径和机器人轨迹。
- `main.py`：示例运行脚本，默认运行优化并弹出动画窗口（或保存到 `traj_viz.mp4`）。
- `requirements.txt`：所需 Python 包。

运行方法：

```powershell
cd ch4_opt_traj_mobile
pip install -r requirements.txt
python .\main.py --debug
```

如果不带 `--debug` 参数，程序会尝试将动画保存为 `traj_viz.mp4`（需要 `ffmpeg`）。

说明：

此处实现为教学用途：优化通过最小化终端位置误差与控制能量的加权和来获得一组速度和角速度序列。你可以把它改成带约束或更复杂的代价函数（例如避障、通过路点等）。
