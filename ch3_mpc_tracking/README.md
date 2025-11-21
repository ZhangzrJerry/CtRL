# Project 3: MPC Tracking

项目中包括以下文件：

- `main.py`：主程序文件，包含仿真和可视化逻辑。
- `robot.py`: 定义了差分驱动机器人模型和运动函数。
- `mpc.py`: 你需要完成的模型预测控制 (MPC) 模块，实现`solve_mpc`函数。
- `visualization.py`：可视化模块，用于显示机器人轨迹和 MPC 预测。

## 运行项目

确保你已经安装了必要的依赖项，然后运行以下命令：

```bash
cd ch3_mpc_tracking
pip install -r requirements.txt
python main.py --debug
```

## 完成效果

[![MPC Tracking Visualization](./viz.mp4)](./viz.mp4)
