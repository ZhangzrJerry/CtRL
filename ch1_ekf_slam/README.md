# Project 1: EKF SLAM

项目中有三个文件：

- `main.py`：主程序文件，包含可视化和主循环逻辑。
- `ekf_slam.py`：EKF SLAM 算法的实现，需要你完成`predict`和`update`函数。
- `visualization.py`：可视化模块，用于显示机器人轨迹和地图。

## 运行项目

确保你已经安装了必要的依赖项，然后运行以下命令：

```bash
cd ch1_ekf_slam
pip install -r requirements.txt
python main.py --debug
```

## 完成效果

[![EKF SLAM Visualization](./viz.mp4)](./viz.mp4)
