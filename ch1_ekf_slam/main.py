import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from scipy.linalg import block_diag
import time
import os
import sys
from tqdm import tqdm
from ekf_slam import EkfSlam
from visualization import EnhancedVisualizer


# 主程序
def main():
    np.random.seed(42)

    # 创建地标
    num_landmarks = 50
    landmarks = np.random.uniform(-8, 8, (num_landmarks, 2))

    # 创建机器人
    robot = type("Robot", (), {})()
    robot.state = np.array([0, 0, 0])
    robot.trajectory = [robot.state.copy()]
    robot.dt = 0.05

    def move(self, v, w):
        x, y, theta = self.state
        if abs(w) < 1e-6:
            x_new = x + v * np.cos(theta) * self.dt
            y_new = y + v * np.sin(theta) * self.dt
            theta_new = theta
        else:
            x_new = x + (v / w) * (np.sin(theta + w * self.dt) - np.sin(theta))
            y_new = y + (v / w) * (np.cos(theta) - np.cos(theta + w * self.dt))
            theta_new = theta + w * self.dt
        self.state = np.array([x_new, y_new, theta_new])
        self.trajectory.append(self.state.copy())

    radius = 6.0
    v_linear = 1.5
    dt = robot.dt

    # 向外走到半径处所需步数
    steps_out = int(np.ceil((radius / v_linear) / dt))
    w_align = np.pi / 4  # 45 deg/s for alignment
    angle_align = np.pi / 2  # 90 degrees to align to tangent at (r,0)
    steps_align = int(np.ceil((angle_align / w_align) / dt))

    # 在圆上匀速行驶：角速度 w_circle = v / r，使轨迹半径为 r
    w_circle = v_linear / radius
    steps_circle = int(np.ceil((2 * np.pi) / (w_circle * dt)))

    # 在圆周结束后，将朝向调整为面向原点（pi）以便沿半径返回
    angle_align_back = np.pi - angle_align  # 从切向(pi/2)到朝向原点(pi)需要再转 pi/2
    steps_align_back = int(np.ceil((angle_align_back / w_align) / dt))

    # 构建计划序列：向外 -> 就位转向 -> 绕圆 -> 就位转向回朝向原点 -> 返回
    plan = [
        (v_linear, 0.0, steps_out),  # outward along +x to (r,0)
        (0.0, w_align, steps_align),  # in-place align to tangent (pi/2)
        (v_linear, w_circle, steps_circle),  # circle around origin (CCW)
        (0.0, w_align, steps_align_back),  # align to face origin (pi)
        (v_linear, 0.0, steps_out),  # return along -x toward origin (with heading=pi)
    ]

    robot._plan = plan
    robot._plan_index = 0
    robot._plan_step = 0

    def get_control_input_plan():
        if robot._plan_index >= len(robot._plan):
            return 0.0, 0.0
        v, w, steps = robot._plan[robot._plan_index]
        robot._plan_step += 1
        if robot._plan_step >= steps:
            robot._plan_index += 1
            robot._plan_step = 0
        return v, w

    robot.move = lambda v, w: move(robot, v, w)
    robot.get_control_input = lambda: get_control_input_plan()

    # 创建EKF SLAM：设定噪声参数
    Q = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks))
    Q[:3, :3] = np.diag([0.01, 0.01, 0.005])  # 减小过程噪声
    R = np.diag([0.1, 0.05])  # 合理的观测噪声

    ekf_slam = EkfSlam(initial_state=[0, 0, 0], num_landmarks=num_landmarks, Q=Q, R=R)

    # 创建可视化器
    visualizer = EnhancedVisualizer(robot, ekf_slam, landmarks)

    # 计算总帧数
    total_frames = sum([seg[2] for seg in robot._plan])

    # 创建动画
    ani = animation.FuncAnimation(
        visualizer.fig,
        visualizer.update,
        frames=total_frames,
        interval=150,
        blit=False,
        repeat=False,
    )

    # 将进度条附加到 visualizer（FFmpeg 保存过程中会调用 update）
    try:
        visualizer.pbar = tqdm(total=total_frames, desc="Frames")
    except Exception:
        visualizer.pbar = None

    # 输出路径
    out_path = os.path.join(os.path.dirname(__file__), "viz.mp4")

    # 如果传入 --debug，则不写入文件，而是弹出交互窗口（plt.show()）
    if "--debug" in sys.argv:
        print(
            "Debug mode (--debug) detected: showing animation window instead of saving to file."
        )
        try:
            # 在调试时直接显示动画窗口（不会写入磁盘）
            plt.show()
        except Exception as e:
            print("Could not show animation window:", e)
        finally:
            # 关闭进度条（如果存在）
            try:
                pbar = getattr(visualizer, "pbar", None)
                if pbar is not None:
                    try:
                        pbar.close()
                    except Exception:
                        pass
            except Exception:
                pass
    else:
        # 正常模式：尝试将动画保存为 MP4（需要系统上安装 ffmpeg）并在保存完成后关闭图形
        try:
            writer = animation.FFMpegWriter(
                fps=30, metadata={"artist": "optimal_control"}, bitrate=1800
            )
            print(f"Saving animation to {out_path} (this may take a moment)...")
            ani.save(out_path, writer=writer)
            print(f"Saved animation to {out_path}")
        except Exception as e:
            print("Could not save animation to MP4:", e)
            print(
                "If ffmpeg is not installed, please install it and ensure it's on your PATH."
            )
        finally:
            # 关闭进度条（如果存在）
            try:
                pbar = getattr(visualizer, "pbar", None)
                if pbar is not None:
                    try:
                        pbar.close()
                    except Exception:
                        pass
            except Exception:
                pass

    # 关闭图形窗口以释放资源
    try:
        plt.close(visualizer.fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
