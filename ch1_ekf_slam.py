import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from scipy.linalg import block_diag
import time
import os


class EKFSLAMWithCovarianceVisualization:
    """带有协方差矩阵可视化的EKF SLAM"""

    def __init__(self, initial_state, num_landmarks, Q, R):
        self.mu = np.zeros(3 + 2 * num_landmarks)
        self.mu[:3] = initial_state
        self.Sigma = np.eye(3 + 2 * num_landmarks) * 0.1
        self.Q = Q
        self.R = R
        self.num_landmarks = num_landmarks
        self.observed_landmarks = set()

        # 存储协方差历史用于可视化
        self.covariance_history = []

    def motion_model(self, state, u, dt):
        """运动模型"""
        x, y, theta = state[:3]
        v, w = u

        #######################################
        ### TODO: 实现运动模型               ###
        #######################################
        x_new = 0
        y_new = 0
        theta_new = 0

        return np.array([x_new, y_new, theta_new])

    def prediction_step(self, u, dt):
        """预测步骤"""
        v, w = u
        x, y, theta = self.mu[:3]

        # 状态预测
        self.mu[:3] = self.motion_model(self.mu, u, dt)

        #######################################
        ### TODO: 实现协方差预测步骤         ###
        #######################################

    def update_step(self, observations):
        """更新步骤"""
        if not observations:
            return

        x, y, theta = self.mu[:3]

        for landmark_id, (r, phi) in observations.items():
            landmark_idx = 3 + 2 * landmark_id

            if landmark_id not in self.observed_landmarks:
                #################################
                ### TODO: 初始化新地标位置      ###
                #################################
                pass

            ####################################
            ### TODO: 实现EKF更新步骤         ###
            ####################################

        # 存储当前协方差用于可视化
        self.covariance_history.append(self.Sigma.copy())


class EnhancedVisualizer:
    """增强的可视化器，包含协方差矩阵可视化"""

    def __init__(self, robot, ekf_slam, landmarks):
        self.robot = robot
        self.ekf_slam = ekf_slam
        self.landmarks = landmarks

        # placeholder for colorbar axes (set from main after visualizer creation)
        self.ax_cax = None

        # 创建主图形
        self.fig = plt.figure(figsize=(15, 10))

        # SLAM可视化
        self.ax_slam = self.fig.add_subplot(2, 3, (1, 5))
        self.ax_slam.set_xlim(-10, 10)
        self.ax_slam.set_ylim(-10, 10)
        self.ax_slam.set_aspect("equal")
        self.ax_slam.grid(True)
        self.ax_slam.set_title("2D EKF SLAM with Covariance Visualization")
        self.ax_slam.set_xlabel("X (m)")
        self.ax_slam.set_ylabel("Y (m)")

        # 协方差矩阵可视化
        self.ax_cov = self.fig.add_subplot(2, 3, 3)
        self.ax_cov.set_title("Covariance Matrix Heatmap")
        self.ax_cov.set_xlabel("State Index")
        self.ax_cov.set_ylabel("State Index")

        # 不确定性增长可视化
        self.ax_uncertainty = self.fig.add_subplot(2, 3, 6)
        self.ax_uncertainty.set_title("Uncertainty Evolution")
        self.ax_uncertainty.set_xlabel("Time Step")
        self.ax_uncertainty.set_ylabel("Uncertainty (σ²)")
        self.ax_uncertainty.grid(True)

        # 初始化SLAM可视化元素
        self.init_slam_visualization()

        # 初始化不确定性历史
        self.robot_uncertainty_history = []
        self.landmark_uncertainty_history = []

        plt.tight_layout()

    def init_slam_visualization(self):
        """初始化SLAM可视化元素"""
        # 真实地标
        self.true_landmarks_scatter = self.ax_slam.scatter(
            self.landmarks[:, 0],
            self.landmarks[:, 1],
            c="red",
            marker="*",
            s=100,
            label="True Landmarks",
        )

        # 估计地标
        self.estimated_landmarks_scatter = self.ax_slam.scatter(
            [], [], c="blue", marker="x", s=50, label="Estimated Landmarks"
        )

        # 轨迹
        (self.true_trajectory_line,) = self.ax_slam.plot(
            [], [], "g-", linewidth=2, label="True Trajectory"
        )
        (self.estimated_trajectory_line,) = self.ax_slam.plot(
            [], [], "b-", linewidth=2, label="Estimated Trajectory"
        )

        # 机器人位置
        self.true_robot_scatter = self.ax_slam.scatter(
            [], [], c="green", marker="s", s=100, label="True Robot"
        )
        self.estimated_robot_scatter = self.ax_slam.scatter(
            [], [], c="blue", marker="s", s=100, label="Estimated Robot"
        )

        # 不确定性椭圆
        self.robot_ellipse = Ellipse(
            (0, 0), 0, 0, angle=0, fill=False, edgecolor="blue", linewidth=2
        )
        self.ax_slam.add_patch(self.robot_ellipse)

        self.landmark_ellipses = []
        for _ in range(len(self.landmarks)):
            ellipse = Ellipse(
                (0, 0),
                0,
                0,
                angle=0,
                fill=False,
                edgecolor="blue",
                linewidth=1,
                alpha=0.5,
            )
            self.ax_slam.add_patch(ellipse)
            self.landmark_ellipses.append(ellipse)

        # 协方差连线（显示关联）
        self.covariance_lines = []
        for _ in range(len(self.landmarks)):
            (line,) = self.ax_slam.plot([], [], "r--", alpha=0.3, linewidth=1)
            self.covariance_lines.append(line)

        self.ax_slam.legend()

    def update_covariance_heatmap(self):
        """更新协方差矩阵热图"""
        self.ax_cov.clear()
        cov_matrix = self.ekf_slam.Sigma

        # 创建热图（不显示颜色条以减少画面复杂度）
        self.ax_cov.imshow(
            np.abs(cov_matrix),
            cmap="hot",
            aspect="auto",
            vmin=0,
            vmax=np.max(np.abs(cov_matrix)) if np.max(np.abs(cov_matrix)) > 0 else 1,
        )

        self.ax_cov.set_title("Covariance Matrix Heatmap")
        self.ax_cov.set_xlabel("State Index")
        self.ax_cov.set_ylabel("State Index")

        # 添加网格区分机器人和地标
        n_robot = 3
        n_landmarks = self.ekf_slam.num_landmarks

        self.ax_cov.axvline(x=n_robot - 0.5, color="white", linestyle="-", linewidth=2)
        self.ax_cov.axhline(y=n_robot - 0.5, color="white", linestyle="-", linewidth=2)

        # 添加文本标注
        self.ax_cov.text(
            n_robot / 2,
            n_robot / 2,
            "Robot",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        self.ax_cov.text(
            n_robot + n_landmarks,
            n_robot + n_landmarks,
            "Landmarks",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    def update_uncertainty_plot(self, step):
        """更新不确定性演化图"""
        # 计算机器人位置不确定性（迹）
        robot_uncertainty = np.trace(self.ekf_slam.Sigma[:3, :3])
        self.robot_uncertainty_history.append(robot_uncertainty)

        # 计算地标平均不确定性
        landmark_uncertainties = []
        for i in range(self.ekf_slam.num_landmarks):
            if i in self.ekf_slam.observed_landmarks:
                idx = 3 + 2 * i
                landmark_uncertainty = np.trace(
                    self.ekf_slam.Sigma[idx : idx + 2, idx : idx + 2]
                )
                landmark_uncertainties.append(landmark_uncertainty)

        avg_landmark_uncertainty = (
            np.mean(landmark_uncertainties) if landmark_uncertainties else 0
        )
        self.landmark_uncertainty_history.append(avg_landmark_uncertainty)

        # 更新图表
        self.ax_uncertainty.clear()
        steps = range(len(self.robot_uncertainty_history))

        self.ax_uncertainty.plot(
            steps,
            self.robot_uncertainty_history,
            "b-",
            label="Robot Uncertainty",
            linewidth=2,
        )
        self.ax_uncertainty.plot(
            steps,
            self.landmark_uncertainty_history,
            "r-",
            label="Avg Landmark Uncertainty",
            linewidth=2,
        )

        self.ax_uncertainty.set_title("Uncertainty Evolution")
        self.ax_uncertainty.set_xlabel("Time Step")
        self.ax_uncertainty.set_ylabel("Uncertainty (Trace of Covariance)")
        self.ax_uncertainty.legend()
        self.ax_uncertainty.grid(True)

    def update_covariance_lines(self):
        """更新协方差关联线"""
        robot_pos = self.ekf_slam.mu[:2]

        for i, line in enumerate(self.covariance_lines):
            if i in self.ekf_slam.observed_landmarks:
                idx = 3 + 2 * i
                landmark_pos = self.ekf_slam.mu[idx : idx + 2]

                # 计算协方差关联强度
                cov_strength = np.abs(
                    self.ekf_slam.Sigma[0, idx] + self.ekf_slam.Sigma[1, idx + 1]
                )

                # 设置线条颜色和透明度基于关联强度
                alpha = min(0.8, cov_strength * 10)
                line.set_alpha(alpha)
                line.set_color("purple")
                line.set_linewidth(1 + cov_strength * 5)

                line.set_data(
                    [robot_pos[0], landmark_pos[0]], [robot_pos[1], landmark_pos[1]]
                )
            else:
                line.set_data([], [])

    def update(self, frame):
        """更新所有可视化元素"""
        # 机器人运动和控制
        v, w = self.robot.get_control_input()
        self.robot.move(v, w)
        self.ekf_slam.prediction_step((v, w), self.robot.dt)

        # 生成观测
        observations = self.generate_observations()
        self.ekf_slam.update_step(observations)

        # 更新SLAM可视化
        self.update_slam_visualization()

        # 更新协方差可视化
        self.update_covariance_heatmap()
        self.update_uncertainty_plot(frame)
        self.update_covariance_lines()

        return self.get_artists()

    def generate_observations(self):
        """生成观测数据"""
        observations = {}
        x_true, y_true, theta_true = self.robot.state

        for i, landmark in enumerate(self.landmarks):
            dx = landmark[0] - x_true
            dy = landmark[1] - y_true
            r = np.sqrt(dx**2 + dy**2)
            phi = np.arctan2(dy, dx) - theta_true

            if r < 5.0:  # 传感器范围
                r_noisy = r + np.random.normal(0, 0.1)
                phi_noisy = phi + np.random.normal(0, 0.05)
                observations[i] = (r_noisy, phi_noisy)

        return observations

    def update_slam_visualization(self):
        """更新SLAM可视化"""
        # 轨迹
        trajectory = np.array(self.robot.trajectory)
        self.true_trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])

        # 估计轨迹
        estimated_trajectory = [self.ekf_slam.mu[:3]] * len(self.robot.trajectory)
        estimated_trajectory = np.array(estimated_trajectory)
        self.estimated_trajectory_line.set_data(
            estimated_trajectory[:, 0], estimated_trajectory[:, 1]
        )

        # 机器人位置
        self.true_robot_scatter.set_offsets([self.robot.state[:2]])
        self.estimated_robot_scatter.set_offsets([self.ekf_slam.mu[:2]])

        # 地标位置
        estimated_landmarks = []
        for i in range(self.ekf_slam.num_landmarks):
            if i in self.ekf_slam.observed_landmarks:
                idx = 3 + 2 * i
                estimated_landmarks.append(
                    [self.ekf_slam.mu[idx], self.ekf_slam.mu[idx + 1]]
                )

        if estimated_landmarks:
            estimated_landmarks = np.array(estimated_landmarks)
            self.estimated_landmarks_scatter.set_offsets(estimated_landmarks)

        # 不确定性椭圆
        self.update_uncertainty_ellipses()

    def update_uncertainty_ellipses(self):
        """更新不确定性椭圆"""
        # 机器人不确定性椭圆
        robot_cov = self.ekf_slam.Sigma[:2, :2]
        if np.linalg.det(robot_cov) > 0:
            eigenvalues, eigenvectors = np.linalg.eig(robot_cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 3 * np.sqrt(eigenvalues)  # 3σ椭圆
            self.robot_ellipse.center = self.ekf_slam.mu[:2]
            self.robot_ellipse.width = width
            self.robot_ellipse.height = height
            self.robot_ellipse.angle = angle

        # 地标不确定性椭圆
        for i in range(self.ekf_slam.num_landmarks):
            if i in self.ekf_slam.observed_landmarks:
                idx = 3 + 2 * i
                landmark_cov = self.ekf_slam.Sigma[idx : idx + 2, idx : idx + 2]
                if np.linalg.det(landmark_cov) > 0:
                    eigenvalues, eigenvectors = np.linalg.eig(landmark_cov)
                    angle = np.degrees(
                        np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                    )
                    width, height = 3 * np.sqrt(eigenvalues)
                    self.landmark_ellipses[i].center = self.ekf_slam.mu[idx : idx + 2]
                    self.landmark_ellipses[i].width = width
                    self.landmark_ellipses[i].height = height
                    self.landmark_ellipses[i].angle = angle
                    self.landmark_ellipses[i].set_visible(True)
            else:
                self.landmark_ellipses[i].set_visible(False)

    def get_artists(self):
        """返回所有需要动画的艺术家对象"""
        artists = [
            self.true_trajectory_line,
            self.estimated_trajectory_line,
            self.true_robot_scatter,
            self.estimated_robot_scatter,
            self.estimated_landmarks_scatter,
            self.robot_ellipse,
        ]
        artists.extend(self.landmark_ellipses)
        artists.extend(self.covariance_lines)
        return artists


# 主程序
def main():
    np.random.seed(42)

    # 创建地标
    num_landmarks = 8
    landmarks = np.random.uniform(-8, 8, (num_landmarks, 2))

    # 创建机器人
    robot = type("Robot", (), {})()
    robot.state = np.array([0, 0, 0])
    robot.trajectory = [robot.state.copy()]
    robot.dt = 0.1

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

    # 运动计划：机器人从原点出发，沿 +x 方向出发到半径 r，然后以原点为圆心、半径 r 绕行一圈，
    # 最后调整朝向并沿半径回到原点（注意会做短暂的就位转向以保证切向运动）
    radius = 5.0
    v_linear = 0.5
    dt = robot.dt

    # 向外走到半径处所需步数
    steps_out = int(np.ceil((radius / v_linear) / dt))

    # 就位转向（将朝向从 0 调整为切线方向 pi/2）使用固定角速度
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

    # 创建EKF SLAM
    Q = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks))
    Q[:3, :3] = np.diag([0.1, 0.1, 0.05])
    R = np.diag([0.1, 0.05])

    ekf_slam = EKFSLAMWithCovarianceVisualization(
        initial_state=[0, 0, 0], num_landmarks=num_landmarks, Q=Q, R=R
    )

    # 创建可视化器
    visualizer = EnhancedVisualizer(robot, ekf_slam, landmarks)

    # 不需要为协方差热图创建单独的颜色条轴（heatmap 不显示颜色条）

    # 计算总帧数：基于运动计划的总步数，这样保存（ani.save）会包含完整的执行直到回到原点
    total_frames = sum([seg[2] for seg in robot._plan])

    # 创建动画，使用 total_frames 并且不循环，确保保存时会记录完整的回到原点过程
    ani = animation.FuncAnimation(
        visualizer.fig,
        visualizer.update,
        frames=total_frames,
        interval=150,
        blit=False,  # disable blitting so axes that are cleared/replotted update correctly
        repeat=False,
    )

    # 尝试将动画保存为 MP4（需要系统上安装 ffmpeg）
    out_path = os.path.join(os.path.dirname(__file__), "viz.mp4")
    try:
        # 使用内置的 ffmpeg writer
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=30, metadata={"artist": "optimal_control"}, bitrate=1800)
        print(f"Saving animation to {out_path} (this may take a moment)...")
        ani.save(out_path, writer=writer)
        print(f"Saved animation to {out_path}")
    except Exception as e:
        # 常见原因是系统没有安装 ffmpeg，可在系统中安装并将其加入 PATH 后重试
        print("Could not save animation to MP4:", e)
        print(
            "If ffmpeg is not installed, please install it and ensure it's on your PATH."
        )

    plt.show()


if __name__ == "__main__":
    main()
