import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class EnhancedVisualizer:
    """增强的可视化器，包含协方差矩阵可视化"""

    def __init__(self, robot, ekf_slam, landmarks):
        self.robot = robot
        self.ekf_slam = ekf_slam
        self.landmarks = landmarks
        # 当前帧的观测集合（由 generate_observations 更新）
        self.current_observations = set()

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

        # 为每个真实地标创建一条可见连线（机器人 -> 真实地标），初始为空
        self.observed_lines = []
        for _ in range(len(self.landmarks)):
            (ln,) = self.ax_slam.plot([], [], "k-", alpha=0.6, linewidth=1)
            ln.set_visible(False)
            self.observed_lines.append(ln)

        self.ax_slam.legend()

    def update_covariance_heatmap(self):
        """更新协方差矩阵热图"""
        self.ax_cov.clear()
        cov_matrix = self.ekf_slam.Sigma

        # 使用对数尺度显示协方差，避免极端值的影响
        log_cov = np.log1p(np.abs(cov_matrix) + 1e-12)

        im = self.ax_cov.imshow(
            log_cov,
            cmap="hot",
            aspect="auto",
            vmin=0,
            vmax=np.max(log_cov) if np.max(log_cov) > 0 else 1,
        )

        self.ax_cov.set_title("Covariance Matrix (log scale)")
        self.ax_cov.set_xlabel("State Index")
        self.ax_cov.set_ylabel("State Index")

        # 添加网格区分机器人和地标
        n_robot = 3
        self.ax_cov.axvline(x=n_robot - 0.5, color="white", linestyle="-", linewidth=2)
        self.ax_cov.axhline(y=n_robot - 0.5, color="white", linestyle="-", linewidth=2)

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

        # 更新进度条（如果 main() 为 visualizer 分配了 pbar）
        try:
            pbar = getattr(self, "pbar", None)
            if pbar is not None:
                pbar.update(1)
                # 当到达总数时尝试关闭（FFmpeg save 会在最后一帧后结束）
                if (
                    getattr(pbar, "total", None) is not None
                    and getattr(pbar, "n", 0) >= pbar.total
                ):
                    try:
                        pbar.close()
                    except Exception:
                        pass
        except Exception:
            pass

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
                r_noisy = r + np.random.normal(0, 0.1)  # 增加一些观测噪声
                phi_noisy = phi + np.random.normal(0, 0.05)
                observations[i] = (r_noisy, phi_noisy)
            # 不在范围内时不加入 observations（后续通过 current_observations 控制可见性）

        # 记录当前帧的观测索引集合，供可视化使用
        self.current_observations = set(observations.keys())

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
        else:
            self.estimated_landmarks_scatter.set_offsets([])

        # 更新机器人到被观测真实地标的连线（robot -> true landmarks）
        robot_pos = self.robot.state[:2]
        for i, ln in enumerate(self.observed_lines):
            # 仅当当前帧传感器观测到该地标时显示连线
            if i in self.current_observations:
                lm_pos = self.landmarks[i]
                ln.set_data([robot_pos[0], lm_pos[0]], [robot_pos[1], lm_pos[1]])
                ln.set_visible(True)
            else:
                ln.set_data([], [])
                ln.set_visible(False)

        # 不确定性椭圆
        self.update_uncertainty_ellipses()

    def update_uncertainty_ellipses(self):
        """更新不确定性椭圆"""
        # 机器人不确定性椭圆
        robot_cov = self.ekf_slam.Sigma[:2, :2]
        if np.linalg.det(robot_cov) > 0:
            eigenvalues, eigenvectors = np.linalg.eig(robot_cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues)  # 2σ椭圆
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
                    width, height = 2 * np.sqrt(eigenvalues)
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
        return artists
