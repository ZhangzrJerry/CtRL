import numpy as np


class DifferentialDrive:
    """简单的差速轮底盘模型"""

    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.05):
        self.state = np.array([x, y, theta], dtype=float)
        self.dt = dt
        self.trajectory = [self.state.copy()]

    def move(self, v, w):
        """按闭式解积分运动学（与 ch1 中一致）"""
        x, y, theta = self.state
        if abs(w) < 1e-8:
            x_new = x + v * np.cos(theta) * self.dt
            y_new = y + v * np.sin(theta) * self.dt
            theta_new = theta
        else:
            x_new = x + (v / w) * (np.sin(theta + w * self.dt) - np.sin(theta))
            y_new = y + (v / w) * (np.cos(theta) - np.cos(theta + w * self.dt))
            theta_new = theta + w * self.dt

        # 归一化角度到 [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        self.state = np.array([x_new, y_new, theta_new])
        self.trajectory.append(self.state.copy())

    def reset(self, x=0.0, y=0.0, theta=0.0):
        self.state = np.array([x, y, theta], dtype=float)
        self.trajectory = [self.state.copy()]
