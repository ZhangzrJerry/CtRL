import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class CartPoleVisualizer:
    def __init__(
        self,
        states,
        controls,
        cart_width=0.4,
        pole_length=0.5,
        dt=0.02,
        show_states=True,
    ):
        """可视化器。

        参数说明：
        - 如果 show_states 为 False，则只生成小车-杆的动画（不绘制状态曲线）。
        - states: 状态序列（N x 4）
        - controls: 控制输入序列（N,）
        - dt: 采样时间，用于绘制时间轴
        """
        self.states = np.array(states)
        self.controls = np.array(controls)
        self.dt = dt
        self.cart_w = cart_width
        self.pole_l = pole_length
        self.show_states = bool(show_states)

        # 图形：如果 show_states 为真，包含动画与状态子图；否则只有动画轴
        if self.show_states:
            self.fig, (self.ax_traj, self.ax_states) = plt.subplots(
                2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1]}
            )
        else:
            self.fig, self.ax_traj = plt.subplots(1, 1, figsize=(6, 4))

        # 轨迹绘图区设置
        self.ax_traj.set_xlim(-2.0, 2.0)
        self.ax_traj.set_ylim(-1.0, 1.5)
        self.ax_traj.set_aspect("equal")
        self.ax_traj.set_title("Cart-Pole animation")

        # 小车与杆的图形元素
        self.cart_patch = plt.Rectangle((0, 0), self.cart_w, 0.2, fc="C0")
        self.ax_traj.add_patch(self.cart_patch)
        (self.pole_line,) = self.ax_traj.plot([], [], lw=3, c="C1")

        # 可选的状态-控制曲线绘制
        if self.show_states:
            t = np.arange(len(self.states)) * self.dt
            self.t = t
            self.ax_states.set_title("States and control")
            self.ax_states.plot(t, self.states[:, 0], label="x")
            self.ax_states.plot(t, self.states[:, 1], label="x_dot")
            self.ax_states.plot(t, self.states[:, 2], label="theta")
            self.ax_states.plot(t, self.states[:, 3], label="theta_dot")
            # controls 长度可能与状态不同，保护性绘制
            try:
                self.ax_states.plot(t, self.controls, label="u")
            except Exception:
                # 如果形状不匹配，忽略控制绘制
                pass
            self.ax_states.legend()

    def _state_to_artists(self, state):
        x, x_dot, theta, theta_dot = state
        # cart rectangle centered at x
        cart_x = x - self.cart_w / 2.0
        cart_y = 0.0
        self.cart_patch.set_xy((cart_x, cart_y))

        # pole end coordinates
        pole_x = x + self.pole_l * np.sin(theta)
        pole_y = 0.2 + self.pole_l * np.cos(theta)
        self.pole_line.set_data([x, pole_x], [0.2, pole_y])

        return [self.cart_patch, self.pole_line]

    def animate(self, interval=50, save_path=None):
        frames = len(self.states)

        def update(i):
            artists = self._state_to_artists(self.states[i])
            return artists

        ani = animation.FuncAnimation(
            self.fig, update, frames=frames, interval=interval, blit=False
        )

        if save_path:
            try:
                ani.save(save_path, fps=30)
            except Exception as e:
                print("Could not save animation:", e)
        else:
            plt.show()
