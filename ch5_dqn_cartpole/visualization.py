import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import threading
import os


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

        if save_path:
            # Save in a background thread but build a fresh headless
            # figure/animation inside that thread. This avoids trying to
            # switch backends after pyplot was already imported and also
            # avoids GUI main-loop requirements that raise
            # "main thread is not in main loop".
            def _save():
                try:
                    import matplotlib as mpl

                    mpl.use("Agg")
                    from matplotlib import pyplot as plt2
                    from matplotlib import animation as anim2

                    # build a fresh figure independent from the main one
                    if self.show_states:
                        fig2, (ax_traj2, ax_states2) = plt2.subplots(
                            2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1]}
                        )
                    else:
                        fig2, ax_traj2 = plt2.subplots(1, 1, figsize=(6, 4))

                    ax_traj2.set_xlim(-2.0, 2.0)
                    ax_traj2.set_ylim(-1.0, 1.5)
                    ax_traj2.set_aspect("equal")
                    ax_traj2.set_title("Cart-Pole animation")

                    cart_patch2 = plt2.Rectangle((0, 0), self.cart_w, 0.2, fc="C0")
                    ax_traj2.add_patch(cart_patch2)
                    (pole_line2,) = ax_traj2.plot([], [], lw=3, c="C1")

                    # plot state curves if requested
                    if self.show_states:
                        t = np.arange(len(self.states)) * self.dt
                        ax_states2.set_title("States and control")
                        ax_states2.plot(t, self.states[:, 0], label="x")
                        ax_states2.plot(t, self.states[:, 1], label="x_dot")
                        ax_states2.plot(t, self.states[:, 2], label="theta")
                        ax_states2.plot(t, self.states[:, 3], label="theta_dot")
                        try:
                            ax_states2.plot(t, self.controls, label="u")
                        except Exception:
                            pass
                        ax_states2.legend()

                    def update2(i):
                        x, x_dot, theta, theta_dot = self.states[i]
                        cart_x = x - self.cart_w / 2.0
                        cart_patch2.set_xy((cart_x, 0.0))
                        pole_x = x + self.pole_l * np.sin(theta)
                        pole_y = 0.2 + self.pole_l * np.cos(theta)
                        pole_line2.set_data([x, pole_x], [0.2, pole_y])
                        return [cart_patch2, pole_line2]

                    ani2 = anim2.FuncAnimation(
                        fig2, update2, frames=frames, interval=interval, blit=False
                    )

                    try:
                        writer = anim2.FFMpegWriter(fps=30)
                        ani2.save(save_path, writer=writer)
                    except Exception:
                        ani2.save(save_path, fps=30)

                    print(f"Saved animation to {save_path}")
                except Exception as e:
                    print("Could not save animation:", e)
                finally:
                    try:
                        plt2.close(fig2)
                    except Exception:
                        pass

            t = threading.Thread(target=_save, daemon=False)
            t.start()
            return t
        else:
            ani = animation.FuncAnimation(
                self.fig, update, frames=frames, interval=interval, blit=False
            )
            plt.show()


def save_training_plot(
    ep_num=None, suffix="", output_dir="output", rewards_hist=[], loss_hist=[]
):
    out_name = (
        os.path.join(output_dir, f"dqn_training_ep{ep_num}{suffix}.png")
        if ep_num is not None
        else os.path.join(output_dir, f"dqn_training{suffix}.png")
    )
    try:
        plt.figure(figsize=(8, 6))
        xs = np.arange(1, len(rewards_hist) + 1)
        plt.subplot(2, 1, 1)
        plt.plot(xs, rewards_hist, lw=2)
        plt.xlim(1, len(rewards_hist))
        plt.ylabel("Reward")
        plt.subplot(2, 1, 2)
        plt.plot(xs, loss_hist, lw=2, color="C3")
        plt.xlim(1, len(rewards_hist))
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(out_name)
        plt.close()
        print(f"Saved training plot to {out_name}")
    except Exception as e:
        print("Could not save training plot:", e)
