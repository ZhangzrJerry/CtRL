import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cartpole import CartPole, rk4_step, saturate
from visualization import CartPoleVisualizer
from tqdm import tqdm
from lqr import discretize, finite_horizon_lqr


def main():
    cart = CartPole()

    # 平衡点（倒立位）
    x_eq = np.array([0.0, 0.0, 0.0, 0.0])
    u_eq = 0.0

    # 仿真时间步长（用于离散化）
    dt = 0.02

    # 在倒立点对系统进行线性化（连续时间）
    A, B = cart.linearize(x_eq, u_eq)

    # 使用零阶保持对 A, B 进行离散化
    Ad, Bd = discretize(A, B, dt)

    # LQR 权重（离散时间）
    Q = np.diag([10.0, 1.0, 100.0, 1.0])
    R = np.array([[0.1]])

    # 使用 LQR 控制器仿真非线性系统
    T = 10.0
    steps = int(T / dt)

    # 计算离散 LQR 增益
    Ks, Ps = finite_horizon_lqr(Ad, Bd, Q, R, N=steps)

    # 初始条件：小角偏移
    x = np.array([0.0, 0.0, 0.4, 0.0])

    states = []
    controls = []

    umax = 50.0

    for i in tqdm(range(steps), desc="Simulating"):
        dx = x - x_eq
        u = -Ks[i].dot(dx)[0]
        u = saturate(u, umax)

        states.append(x.copy())
        controls.append(u)

        # integrate
        x = rk4_step(cart.dynamics, x, u, dt)

    states = np.array(states)
    controls = np.array(controls)

    # 可选的状态绘图：默认禁用（仅生成动态动画）
    show_states = False
    if show_states:
        t = np.arange(len(states)) * dt
        fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        axs[0].plot(t, states[:, 0])
        axs[0].set_ylabel("x (m)")
        axs[1].plot(t, states[:, 1])
        axs[1].set_ylabel("x_dot (m/s)")
        axs[2].plot(t, states[:, 2])
        axs[2].set_ylabel("theta (rad)")
        axs[3].plot(t, states[:, 3])
        axs[3].set_ylabel("theta_dot (rad/s)")
        axs[3].set_xlabel("time (s)")
        fig.suptitle("Cart-Pole states under LQR")

        # control plot below
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 2))
        ax2.plot(t, controls)
        ax2.set_ylabel("u (N)")
        ax2.set_xlabel("time (s)")
        ax2.set_title("Control input")

        plt.tight_layout()

    # 动画：如果传入 --debug 则交互显示，否则尝试保存为 viz.mp4
    # 为仅动态部分创建可视化器
    vis = CartPoleVisualizer(states, controls, pole_length=cart.l, show_states=False)
    out_path = os.path.join(os.path.dirname(__file__), "viz.mp4")

    if "--debug" in sys.argv:
        # interactive mode: show plots/animation
        try:
            # show dynamic-only animation
            vis.animate()
        except Exception as e:
            print("Animation/show failed:", e)
            try:
                plt.show()
            except Exception:
                pass
    else:
        # non-interactive mode: try to save to MP4, fallback to showing
        try:
            print(f"Saving animation to {out_path} (may require ffmpeg)...")
            vis.animate(save_path=out_path)
            print(f"Saved animation to {out_path}")
        except Exception as e:
            print("Could not save animation to MP4:", e)
            print("Falling back to showing animation window.")
            try:
                vis.animate()
            except Exception as e2:
                print("Animation/show also failed:", e2)


if __name__ == "__main__":
    main()
