import os
import argparse
import numpy as np
from tqdm import trange
from cartpole import CartPole
from visualization import CartPoleVisualizer, save_training_plot
from dqn import DQNAgent, calc_reward
import matplotlib.pyplot as plt


def evaluate_policy(
    agent,
    cart,
    u,
    dt,
    max_steps,
    start_state,
    save_path=None,
    show_states=False,
):
    """对当前策略做一次贪婪评估并返回轨迹；如果提供 save_path 则尝试保存动画，否则直接显示。"""
    state = start_state.copy()
    states = []
    controls = []
    for t in range(max_steps):
        states.append(state.copy())
        controls.append(u)
        state = cart.discrete_step(state, u, dt)
        # 不再基于角度立即终止评估（以便观察摆起过程），仅在位置越界时结束
        if abs(state[0]) > 2.4:
            break

    states = np.array(states)
    controls = np.array(controls)

    vis = CartPoleVisualizer(
        states, controls, pole_length=cart.l, show_states=show_states, dt=dt
    )

    if save_path is not None:
        try:
            vis.animate(save_path=save_path)
            print(f"Saved evaluation video to {save_path}")
        except Exception as e:
            print("Could not save evaluation video:", e)
            vis.animate()
    else:
        vis.animate()


def train(debug=False, max_epoches=200):
    """训练 DQN 智能体来控制连续动力学的 CartPole"""
    # 环境与参数
    cart = CartPole()
    dt = 0.02
    state_dim = 4
    action_dim = 1

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-4)

    # number of epoches can be overridden for quick tests
    # default comes from function argument
    max_steps = int(10.0 / dt)

    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.995
    eps = eps_start

    rewards_hist = []
    loss_hist = []

    # ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    for ep in trange(max_epoches, desc="Training"):
        # 初始状态：从“完全垂下”开始训练（theta = pi），加入少量噪声帮助探索
        state = np.array([0.0, 0.0, np.pi + 0.01 * np.random.randn(), 0.0])
        ep_reward = 0.0
        ep_losses = []

        for t in range(max_steps):
            u = agent.select_action(state, eps=eps)

            next_state = cart.discrete_step(state, u, dt)

            done = False
            if abs(next_state[0]) > 2.4:  # 位置边界
                done = True
                reward = -50.0
            else:
                reward = calc_reward(next_state)

            agent.store(state, u, reward, next_state, float(done))
            loss = agent.update()
            if loss is not None:
                try:
                    if loss > 0.0:
                        ep_losses.append(loss)
                except Exception:
                    pass

            state = next_state
            ep_reward += reward

            if done:
                break

        # epsilon 衰减
        eps = max(eps * eps_decay, eps_end)
        rewards_hist.append(ep_reward)
        if len(ep_losses) > 0:
            loss_hist.append(float(np.mean(ep_losses)))
        else:
            loss_hist.append(float(loss_hist[-1]) if len(loss_hist) > 0 else 0.0)

        # 做一次贪婪评估并保存视频
        if (ep + 1) % 50 == 0:
            save_path = None
            show_states = False
            if debug:
                # 在调试模式下直接展示（plt.show）
                save_path = None
                show_states = True
                print(f"[DEBUG] Evaluation after episode {ep+1}: showing animation")
            else:
                # 非调试模式下保存为 mp4 到 output 目录
                save_path = os.path.join(output_dir, f"dqn_eval_ep{ep+1}.mp4")
                print(f"Saving evaluation video to {save_path} (may require ffmpeg)...")
                print(
                    f"Episode {ep+1}, Loss: {loss:.4f}, Reward: {ep_reward:.2f}, "
                    f"Epsilon: {eps:.3f}, Buffer: {len(agent.replay)}"
                )

            start_state = np.array([0.0, 0.0, np.pi, 0.0])
            evaluate_policy(
                agent,
                cart,
                u,
                dt,
                max_steps,
                start_state,
                save_path=save_path,
                show_states=show_states,
            )
            # save training plot alongside the video
            try:
                save_training_plot(
                    ep + 1,
                    output_dir=output_dir,
                    rewards_hist=rewards_hist,
                    loss_hist=loss_hist,
                )
            except Exception:
                pass

    # 训练结束后做一次最终评估并保存/展示
    final_save = None if debug else os.path.join(output_dir, "dqn_final.mp4")
    # save final training plot
    try:
        save_training_plot(
            None,
            "_final",
            output_dir=output_dir,
            rewards_hist=rewards_hist,
            loss_hist=loss_hist,
        )
    except Exception:
        pass
    start_state = np.array([0.0, 0.0, np.pi, 0.0])
    evaluate_policy(
        agent,
        cart,
        u,
        dt,
        max_steps,
        start_state,
        save_path=final_save,
        show_states=(debug),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="在调试模式下使用 plt.show() 显示动画而不是保存",
    )
    parser.add_argument(
        "--epoches",
        type=int,
        default=200,
        help="训练轮数（用于快速测试）",
    )
    args = parser.parse_args()
    train(debug=args.debug, max_epoches=args.epoches)
