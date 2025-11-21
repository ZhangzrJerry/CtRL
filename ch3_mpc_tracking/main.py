import numpy as np
import matplotlib.animation as animation
import sys
from robot import DifferentialDrive
from mpc import solve_mpc
from visualization import MPCVisualizer


def make_square_path(start=(0, 0), side=4.0, points_per_edge=80):
    # 顺时针正方形：以起点为左上角，顺时针走
    x0, y0 = start
    corners = [
        (x0, y0),
        (x0 + side, y0),
        (x0 + side, y0 - side),
        (x0, y0 - side),
        (x0, y0),
    ]
    path = []
    for i in range(len(corners) - 1):
        a = np.array(corners[i])
        b = np.array(corners[i + 1])
        for t in np.linspace(0, 1, points_per_edge, endpoint=False):
            p = (1 - t) * a + t * b
            path.append((p[0], p[1], 0.0))
    # 最后添加终点
    path.append((corners[-1][0], corners[-1][1], 0.0))
    return np.array(path)


def run_simulation():
    dt = 0.05
    robot = DifferentialDrive(x=0.0, y=0.0, theta=0.0, dt=dt)

    # 目标正方形：顺时针
    path = make_square_path(start=(0.0, 0.0), side=4.0, points_per_edge=80)

    visualizer = MPCVisualizer(robot, path[:, :2])

    horizon = 12

    total_steps = len(path)

    # For animation we will compute one control per frame
    def update(frame):
        # Build reference trajectory starting from current closest point in path
        # find nearest path index
        x_cur = robot.state[0]
        y_cur = robot.state[1]
        dists = np.linalg.norm(path[:, :2] - np.array([x_cur, y_cur]), axis=1)
        idx0 = int(np.argmin(dists))

        # reference for horizon+1 states
        ref_idx = [min(idx0 + i, len(path) - 1) for i in range(horizon + 1)]
        ref_traj = path[ref_idx]

        # solve mpc for current state -> returns first control, full control sequence and predicted trajectory
        v, w, us_opt, pred_traj = solve_mpc(
            robot.state.copy(), ref_traj, horizon=horizon, dt=dt, vmax=1.5, wmax=2
        )

        # apply control
        robot.move(v, w)

        # update visualization: include horizon reference and MPC predicted rollout
        artists = visualizer.update_with_ref_and_pred(
            ref_traj=ref_traj, pred_traj=pred_traj
        )
        return artists

    ani = animation.FuncAnimation(
        visualizer.fig,
        update,
        frames=total_steps,
        interval=50,
        blit=False,
        repeat=False,
    )

    if "--debug" in sys.argv:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        # try save
        try:
            writer = animation.FFMpegWriter(
                fps=20, metadata={"artist": "mpc_tracking"}, bitrate=1500
            )
            out_path = "viz.mp4"
            print(f"Saving animation to {out_path}...")
            ani.save(out_path, writer=writer)
            print("Saved.")
        except Exception as e:
            print(
                "Could not save animation (ffmpeg may be missing). Use --debug to show window."
            )
            print(e)


if __name__ == "__main__":
    run_simulation()
