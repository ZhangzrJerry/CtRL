import sys
import os
import numpy as np
from traj_opt import optimize_trajectory
from visualization import TrajVisualizer
from rrt import rrt_plan


def main():
    # simple example: start -> goal
    start = [0.0, 0.0, 0.0]
    goal = [5.0, 5.0, np.pi / 2]

    N = 80
    dt = 0.1

    # define obstacles (circle format: x,y,r)
    obstacles = [
        (2.0, 3.0, 1.0),
        (3.5, 1.5, 0.6),
        (4.0, 4.5, 0.6),
        (1.5, 0.5, 0.8),
    ]

    print("Running RRT to get reference path around obstacles...")
    # search space bounds
    xlim = (-1, 6)
    ylim = (-1, 6)
    path = rrt_plan(
        start[:2],
        goal[:2],
        obstacles,
        xlim,
        ylim,
        max_iter=3000,
        step_size=0.4,
        goal_radius=0.4,
    )
    if path is None:
        print("RRT failed to find a path; optimizing without RRT guidance.")

    print("Optimizing trajectory (with obstacle and path-following terms)...")
    xs, u_opt, res = optimize_trajectory(
        start, goal, N=N, dt=dt, obstacles=obstacles, path_ref=path
    )

    final = xs[-1]
    print(f"Final state: {final}")
    pos_err = np.linalg.norm(final[:2] - np.array(goal[:2]))
    ang_err = abs(np.arctan2(np.sin(final[2] - goal[2]), np.cos(final[2] - goal[2])))
    print(f"Position error: {pos_err:.4f}, Angle error: {ang_err:.4f}")

    # visualize
    vis = TrajVisualizer(xs, u_opt, dt=dt, obstacles=obstacles, path_ref=path)

    debug = "--debug" in sys.argv
    out_path = os.path.join(os.path.dirname(__file__), "viz.mp4")
    if debug:
        vis.animate(interval=80, debug=True)
    else:
        vis.animate(interval=80, save_path=out_path, debug=False)


if __name__ == "__main__":
    main()
