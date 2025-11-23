import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class MPCVisualizer:
    def __init__(self, robot, path_points):
        self.robot = robot
        self.path_points = np.array(path_points)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(
            "MPC Tracking: Differential Drive Following Square (Clockwise)"
        )

        # plot reference path
        self.ax.plot(self.path_points[:, 0], self.path_points[:, 1], "k--", linewidth=1)
        self.ax.scatter(self.path_points[:, 0], self.path_points[:, 1], c="red", s=40)

        # robot trajectory line
        (self.traj_line,) = self.ax.plot([], [], "b-", linewidth=2, label="Trajectory")
        # reference horizon (what MPC is trying to follow for the next N steps)
        (self.ref_horizon_line,) = self.ax.plot(
            [],
            [],
            "orange",
            linestyle="--",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Ref Horizon",
        )
        # predicted trajectory from MPC (predicted rollout)
        (self.pred_traj_line,) = self.ax.plot(
            [],
            [],
            "lime",
            linestyle="-",
            linewidth=2,
            marker=".",
            markersize=4,
            label="MPC Predicted",
        )
        # robot current pose marker
        self.robot_marker = plt.Circle((0, 0), 0.12, color="green", zorder=5)
        self.ax.add_patch(self.robot_marker)

        # heading arrow
        self.heading_arrow = FancyArrowPatch(
            (0, 0), (0.2, 0), mutation_scale=15, color="green"
        )
        self.ax.add_patch(self.heading_arrow)

        # axes limits based on path
        margin = 1.0
        minx, miny = np.min(self.path_points, axis=0) - margin
        maxx, maxy = np.max(self.path_points, axis=0) + margin
        self.ax.set_xlim(minx, maxx)
        self.ax.set_ylim(miny, maxy)

    def update(self, frame=None):
        traj = np.array(self.robot.trajectory)
        if traj.shape[0] > 0:
            self.traj_line.set_data(traj[:, 0], traj[:, 1])
            x, y, th = traj[-1]
            self.robot_marker.center = (x, y)
            # heading arrow
            dx = 0.3 * np.cos(th)
            dy = 0.3 * np.sin(th)
            self.heading_arrow.set_positions((x, y), (x + dx, y + dy))
        # clear ref/pred lines by default
        try:
            self.ref_horizon_line.set_data([], [])
            self.pred_traj_line.set_data([], [])
        except Exception:
            pass

        return [
            self.traj_line,
            self.robot_marker,
            self.heading_arrow,
            self.ref_horizon_line,
            self.pred_traj_line,
        ]

    def update_with_ref_and_pred(self, ref_traj=None, pred_traj=None):
        """更新并显示参考 horizon（ref_traj, shape (N+1,3)）和 MPC 预测轨迹 (pred_traj, (N+1,3))"""
        artists = self.update()

        if ref_traj is not None and len(ref_traj) > 0:
            pts = np.array(ref_traj)
            self.ref_horizon_line.set_data(pts[:, 0], pts[:, 1])

        if pred_traj is not None and len(pred_traj) > 0:
            pts = np.array(pred_traj)
            self.pred_traj_line.set_data(pts[:, 0], pts[:, 1])

        # return artists including the two horizon lines
        return [
            self.traj_line,
            self.robot_marker,
            self.heading_arrow,
            self.ref_horizon_line,
            self.pred_traj_line,
        ]
