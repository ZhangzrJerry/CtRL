import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow


class TrajVisualizer:
    def __init__(self, xs, u, dt=0.1, obstacles=None, path_ref=None):
        """xs: planned states (N+1,3), u: controls (N,2)
        obstacles: list of (cx,cy,r)
        path_ref: list of (x,y) waypoints from RRT
        """
        self.xs = xs
        self.u = u
        self.dt = dt
        self.N = u.shape[0]
        self.obstacles = obstacles if obstacles is not None else []
        self.path_ref = path_ref

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        # plot planned path
        (self.planned_line,) = self.ax.plot(xs[:, 0], xs[:, 1], "b--", label="Planned")

        # RRT path (if provided)
        if self.path_ref is not None:
            pr = np.array(self.path_ref)
            (self.rrt_line,) = self.ax.plot(
                pr[:, 0], pr[:, 1], "g-", linewidth=2, label="RRT Path"
            )
        else:
            (self.rrt_line,) = self.ax.plot([], [], "g-", linewidth=2, label="RRT Path")

        # obstacles
        self.obs_patches = []
        for cx, cy, r in self.obstacles:
            c = Circle((cx, cy), r, color="red", alpha=0.4)
            self.ax.add_patch(c)
            self.obs_patches.append(c)

        # actual robot marker (will follow planned)
        self.robot_patch = Circle((xs[0, 0], xs[0, 1]), 0.12, color="orange", zorder=5)
        self.ax.add_patch(self.robot_patch)

        # heading arrow
        self.heading = FancyArrow(
            xs[0, 0],
            xs[0, 1],
            0.5 * np.cos(xs[0, 2]),
            0.5 * np.sin(xs[0, 2]),
            width=0.05,
            color="k",
        )
        self.ax.add_patch(self.heading)

        self.trace_x = []
        self.trace_y = []
        (self.trace_line,) = self.ax.plot([], [], "r-", linewidth=2, label="Executed")

        # set limits
        margin = 1.0
        minx, maxx = np.min(xs[:, 0]) - margin, np.max(xs[:, 0]) + margin
        miny, maxy = np.min(xs[:, 1]) - margin, np.max(xs[:, 1]) + margin
        # extend limits for obstacles
        for cx, cy, r in self.obstacles:
            minx = min(minx, cx - r - margin)
            maxx = max(maxx, cx + r + margin)
            miny = min(miny, cy - r - margin)
            maxy = max(maxy, cy + r + margin)

        self.ax.set_xlim(minx, maxx)
        self.ax.set_ylim(miny, maxy)
        self.ax.legend()

    def update(self, k):
        # robot follows planned state at step k (k from 0..N)
        idx = min(k, self.N)
        x, y, th = self.xs[idx]
        self.robot_patch.center = (x, y)

        # update heading: remove old and draw new
        try:
            self.heading.remove()
        except Exception:
            pass
        self.heading = FancyArrow(
            x, y, 0.5 * np.cos(th), 0.5 * np.sin(th), width=0.05, color="k"
        )
        self.ax.add_patch(self.heading)

        self.trace_x.append(x)
        self.trace_y.append(y)
        self.trace_line.set_data(self.trace_x, self.trace_y)

        # when k >= N, leave robot at final pose (animation stays on last frame)
        return [
            self.planned_line,
            self.robot_patch,
            self.heading,
            self.trace_line,
            self.rrt_line,
        ]

    def animate(self, interval=100, save_path=None, debug=False):
        # play frames 0..N once, and then stop (no repeat)
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=range(self.N + 1),
            interval=interval,
            blit=False,
            repeat=False,
        )
        if debug or save_path is None:
            plt.show()
        else:
            try:
                writer = animation.FFMpegWriter(fps=30)
                ani.save(save_path, writer=writer)
                print(f"Saved animation to {save_path}")
            except Exception as e:
                print("Could not save animation:", e)
                plt.show()
