import random
import math
import numpy as np


def line_collision(p1, p2, obstacles, step=0.05):
    """Check collision between segment p1->p2 and circular obstacles."""
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    steps = max(2, int(dist / step))
    for i in range(steps + 1):
        t = i / steps
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        for cx, cy, r in obstacles:
            if math.hypot(x - cx, y - cy) <= r:
                return True
    return False


def rrt_plan(
    start, goal, obstacles, xlim, ylim, max_iter=5000, step_size=0.5, goal_radius=0.5
):
    """
    TODO: 完成 RRT 路径规划算法，返回路径点列表

    start, goal: (x,y)
    obstacles: list of (x,y,radius)
    xlim, ylim: (min, max)

    Return: list of (x,y) waypoints from start to goal, or None if failed
    """
    pass
