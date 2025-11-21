import numpy as np
from scipy.optimize import minimize
from math import hypot


def simulate_dynamics(u, start, dt):
    """Simulate differential-drive discrete dynamics given control sequence u.
    u: shape (N,2) controls [v, w]
    start: [x,y,theta]
    returns states shape (N+1,3)
    """
    N = u.shape[0]
    xs = np.zeros((N + 1, 3))
    xs[0] = start
    for k in range(N):
        x, y, th = xs[k]
        v, w = u[k]
        if abs(w) < 1e-8:
            x_new = x + v * np.cos(th) * dt
            y_new = y + v * np.sin(th) * dt
            th_new = th
        else:
            x_new = x + (v / w) * (np.sin(th + w * dt) - np.sin(th))
            y_new = y + (v / w) * (-np.cos(th + w * dt) + np.cos(th))
            th_new = th + w * dt
        xs[k + 1] = np.array([x_new, y_new, th_new])
    return xs


def optimize_trajectory(
    start,
    goal,
    N=50,
    dt=0.1,
    v_bounds=(0.0, 2.0),
    w_bounds=(-3.14, 3.14),
    obstacles=None,
    path_ref=None,
    **kwargs,
):
    """
    TODO: 实现移动机器人轨迹优化，考虑避障与路径跟踪目标

    start: 起始状态 [x,y,theta]
    goal: 目标状态 [x,y,theta]
    N: 优化步数
    dt: 时间步长
    v_bounds: 线速度约束 (v_min, v_max)
    w_bounds: 角速度约束 (w_min, w_max)
    obstacles: 障碍物列表，格式 [(cx,cy,r), ...]
    path_ref: 参考路径点列表 [(x1,y1), (x2,y2), ...]，可选

    返回: xs (N+1,3) 优化后状态序列, u_opt (N,2) 优化后控制序列, res 优化结果
    """
    pass
