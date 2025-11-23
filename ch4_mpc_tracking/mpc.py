import numpy as np
from scipy.optimize import minimize


def simulate_trajectory(x0, us, dt):
    """使用非线性模型前向仿真：x = [x,y,theta], u = [v,w]"""
    x = x0.copy()
    traj = [x.copy()]
    for v, w in us:
        x = step_dynamics(x, v, w, dt)
        traj.append(x.copy())
    return np.array(traj)


def step_dynamics(x, v, w, dt):
    xx, yy, th = x
    if abs(w) < 1e-8:
        x_new = xx + v * np.cos(th) * dt
        y_new = yy + v * np.sin(th) * dt
        th_new = th
    else:
        x_new = xx + (v / w) * (np.sin(th + w * dt) - np.sin(th))
        y_new = yy + (v / w) * (np.cos(th) - np.cos(th + w * dt))
        th_new = th + w * dt
    th_new = (th_new + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new, th_new])


def solve_mpc(x0, ref_traj, horizon=10, dt=0.05, vmax=1.5, wmax=1.5, Q=None, R=None):
    """
    TODO: 实现直接法MPC：优化整个控制序列（v0..vN-1, w0..wN-1）

    x0: 当前状态 (3,)
    ref_traj: 目标状态序列，长度 horizon+1，形状 (horizon+1, 3)
    返回： v0, w0, us_opt, pred_traj
        v0, w0: 首个控制量
        us_opt: 优化后的完整控制序列，长度 horizon，列表 [(v0,w0),...,(vN-1,wN-1)]
        pred_traj: 预测轨迹，形状 (horizon+1, 3)
    """
    v0, w0 = 0.0, 0.0
    us_opt = np.array([[v0, w0]] * horizon).flatten()
    pred_traj = simulate_trajectory(x0, us_opt.reshape(-1, 2), dt)

    return v0, w0, us_opt, pred_traj
