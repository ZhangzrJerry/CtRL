import numpy as np


class CartPole:
    """CartPole 系统模型。

    状态向量按顺序为 [x, x_dot, theta, theta_dot]
    其中 theta=0 表示倒立平衡（杆竖直向上）。

    本模块提供连续时间动力学、数值线性化以及基于连续动力学的离散步进函数。
    """

    def __init__(self, M=1.0, m=0.1, l=0.5, g=9.81):
        # 小车质量 M，摆杆质量 m，摆长 l，重力加速度 g
        self.M = M
        self.m = m
        self.l = l
        self.g = g

    def dynamics(self, state, u):
        """连续时间动力学：返回状态导数 x_dot = f(x, u)。

        参数:
            state: 长度为4的向量 [x, x_dot, theta, theta_dot]
            u: 标量，施加在小车上的水平力

        返回:
            ndarray, 形状(4,), 对应 [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x, x_dot, theta, theta_dot = state

        M = self.M
        m = self.m
        l = self.l
        g = self.g

        # 方便起见，引入 sin/cos
        s = np.sin(theta)
        c = np.cos(theta)

        # 常见的 cart-pole 非线性动力学（参照常见教材与开源实现）
        # 先构造临时量 temp = (u + m*l*theta_dot^2*sin(theta)) / (M + m)
        # 然后解出 theta_ddot，再得到 x_ddot
        denom = M + m
        temp = (u + m * l * theta_dot * theta_dot * s) / denom

        theta_dd = (g * s - c * temp) / (l * (4.0 / 3.0 - (m * c * c) / denom))
        x_dd = temp - (m * l * theta_dd * c) / denom

        return np.array([x_dot, x_dd, theta_dot, theta_dd])

    def linearize(self, x_eq, u_eq, eps=1e-6):
        """对任意工作点进行数值线性化，返回连续时间的 A, B 矩阵。

        使用有限差分计算雅可比矩阵：A = df/dx, B = df/du

        参数:
            x_eq: 状态平衡点，长度4向量
            u_eq: 输入平衡点（标量）
            eps: 有限差分扰动大小

        返回:
            A, B 矩阵，形状分别为 (4,4) 和 (4,1)
        """
        x_eq = np.asarray(x_eq, dtype=float)
        n = x_eq.size

        f0 = self.dynamics(x_eq, u_eq)

        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f1 = self.dynamics(x_eq + dx, u_eq)
            A[:, i] = (f1 - f0) / eps

        # B 矩阵（列向量）
        du = eps
        f2 = self.dynamics(x_eq, u_eq + du)
        B = ((f2 - f0) / du).reshape(n, 1)

        return A, B

    def discrete_step(self, state, u, dt):
        """离散时间步进：给定当前状态 state 和保持输入 u（零阶保持），前进 dt 后的状态。

        这里直接用经典的 RK4 对连续动力学积分以得到离散映射 f_d(x,u;dt)，
        使得 x_{k+1} = discrete_step(x_k, u_k, dt)。

        这样可以在保留非线性的同时以任意采样时间进行仿真。
        """

        # 使用四阶龙格-库塔法（RK4）进行一步积分
        def _f(x, u_local):
            return self.dynamics(x, u_local)

        k1 = _f(state, u)
        k2 = _f(state + 0.5 * dt * k1, u)
        k3 = _f(state + 0.5 * dt * k2, u)
        k4 = _f(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step(f, x, u, dt):
    """通用的 RK4 单步积分器（保留以兼容旧调用）。"""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def saturate(u, umax):
    """控制输入饱和函数，返回裁剪后的输入。"""
    return np.clip(u, -umax, umax)
