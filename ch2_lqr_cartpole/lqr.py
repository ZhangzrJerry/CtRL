import numpy as np
from scipy.linalg import expm, solve_discrete_are


def discretize(A, B, dt):
    """将连续时间线性系统 x_dot = A x + B u 离散化（零阶保持，ZOH）。

    使用矩阵指数的方法：
            exp([A B; 0 0] * dt) = [Ad  Bd; 0  I]

    参数:
            A: 连续时间A矩阵，shape (n,n)
            B: 连续时间B矩阵，shape (n,m)
            dt: 采样时间（秒）

    返回:
            Ad, Bd: 离散时间矩阵，使得 x_{k+1} = Ad x_k + Bd u_k
    """
    A = np.asarray(A)
    B = np.asarray(B)
    n = A.shape[0]
    m = B.shape[1]

    # 构造块矩阵
    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B

    Mexp = expm(M * dt)

    Ad = Mexp[:n, :n]
    Bd = Mexp[:n, n:]
    return Ad, Bd


def finite_horizon_lqr(Ad, Bd, Q, R, N):
    """有限时域 LQR：通过反向递推计算时变增益序列。

    参数:
            Ad, Bd: 离散时间系统矩阵
            Q, R: 权重矩阵
            N: 规划步数（正整数）

    返回:
            K_seq: ndarray, 形状 (N, m, n)，K_seq[k] 为时刻 k 使用的反馈增益
            P_seq: ndarray, 形状 (N+1, n, n)，包含 P_0 .. P_N
    """
    P_seq = [None] * (N + 1)
    K_seq = [None] * N

    ###############################################
    ### TODO: 实现有限时域 LQR 反向递推算法 #########
    ###############################################

    return K_seq, P_seq
