import numpy as np


class EkfSlam:
    """带有协方差矩阵可视化的 EKF SLAM 实现"""

    def __init__(self, initial_state, num_landmarks, Q, R):
        self.mu = np.zeros(3 + 2 * num_landmarks)
        self.mu[:3] = initial_state
        # 初始化协方差：为未观测地标设置较大的初始不确定性
        self.Sigma = np.eye(3 + 2 * num_landmarks) * 1e-4
        for i in range(num_landmarks):
            idx = 3 + 2 * i
            self.Sigma[idx : idx + 2, idx : idx + 2] = (
                np.eye(2) * 100.0
            )  # 大地标初始不确定性

        self.Q = Q
        self.R = R
        self.num_landmarks = num_landmarks
        self.observed_landmarks = set()

        # 存储协方差历史用于可视化
        self.covariance_history = []

    def motion_model(self, state, u, dt):
        """运动模型"""
        x, y, theta = state[:3]
        v, w = u

        ###############################################
        ### TODO: 实现运动模型，返回新的状态          ###
        ###############################################
        if abs(w) < 1e-6:
            x_new = 0
            y_new = 0
            theta_new = 0
        else:
            x_new = 0
            y_new = 0
            theta_new = 0

        return np.array([x_new, y_new, theta_new])

    def prediction_step(self, u, dt):
        """预测步骤：只更新与机器人状态相关的协方差分量"""
        #################################################
        ### TODO: 实现预测步骤，更新状态均值和协方差矩阵 ###
        #################################################

    def update_step(self, observations):
        """更新步骤：处理观测并更新已观测地标和协方差"""
        if not observations:
            return

        x, y, theta = self.mu[:3]

        for landmark_id, (r, phi) in observations.items():
            landmark_idx = 3 + 2 * landmark_id

            if landmark_id not in self.observed_landmarks:
                ###############################################
                ### TODO: 初始化新观测到的地标位置            ###
                ###############################################
                pass

            ###############################################
            ### TODO: 计算观测预测值和雅可比矩阵 H        ###
            ###############################################

            ###############################################
            ### TODO: 计算卡尔曼增益 K 并更新状态和协方差  ###
            ###############################################

        # 存储当前协方差用于可视化
        self.covariance_history.append(self.Sigma.copy())
