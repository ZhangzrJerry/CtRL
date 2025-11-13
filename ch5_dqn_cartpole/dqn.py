import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    """经验回放缓冲区（简单实现）。"""

    def __init__(self, capacity=100000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    ##################################################
    ### TODO: 实现一个简单的前馈神经网络，作为 Q 网络 ###
    ##################################################
    pass


class DQNAgent:
    """DQN 智能体：包含行为选择、经验存储与网络更新逻辑。"""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.95,
        batch_size=64,
        buffer_size=100000,
        target_update=100,
        device=None,
    ):
        self.device = device or torch.device("cpu")
        self.q = QNet(state_dim, action_dim).to(self.device)
        self.q_target = QNet(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.AdamW(self.q.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.learn_step = 0
        self.target_update = target_update

    def select_action(self, state, eps=0.1):
        """epsilon-greedy 策略：state 为 numpy 数组"""
        if random.random() < eps:
            return random.uniform(-30, 30)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s)
            return qvals.item()

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        #########################################################
        ### TODO: 实现 DQN 的网络更新步骤，包括计算损失与梯度下降 ###
        ##########################################################
        return 0.0  # 返回当前的损失值


def calc_reward(state):
    ####################################################
    ### TODO: 实现一个简单的奖励函数，根据状态计算奖励值 ###
    ####################################################
    return 0.0  # 返回奖励值
