import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化Q网络
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """
        前向传播
        
        :param state: 输入状态
        :return: 各动作的Q值
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        :param capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区采样一批经验
        
        :param batch_size: 批大小
        :return: 一批经验 (state, action, reward, next_state, done)
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """
        获取缓冲区当前大小
        
        :return: 缓冲区大小
        """
        return len(self.buffer)