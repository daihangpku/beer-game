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
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化 Dueling Q 网络
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # 第一层全连接
        self.fc2 = nn.Linear(64, 64)  # 第二层全连接
        
        # 状态值流 (V)
        self.value_stream = nn.Linear(64, 1)
        # 优势函数流 (A)
        self.advantage_stream = nn.Linear(64, action_size)
    
    def forward(self, state):
        """
        前向传播
        
        :param state: 输入状态
        :return: 各动作的 Q 值
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        # 计算状态值和优势函数
        value = self.value_stream(x)  # 状态值 V(s)
        advantage = self.advantage_stream(x)  # 优势函数 A(s, a)
        
        # 合并状态值和优势函数，计算 Q 值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        初始化优先经验回放缓冲区
        
        :param capacity: 缓冲区容量
        :param alpha: 优先级采样的权重，控制优先级的影响程度 (0表示完全随机采样，1表示完全按优先级采样)
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # 存储优先级
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done, td_error=1000.0):
        """
        添加经验到缓冲区
        
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        :param td_error: TD 误差，用于计算优先级
        """
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(float(abs(td_error) + 1e-5))  # 避免优先级为 0

    def sample(self, batch_size, beta=0.4):
        """
        从缓冲区采样一批经验
        
        :param batch_size: 批大小
        :param beta: 重要性采样的权重，修正采样偏差
        :return: 采样的经验和重要性采样权重
        """
        # 计算采样概率
        #print("Priorities:", self.priorities)
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # 按概率采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # 归一化

        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, td_errors):
        """
        更新采样的经验的优先级
        
        :param indices: 被采样的经验的索引
        :param td_errors: 对应的 TD 误差
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = float(abs(td_error) + 1e-5)

    def __len__(self):
        """
        获取缓冲区当前大小
        
        :return: 缓冲区大小
        """
        return len(self.buffer)