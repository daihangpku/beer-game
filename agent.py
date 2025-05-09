import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from network import QNetwork, ReplayBuffer
class DQNAgent:
    def __init__(self, state_size, action_size, firm_id, max_order=20, buffer_size=10000, batch_size=64, gamma=0.99, 
                 learning_rate=1e-3, tau=1e-3, update_every=4):
        """
        初始化DQN智能体
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        :param firm_id: 企业ID，用于标识训练哪个企业
        :param max_order: 最大订单量，用于离散化动作空间
        :param buffer_size: 回放缓冲区大小
        :param batch_size: 批大小
        :param gamma: 折扣因子
        :param learning_rate: 学习率
        :param tau: 软更新参数
        :param update_every: 更新目标网络的频率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learning_step = 0
        
        # 创建Q网络和目标网络
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 跟踪训练进度
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """
        添加经验到回放缓冲区并按需学习
        
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        # 添加经验到回放缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 每隔一定步数学习
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def act(self, state, epsilon=0.0):
        """
        根据当前状态选择动作
        
        :param state: 当前状态
        :param epsilon: epsilon-贪婪策略参数
        :return: 选择的动作
        """
        # 从3维numpy数组转换为1维向量
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
        
        # 切换到评估模式
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        # 切换回训练模式
        self.q_network.train()
        
        # epsilon-贪婪策略
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy()) + 1  # +1 因为我们的动作从1开始
        else:
            return random.randint(1, self.max_order)
    
    def learn(self, experiences):
        """
        从经验批次中学习
        
        :param experiences: (state, action, reward, next_state, done) 元组
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # 转换为torch张量
        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float()
        actions = torch.from_numpy(np.vstack([a-1 for a in actions])).long()  # -1 因为我们的动作从1开始，但索引从0开始
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack([ns.flatten() for ns in next_states])).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()
        
        # 从目标网络获取下一个状态的最大预测Q值
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 计算目标Q值
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # 获取当前Q值估计
        Q_expected = self.q_network(states).gather(1, actions)
        
        # 计算损失
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # 最小化损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            self.soft_update()
        
        return loss.item()
    
    def soft_update(self):
        """
        软更新目标网络参数：θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        保存模型参数
        
        :param filename: 文件名
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 保存模型状态字典
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"模型已保存到 {filename}")
    
    def load(self, filename):
        """
        加载模型参数
        
        :param filename: 文件名
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"从 {filename} 加载了模型")
            return True
        return False