import numpy as np
import matplotlib.pyplot as plt
import os
from env import Env
from agent import DQNAgent
from train import train_dqn, test_agent
from plot_utils import plot_training_results, plot_test_results

if __name__ == "__main__":
    # 创建保存模型和图表的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # 初始化环境参数
    num_firms = 3  # 假设有3个企业
    p = [10, 9, 8]  # 价格列表
    h = 0.5  # 库存持有成本
    c = 2  # 损失销售成本
    initial_inventory = 100  # 初始库存
    poisson_lambda = 10  # 泊松分布的均值
    max_steps = 100  # 每个episode的最大步数
    
    # 创建仿真环境
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)
    
    # 为第二个企业创建DQN智能体
    firm_id = 1  # 选择第二个企业进行训练
    state_size = 3  # 每个企业的状态维度：订单、满足的需求和库存
    action_size = 20  # 假设最大订单量为20
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size)
    
    # 训练DQN智能体
    scores = train_dqn(env, agent, num_episodes=2000, max_t=max_steps, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 绘图显示负号

    # 绘制训练结果
    plot_training_results(scores)
    
    # 测试训练好的智能体
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agent, num_episodes=10)
    
    # 绘制测试结果
    plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history)