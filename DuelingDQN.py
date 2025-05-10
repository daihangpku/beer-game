import numpy as np
import matplotlib.pyplot as plt
import os
from env import Env
from agent import DuelingDQNAgent, DQNAgent
from train import train_dqn, test_agent
from plot_utils import plot_training_results, plot_test_results
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN训练和测试')
    parser.add_argument('--model', type=str, default='DuelingDQN', help='model name')
    parser.add_argument('--num_firms', type=int, default=3, help='企业数量')
    parser.add_argument('--num_episodes', type=int, default=5000, help='训练的episode数量')
    parser.add_argument('--max_steps', type=int, default=100, help='每个episode的最大步数')
    parser.add_argument('--eps_start', type=float, default=1.0, help='起始epsilon值')
    parser.add_argument('--eps_end', type=float, default=0.01, help='最小epsilon值')
    parser.add_argument('--eps_decay', type=float, default=0.995, help='epsilon衰减率')
    parser.add_argument('--state_size', type=int, default=3, help='每个企业的状态维度')
    parser.add_argument('--action_size', type=int, default=20, help='最大订单量')
    parser.add_argument('--firm_id', type=int, default=1, help='选择的企业ID进行训练')
    parser.add_argument('--dir', type=str, default='DuelingDQN', help='保存模型和图表的目录')
    parser.add_argument('--model_dir', type=str, default='models', help='保存模型的目录')
    parser.add_argument('--fig_dir', type=str, default='figures', help='保存图表的目录')
    parser.add_argument('--test_only', action='store_true', help='')
    args = parser.parse_args()
    # 创建保存模型和图表的目录
    fig_dir = f'figures/{args.dir}'
    args.fig_dir = fig_dir
    model_dir = f'models/{args.dir}'
    args.model_dir = model_dir
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 绘图显示负号
    plt.rcParams['font.size'] = 20  # 设置全局字体大小为14
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
    if args.model == 'DuelingDQN':
        agent = DuelingDQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size)
    elif args.model == 'DQN':
        agent = DQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size)
    scores = train_dqn(env, agent, args)

    test_agent(env, agent, num_episodes=10, args=args)