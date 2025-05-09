import numpy as np
import matplotlib.pyplot as plt
def plot_training_results(scores, window_size=100):
    """
    绘制训练结果
    
    :param scores: 每个episode的奖励
    :param window_size: 移动平均窗口大小
    """
    # 计算移动平均
    def moving_average(data, window_size):
        return [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
    
    avg_scores = moving_average(scores, window_size)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label='原始奖励')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label=f'{window_size}个episode的移动平均')
    plt.title('DQN训练过程中的奖励')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.legend()
    plt.savefig('figures/training_rewards.png')
    plt.close()

def plot_test_results(scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
    """
    绘制测试结果
    
    :param scores: 每个episode的奖励
    :param inventory_history: 每个episode的库存历史
    :param orders_history: 每个episode的订单历史
    :param demand_history: 每个episode的需求历史
    :param satisfied_demand_history: 每个episode的满足需求历史
    """
    # 计算平均值，用于绘图
    avg_inventory = np.mean(inventory_history, axis=0)
    avg_orders = np.mean(orders_history, axis=0)
    avg_demand = np.mean(demand_history, axis=0)
    avg_satisfied_demand = np.mean(satisfied_demand_history, axis=0)
    
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 库存图表
    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title('平均库存')
    axs[0, 0].set_xlabel('时间步')
    axs[0, 0].set_ylabel('库存量')
    
    # 订单图表
    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title('平均订单量')
    axs[0, 1].set_xlabel('时间步')
    axs[0, 1].set_ylabel('订单量')
    
    # 需求和满足需求图表
    axs[1, 0].plot(avg_demand, label='需求')
    axs[1, 0].plot(avg_satisfied_demand, label='满足的需求')
    axs[1, 0].set_title('平均需求 vs 满足的需求')
    axs[1, 0].set_xlabel('时间步')
    axs[1, 0].set_ylabel('数量')
    axs[1, 0].legend()
    
    # 奖励柱状图
    axs[1, 1].bar(range(len(scores)), scores)
    axs[1, 1].set_title('测试episode奖励')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('总奖励')
    
    plt.tight_layout()
    plt.savefig('figures/test_results.png')
    plt.close()
