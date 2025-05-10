import numpy as np
from plot_utils import plot_training_results, plot_test_results, plot_mean_and_variance
from tqdm import tqdm
def train_dqn(env, agent, args):
    """
    训练DQN智能体
    
    :param env: 环境
    :param agent: DQN智能体
    :param num_episodes: 训练的episodes数量
    :param max_t: 每个episode的最大步数
    :param eps_start: 起始epsilon值
    :param eps_end: 最小epsilon值
    :param eps_decay: epsilon衰减率
    :return: 所有episode的奖励
    """
    num_episodes=args.num_episodes
    max_t=args.max_steps
    eps_start=args.eps_start
    eps_end=args.eps_end
    eps_decay=args.eps_decay
    model_dir=args.model_dir
    fig_dir=args.fig_dir
    scores = []  # 每个episode的总奖励
    mean = []
    variance = []
    idx = []
    eps = eps_start  # 初始epsilon值
    for i_episode in tqdm(range(1, num_episodes + 1), desc="Training Episodes"):
        state = env.reset()
        score = 0
        
        for t in range(max_t):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 使用智能体策略
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, eps)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)
            
            # 执行动作
            next_state, rewards, done = env.step(actions)
            
            # 该企业的奖励
            reward = rewards[agent.firm_id][0]
            
            # 保存经验并学习
            agent.step(state[agent.firm_id].reshape(1, -1), actions[agent.firm_id], reward, next_state[agent.firm_id].reshape(1, -1), done)
            
            # 更新状态和奖励
            state = next_state
            score += reward
            
            if done:
                break
        
        # 更新epsilon
        eps = max(eps_end, eps_decay * eps)
        
        # 记录分数
        scores.append(score)
        
        # 输出进度
        # if i_episode % 100 == 0:
        #     print(f'Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f} | Epsilon: {eps:.4f}')
        
        # 每隔一定episode保存模型
        if i_episode % 500 == 0:
            agent.save(f'{model_dir}/dqn_agent_firm_{agent.firm_id}_episode_{i_episode}.pth')
        if i_episode % 100 == 0:
            mean_score, variance_score = test_agent(env, agent, num_episodes=100, args=args, name=f'{i_episode}')
            mean.append(mean_score)
            variance.append(variance_score)
            idx.append(i_episode)
    # 训练结束后保存最终模型
    agent.save(f'{model_dir}/dqn_agent_firm_{agent.firm_id}_final.pth')
    plot_training_results(scores, fig_dir)
    plot_mean_and_variance(idx, mean, variance,  fig_dir)
    
    # 保存均值和方差
    with open(f'{fig_dir}/mean_variance_log.txt', 'w') as log_file:
        for i, (m, v) in enumerate(zip(mean, variance)):
            log_file.write(f'Episode {idx[i]}: Mean: {m}, Variance: {v}\n')
    return scores

def test_agent(env, agent, num_episodes=10, args=None, name="final"):
    """
    测试训练好的DQN智能体
    
    :param env: 环境
    :param agent: 训练好的DQN智能体
    :param num_episodes: 测试的episodes数量
    :return: 所有episode的奖励和详细信息
    """
    fig_dir = args.fig_dir

    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []
    
    for i_episode in tqdm(range(1, num_episodes + 1), desc="Testing Episodes"):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []
        
        for t in range(env.max_steps):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 使用智能体策略，不使用探索
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, epsilon=0.0)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)
            
            # 执行动作
            next_state, rewards, done = env.step(actions)
            
            # 记录关键指标
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])
            
            # 该企业的奖励
            reward = rewards[agent.firm_id][0]
            score += reward
            
            # 更新状态
            state = next_state
            
            if done:
                break
        
        # 记录分数和历史数据
        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)
        
        #print(f'Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}')
    mean_score, variance_score=plot_test_results(scores, inventory_history, orders_history, demand_history, satisfied_demand_history, fig_dir, name)
    return mean_score, variance_score
