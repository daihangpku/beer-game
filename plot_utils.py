import numpy as np
import matplotlib.pyplot as plt

def plot_training_results(scores, fig_dir, window_size=100):
    """
    Plot training results
    
    :param scores: Rewards for each episode
    :param fig_dir: Directory to save the plots
    :param window_size: Window size for moving average
    """
    # Calculate moving average
    def moving_average(data, window_size):
        return [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
    
    avg_scores = moving_average(scores, window_size)
    
    # Plot original rewards and moving average
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label='Original Rewards')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label=f'Moving Average ({window_size} episodes)')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(f'{fig_dir}/training_rewards.png')
    plt.close()
    
    # Plot rewards focusing on values greater than 0
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label='Original Rewards')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label=f'Moving Average ({window_size} episodes)')
    plt.title('Rewards Focused on Positive Values')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(0, max(max(scores), max(avg_scores)) * 1.1)  # Focus on positive values
    plt.legend()
    plt.savefig(f'{fig_dir}/positive_rewards_focus.png')
    plt.close()

def plot_test_results(scores, inventory_history, orders_history, demand_history, satisfied_demand_history, fig_dir, name="final"):
    """
    Plot test results
    
    :param scores: Rewards for each episode
    :param inventory_history: Inventory history for each episode
    :param orders_history: Orders history for each episode
    :param demand_history: Demand history for each episode
    :param satisfied_demand_history: Satisfied demand history for each episode
    :param fig_dir: Directory to save the plots
    :param name: Suffix for the plot file name
    """
    # Calculate mean and variance
    mean_score = np.mean(scores)
    variance_score = np.var(scores)
    print(f"Test Results - Mean Reward: {mean_score:.2f}, Reward Variance: {variance_score:.2f}")
    
    # Calculate averages for plotting
    avg_inventory = np.mean(inventory_history, axis=0)
    avg_orders = np.mean(orders_history, axis=0)
    avg_demand = np.mean(demand_history, axis=0)
    avg_satisfied_demand = np.mean(satisfied_demand_history, axis=0)
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Inventory plot
    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title('Average Inventory')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Inventory')
    
    # Orders plot
    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title('Average Orders')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Orders')
    
    # Demand and satisfied demand plot
    axs[1, 0].plot(avg_demand, label='Demand')
    axs[1, 0].plot(avg_satisfied_demand, label='Satisfied Demand')
    axs[1, 0].set_title('Average Demand vs Satisfied Demand')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Quantity')
    axs[1, 0].legend()
    
    # Reward bar chart
    axs[1, 1].bar(range(len(scores)), scores)
    axs[1, 1].set_title('Test Episode Rewards')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/test_results_{name}.png')
    plt.close()
    return mean_score, variance_score

def plot_mean_and_variance(idx, mean_values, variance_values, fig_dir, name="mean_variance"):
    """
    Plot mean and variance changes
    
    :param idx: Index for the x-axis
    :param mean_values: List of mean values
    :param variance_values: List of variance values
    :param fig_dir: Directory to save the plots
    :param name: Plot file name
    """
    plt.figure(figsize=(10, 6))
    plt.plot(idx, mean_values, label='Mean', marker='o')
    plt.plot(idx, variance_values, label='Variance', marker='x')
    plt.title('Mean and Variance Changes')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{fig_dir}/{name}.png')
    plt.close()