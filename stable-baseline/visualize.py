# visualize the training process of a Double DQN agent
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def load_tensorboard_data_by_index(log_dir, index):
    """Load data from TensorBoard logs for a specific index"""
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    # Locate TensorBoard event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    print(f"Found {len(event_files)} event files: {event_files}")
    
    if not event_files:
        print("No TensorBoard event files found! Make sure training has been run and logs are generated.")
        return None, None
    if index < 0 or index >= len(event_files):
        print(f"Index {index} is out of range for the available event files.")
        return None, None

    # Load the specified event file
    ea = event_accumulator.EventAccumulator(event_files[index])
    ea.Reload()
    
    # Get available scalar tags
    tags = ea.Tags()["scalars"]
    print(f"Available scalar tags: {tags}")
    
    # Extract reward and epsilon data
    reward_events = ea.Scalars("rollout/ep_rew_mean") if "rollout/ep_rew_mean" in tags else []
    epsilon_events = ea.Scalars("rollout/exploration_rate") if "rollout/exploration_rate" in tags else []
    
    return reward_events, epsilon_events

def load_latest_tensorboard_data(log_dir):
    """Load data from TensorBoard logs"""
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    # Locate TensorBoard event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    print(f"Found {len(event_files)} event files: {event_files}")
    
    if not event_files:
        print("No TensorBoard event files found! Make sure training has been run and logs are generated.")
        return None, None
    
    # Load the most recent event file
    ea = event_accumulator.EventAccumulator(event_files[-1])
    ea.Reload()
    
    # Get available scalar tags
    tags = ea.Tags()["scalars"]
    print(f"Available scalar tags: {tags}")
    
    # Extract reward and epsilon data
    reward_events = ea.Scalars("rollout/ep_rew_mean") if "rollout/ep_rew_mean" in tags else []
    epsilon_events = ea.Scalars("rollout/exploration_rate") if "rollout/exploration_rate" in tags else []
    
    return reward_events, epsilon_events

def create_dual_axis_training_plot(reward_events, epsilon_events, algorithm_name, save_dir=None):
    """Create plots for training progress"""
    if not reward_events:
        print("No reward data found!")
        return
    
    # Extract data from events
    reward_steps = [e.step for e in reward_events]
    reward_values = [e.value for e in reward_events]
    
    epsilon_steps = [e.step for e in epsilon_events] if epsilon_events else []
    epsilon_values = [e.value for e in epsilon_events] if epsilon_events else []
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot mean reward
    color1 = 'tab:blue'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Mean Reward per Episode', color=color1)
    line1 = ax1.plot(reward_steps, reward_values, color=color1, linewidth=2, label='Mean Reward')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot epsilon decay on secondary y-axis if available
    if epsilon_events:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Exploration Rate (Epsilon)', color=color2)
        line2 = ax2.plot(epsilon_steps, epsilon_values, color=color2, linewidth=2, label='Epsilon')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
    else:
        ax1.legend(loc='upper right')
        print("Note: No epsilon data found in logs")
    
    plt.title(f"Training Progress - {algorithm_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Show statistics
    print(f"\nTraining Statistics:")
    print(f"Total training steps: {max(reward_steps):,}")
    print(f"Initial mean reward: {reward_values[0]:.2f}")
    print(f"Final mean reward: {reward_values[-1]:.2f}")
    print(f"Best mean reward: {max(reward_values):.2f}")
    
    if epsilon_values:
        print(f"Initial epsilon: {epsilon_values[0]:.4f}")
        print(f"Final epsilon: {epsilon_values[-1]:.4f}")
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{algorithm_name}_training_progress.png"))

    plt.show()

def visualize_single_training_run(algorithm_name, log_dir, save_dir=None):
    """Main function to run the visualization"""
    # Load data
    reward_events, epsilon_events = load_latest_tensorboard_data(log_dir)
    
    if reward_events is None:
        return
    
    # Create plots
    create_dual_axis_training_plot(reward_events, epsilon_events, algorithm_name, save_dir=save_dir)

def visualize_averaged_training_runs(algorithm_name, log_dir):
    """Plot averaged training data"""
    reward_values_list = []
    epsilon_values_list = []
    for i in range(10):
        reward_events, epsilon_events = load_tensorboard_data_by_index(log_dir, i)
        if reward_events:
            reward_values = [e.value for e in reward_events]
            reward_values_list.append(reward_values)
        if epsilon_events:
            epsilon_values = [e.value for e in epsilon_events]
            epsilon_values_list.append(epsilon_values)

    if not reward_values_list or not epsilon_values_list:
        print("No valid data found for averaging.")
        return

    # Average the rewards and epsilon values
    avg_reward_values = np.mean(reward_values_list, axis=0)
    avg_epsilon_values = np.mean(epsilon_values_list, axis=0)

    # Create plots
    create_dual_axis_training_plot(avg_reward_values, avg_epsilon_values, algorithm_name)

if __name__ == "__main__":
    """plots the last training progress of DQN and Double DQN"""
    visualize_single_training_run("DQN", "./stable-baseline/dqn/train_logs/")
    # visualize_single_training_run("Double DQN", "./stable-baseline/double dqn/train_logs/")

    # Not sure if this works properly
    # visualize_averaged_training_runs("Averaged DQN", "./stable-baseline/dqn/train_logs/")
    # visualize_averaged_training_runs("Averaged Double DQN", "./stable-baseline/double dqn/train_logs/")