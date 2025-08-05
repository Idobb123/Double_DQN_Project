import numpy as np
import glob
import os
from regularAgent import RegularAgent
from doubleAgent import DoubleAgent
import yaml
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import calculate_agent_groundtruth

def load_hyperparameters():
    """Load hyperparameters from YAML file"""
    with open('hyperparameters.yml', 'r') as file:
        return yaml.safe_load(file)

def load_value_function_data(log_dir, scalar_tag='evaluation/avg_value_function'):
    """Load value function data from TensorBoard logs"""
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        if scalar_tag in event_acc.Tags()['scalars']:
            scalar_events = event_acc.Scalars(scalar_tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            return steps, values
        else:
            print(f"Warning: {scalar_tag} not found in {log_dir}")
            return [], []
    except Exception as e:
        print(f"Error loading {log_dir}: {e}")
        return [], []

def load_agent_value_functions(agent_pattern):
    """Load value function data for all runs of a specific agent type"""
    log_dirs = glob.glob(f"logs/{agent_pattern}")
    
    if not log_dirs:
        print(f"No logs found matching pattern: logs/{agent_pattern}")
        return []
    
    all_value_data = []
    
    for log_dir in sorted(log_dirs):
        steps, values = load_value_function_data(log_dir)
        if steps and values:
            all_value_data.append((steps, values))
            print(f"  Loaded value function data from: {os.path.basename(log_dir)}")
    
    return all_value_data

def interpolate_value_functions(all_data, max_steps=None):
    """Interpolate value function data to common step grid"""
    if not all_data:
        return [], []
    
    # Find common step range
    all_steps = []
    for steps, _ in all_data:
        if steps:
            all_steps.extend(steps)
    
    if not all_steps:
        return [], []
    
    min_step = min(all_steps)
    if max_steps is None:
        max_step = min(max(steps) for steps, _ in all_data if steps)
    else:
        max_step = min(max_steps, min(max(steps) for steps, _ in all_data if steps))
    
    # Create common step grid
    common_steps = np.linspace(min_step, max_step, 500)  # Use fewer points for value function
    
    # Interpolate each run to common steps
    interpolated_values = []
    for steps, values in all_data:
        if len(steps) > 1 and len(values) > 1:
            interp_values = np.interp(common_steps, steps, values)
            interpolated_values.append(interp_values)
    
    return common_steps, interpolated_values

def create_value_function_comparison(config_name="lunarlander4-windy"):
    """Create comparison plot of value functions between DQN and DDQN"""
    print("\nCreating Value Function Comparison Plot...")
    print("=" * 50)
    print(f"Using configuration: {config_name}")
    
    # Get ground truth values
    print("Calculating ground truth value functions...")
    try:
        # dqn_groundtruth, ddqn_groundtruth = calculate_agent_groundtruth.get_agents_groundtruth(config=config_name, episodes=5, max_steps=1_000_000)
        dqn_groundtruth, ddqn_groundtruth = calculate_agent_groundtruth.get_agents_groundtruth(config=config_name, episodes=10)
        print(f"DQN Ground Truth Value: {dqn_groundtruth:.4f}")
        print(f"DDQN Ground Truth Value: {ddqn_groundtruth:.4f}")
    except Exception as e:
        print(f"Error calculating ground truth: {e}")
        dqn_groundtruth = None
        ddqn_groundtruth = None
    
    # Load DQN value function data
    print("Loading DQN value function data...")
    dqn_value_data = load_agent_value_functions("*regularagent*dqn*")
    
    # Load DDQN value function data
    print("Loading DDQN value function data...")
    ddqn_value_data = load_agent_value_functions("*doubleagent*ddqn*")
    
    if not dqn_value_data and not ddqn_value_data:
        print("No value function data found for either algorithm!")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    color_dqn = 'tab:blue'
    color_ddqn = 'tab:red'
    
    # Find x-axis range for ground truth lines
    x_min, x_max = 0, 1
    
    # Process DQN data
    if dqn_value_data:
        dqn_steps, dqn_interpolated = interpolate_value_functions(dqn_value_data)
        if dqn_interpolated:
            dqn_array = np.array(dqn_interpolated)
            dqn_mean = np.mean(dqn_array, axis=0)
            dqn_std = np.std(dqn_array, axis=0)
            
            # Update x-axis range
            x_min = min(x_min, dqn_steps[0]) if len(dqn_steps) > 0 else x_min
            x_max = max(x_max, dqn_steps[-1]) if len(dqn_steps) > 0 else x_max
            
            # Plot DQN mean and std
            ax.plot(dqn_steps, dqn_mean, color=color_dqn, linewidth=2, 
                   label=f'DQN Mean Value Function (n={len(dqn_interpolated)})', alpha=0.8)
            ax.fill_between(dqn_steps, dqn_mean - dqn_std, dqn_mean + dqn_std, 
                           color=color_dqn, alpha=0.2, label='DQN ± 1 Std')
            
            print(f"DQN: Processed {len(dqn_interpolated)} runs")
            print(f"DQN Final Value Function: {dqn_mean[-1]:.4f} ± {dqn_std[-1]:.4f}")
    
    # Process DDQN data
    if ddqn_value_data:
        ddqn_steps, ddqn_interpolated = interpolate_value_functions(ddqn_value_data)
        if ddqn_interpolated:
            ddqn_array = np.array(ddqn_interpolated)
            ddqn_mean = np.mean(ddqn_array, axis=0)
            ddqn_std = np.std(ddqn_array, axis=0)
            
            # Update x-axis range
            x_min = min(x_min, ddqn_steps[0]) if len(ddqn_steps) > 0 else x_min
            x_max = max(x_max, ddqn_steps[-1]) if len(ddqn_steps) > 0 else x_max
            
            # Plot DDQN mean and std
            ax.plot(ddqn_steps, ddqn_mean, color=color_ddqn, linewidth=2, 
                   label=f'Double DQN Mean Value Function (n={len(ddqn_interpolated)})', alpha=0.8)
            ax.fill_between(ddqn_steps, ddqn_mean - ddqn_std, ddqn_mean + ddqn_std, 
                           color=color_ddqn, alpha=0.2, label='Double DQN ± 1 Std')
            
            print(f"DDQN: Processed {len(ddqn_interpolated)} runs")
            print(f"DDQN Final Value Function: {ddqn_mean[-1]:.4f} ± {ddqn_std[-1]:.4f}")
    
    # Add ground truth horizontal lines
    if dqn_groundtruth is not None:
        ax.axhline(y=dqn_groundtruth, color=color_dqn, linestyle='--', linewidth=2, 
                  alpha=0.9, label=f'DQN Ground Truth ({dqn_groundtruth:.3f})')
        print(f"Added DQN ground truth line at y={dqn_groundtruth:.4f}")
    
    if ddqn_groundtruth is not None:
        ax.axhline(y=ddqn_groundtruth, color=color_ddqn, linestyle='--', linewidth=2, 
                  alpha=0.9, label=f'DDQN Ground Truth ({ddqn_groundtruth:.3f})')
        print(f"Added DDQN ground truth line at y={ddqn_groundtruth:.4f}")
    
    # Formatting
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Average Value Function (Max Q-Values)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.title('DQN vs Double DQN Value Function Comparison\nAverage Max Q-Values During Training + Ground Truth', 
              fontsize=14, fontweight='bold')
    
    # Summary comparison
    if dqn_value_data and ddqn_value_data and dqn_interpolated and ddqn_interpolated:
        dqn_final = dqn_mean[-1]
        ddqn_final = ddqn_mean[-1]
        improvement = ddqn_final - dqn_final
        
        print(f"\nValue Function Comparison Summary:")
        print(f"DQN Final Value Function:    {dqn_final:.4f}")
        print(f"DDQN Final Value Function:   {ddqn_final:.4f}")
        print(f"Value Function Improvement:  {improvement:+.4f}")
        
        # Compare with ground truth if available
        if dqn_groundtruth is not None:
            dqn_error = abs(dqn_final - dqn_groundtruth)
            print(f"DQN Error from Ground Truth: {dqn_error:.4f}")
        
        if ddqn_groundtruth is not None:
            ddqn_error = abs(ddqn_final - ddqn_groundtruth)
            print(f"DDQN Error from Ground Truth: {ddqn_error:.4f}")
        
        if improvement > 0:
            print("🎯 DDQN achieves higher value function estimates!")
        elif improvement < 0:
            print("🎯 DQN achieves higher value function estimates!")
        else:
            print("🤝 Both algorithms achieve similar value function estimates!")
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "output/value_function_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nValue function comparison plot saved to: {output_file}")
    
    plt.show()

def evaluate_agent_models(agent_class, model_pattern, hyperparameter_set, episodes=100):
    """Evaluate all models for a specific agent type"""
    print(f"\nEvaluating {agent_class.__name__} models...")
    print("=" * 50)
    
    # Find all model files matching the pattern
    model_files = glob.glob(f"output/{model_pattern}")
    model_files.sort()  # Sort to ensure consistent order
    
    if not model_files:
        print(f"No models found matching pattern: output/{model_pattern}")
        return []
    
    print(f"Found {len(model_files)} models:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")
    
    # Initialize agent
    agent = agent_class(hyperparameter_set)
    
    all_results = []
    
    # Evaluate each model
    for i, model_file in enumerate(model_files, 1):
        print(f"\nEvaluating model {i}/{len(model_files)}: {os.path.basename(model_file)}")
        
        try:
            # Create environment for testing
            env = agent.create_env(agent.env_id, render_mode=None)
            
            # Load the trained model
            from dqn import DQN
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")  # Force CPU for compatibility
            
            num_features = env.observation_space.shape[0]
            num_actions = env.action_space.n
            
            policy_dqn = DQN(num_features, num_actions, agent.first_layer_dim, agent.second_layer_dim).to(device)
            policy_dqn.load_state_dict(torch.load(model_file, map_location=device))
            policy_dqn.eval()  # Set to evaluation mode
            
            episode_rewards = []
            episode_lengths = []
            successful_landings = 0
            
            # Run evaluation episodes
            for episode in range(episodes):
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32).to(device)
                episode_reward = 0
                episode_length = 0
                
                max_steps = 1000  # Prevent infinite episodes
                for step in range(max_steps):
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
                    
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                    
                    state = new_state
                    episode_reward += reward
                    episode_length += 1
                    
                    if terminated or truncated:
                        # Check if it's a successful landing (reward >= 200 is typically successful)
                        if episode_reward >= 200:
                            successful_landings += 1
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            env.close()
            
            # Calculate statistics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            success_rate = (successful_landings / episodes) * 100
            
            result = {
                'model_file': os.path.basename(model_file),
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_length': mean_length,
                'success_rate': success_rate,
                'episodes': episodes,
                'rewards': episode_rewards
            }
            
            all_results.append(result)
            
            print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Mean Episode Length: {mean_length:.1f}")
            
        except Exception as e:
            print(f"  Error evaluating {model_file}: {e}")
            continue
    
    return all_results

def print_group_statistics(results, algorithm_name):
    """Print aggregate statistics for a group of models"""
    if not results:
        print(f"No results available for {algorithm_name}")
        return None
    
    print(f"\n{algorithm_name} GROUP STATISTICS:")
    print("=" * 60)
    
    # Aggregate statistics across all models
    all_mean_rewards = [r['mean_reward'] for r in results]
    all_success_rates = [r['success_rate'] for r in results]
    all_mean_lengths = [r['mean_length'] for r in results]
    
    # All individual episode rewards for overall statistics
    all_episode_rewards = []
    for result in results:
        all_episode_rewards.extend(result['rewards'])
    
    group_stats = {
        'mean_reward_across_models': np.mean(all_mean_rewards),
        'std_reward_across_models': np.std(all_mean_rewards),
        'mean_success_rate': np.mean(all_success_rates),
        'std_success_rate': np.std(all_success_rates),
        'overall_mean_reward': np.mean(all_episode_rewards),
        'overall_std_reward': np.std(all_episode_rewards),
        'best_model_reward': max(all_mean_rewards),
        'worst_model_reward': min(all_mean_rewards),
        'num_models': len(results),
        'total_episodes': sum(r['episodes'] for r in results)
    }
    
    print(f"Number of Models: {group_stats['num_models']}")
    print(f"Total Episodes: {group_stats['total_episodes']}")
    print(f"")
    print(f"PERFORMANCE ACROSS MODELS:")
    print(f"  Mean Reward per Model: {group_stats['mean_reward_across_models']:.2f} ± {group_stats['std_reward_across_models']:.2f}")
    print(f"  Success Rate per Model: {group_stats['mean_success_rate']:.1f}% ± {group_stats['std_success_rate']:.1f}%")
    print(f"  Best Model Performance: {group_stats['best_model_reward']:.2f}")
    print(f"  Worst Model Performance: {group_stats['worst_model_reward']:.2f}")
    print(f"")
    print(f"OVERALL PERFORMANCE (All Episodes):")
    print(f"  Overall Mean Reward: {group_stats['overall_mean_reward']:.2f} ± {group_stats['overall_std_reward']:.2f}")
    
    return group_stats

def compare_algorithms(config_name="lunarlander4-windy"):
    """Main comparison function"""
    print("LUNARLANDER DQN vs DOUBLE DQN COMPARISON")
    print("=" * 80)
    print(f"Using configuration: {config_name}")
    print("Evaluating trained models on 100 episodes each...")
    
    # Load hyperparameters
    hyperparams = load_hyperparameters()
    
    # Validate that the provided config exists
    if config_name not in hyperparams:
        print(f"Error: Configuration '{config_name}' not found in hyperparameters.yml!")
        print(f"Available configurations: {list(hyperparams.keys())}")
        return
    
    config = hyperparams[config_name]
    env_id = config.get('env_id')
    print(f"Environment: {env_id}")
    
    # Evaluate DQN models
    dqn_results = evaluate_agent_models(
        RegularAgent, 
        f"{env_id}_dqn_*.pt", 
        config_name, 
        episodes=100
    )
    
    # Evaluate DDQN models
    ddqn_results = evaluate_agent_models(
        DoubleAgent, 
        f"{env_id}_ddqn_*.pt", 
        config_name, 
        episodes=100
    )
    
    # Print group statistics
    dqn_stats = print_group_statistics(dqn_results, "DQN")
    ddqn_stats = print_group_statistics(ddqn_results, "DOUBLE DQN")
    
    # Final comparison
    if dqn_stats and ddqn_stats:
        print(f"\nFINAL COMPARISON:")
        print("=" * 60)
        
        reward_improvement = ddqn_stats['overall_mean_reward'] - dqn_stats['overall_mean_reward']
        success_improvement = ddqn_stats['mean_success_rate'] - dqn_stats['mean_success_rate']
        
        print(f"DQN Overall Performance:    {dqn_stats['overall_mean_reward']:.2f} ± {dqn_stats['overall_std_reward']:.2f}")
        print(f"DDQN Overall Performance:   {ddqn_stats['overall_mean_reward']:.2f} ± {ddqn_stats['overall_std_reward']:.2f}")
        print(f"")
        print(f"Reward Improvement:         {reward_improvement:+.2f}")
        print(f"Success Rate Improvement:   {success_improvement:+.1f}%")
        print(f"")
        
        if reward_improvement > 0:
            print("🏆 DOUBLE DQN performs better overall!")
        elif reward_improvement < 0:
            print("🏆 DQN performs better overall!")
        else:
            print("🤝 Both algorithms perform equally!")
        
        # Statistical significance (basic test)
        from scipy import stats
        if len(dqn_results) > 1 and len(ddqn_results) > 1:
            dqn_means = [r['mean_reward'] for r in dqn_results]
            ddqn_means = [r['mean_reward'] for r in ddqn_results]
            
            try:
                t_stat, p_value = stats.ttest_ind(ddqn_means, dqn_means)
                print(f"")
                print(f"Statistical Test (t-test on model means):")
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.3f}")
                if p_value < 0.05:
                    print(f"  Result: Statistically significant difference (p < 0.05)")
                else:
                    print(f"  Result: No statistically significant difference (p >= 0.05)")
            except ImportError:
                print("Note: Install scipy for statistical significance testing")
    
    # Create value function comparison plot
    create_value_function_comparison(config_name)

if __name__ == "__main__":
    
    config_name = "cartpole1"  # Default configuration
    # config_name = "lunarlander4-windy"  # Default configuration
    # config_name = "flappybird1"  # Example configuration
    
    compare_algorithms(config_name)
