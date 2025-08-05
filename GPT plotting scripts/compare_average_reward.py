import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def load_tensorboard_data(log_dir):
    """Load scalar data from TensorBoard log directory"""
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        scalar_tags = event_acc.Tags()['scalars']
        
        data = {}
        for tag in scalar_tags:
            scalar_events = event_acc.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
        
        return data
    except Exception as e:
        print(f"Error loading {log_dir}: {str(e)}")
        return {}

def interpolate_data_to_common_steps(data_list, step_interval=1000):
    """Interpolate all data series to common steps for averaging"""
    if not data_list:
        return np.array([]), []
    
    # Find the common range of steps
    all_steps = []
    for data in data_list:
        if len(data['steps']) > 0:
            all_steps.extend(data['steps'])
    
    if not all_steps:
        return np.array([]), []
    
    min_step = min(all_steps)
    max_step = max(all_steps)
    
    # Create common step points
    common_steps = np.arange(min_step, max_step + step_interval, step_interval)
    
    # Interpolate each series to common steps
    interpolated_data = []
    for data in data_list:
        if len(data['steps']) > 1 and len(data['values']) > 1:
            interpolated_values = np.interp(common_steps, data['steps'], data['values'])
            interpolated_data.append(interpolated_values)
        else:
            print(f"Skipping series with insufficient data points")
    
    return common_steps, interpolated_data

def compare_dqn_vs_ddqn(logs_dir='logs', output_dir='output'):
    """Compare DQN vs DDQN mean rewards"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all log directories
    log_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    
    # Separate DQN and DDQN logs
    dqn_data = []
    ddqn_data = []
    
    print("Loading TensorBoard logs...")
    
    for log_dir in log_dirs:
        log_path = os.path.join(logs_dir, log_dir)
        
        # Determine algorithm type from directory name
        if 'dqn' in log_dir.lower() and 'ddqn' not in log_dir.lower():
            algorithm_type = 'DQN'
            target_list = dqn_data
        elif 'ddqn' in log_dir.lower():
            algorithm_type = 'DDQN'
            target_list = ddqn_data
        else:
            print(f"Skipping {log_dir}: cannot determine algorithm type")
            continue
        
        print(f"Loading {algorithm_type}: {log_dir}")
        
        # Load data
        data = load_tensorboard_data(log_path)
        
        # Extract mean episode reward data
        if 'rollout/ep_rew_mean' in data:
            reward_data = data['rollout/ep_rew_mean']
            target_list.append({
                'steps': reward_data['steps'],
                'values': reward_data['values'],
                'log_name': log_dir
            })
            print(f"  Loaded {len(reward_data['steps'])} data points")
        else:
            print(f"  No reward data found in {log_dir}")
    
    print(f"\nFound {len(dqn_data)} DQN runs and {len(ddqn_data)} DDQN runs")
    
    if len(dqn_data) == 0 and len(ddqn_data) == 0:
        print("No valid data found!")
        return
    
    # Create the comparison plot
    plt.figure(figsize=(12, 8))
    
    # Process DQN data
    if len(dqn_data) > 0:
        print("\nProcessing DQN data...")
        dqn_steps, dqn_interpolated = interpolate_data_to_common_steps(dqn_data)
        
        if len(dqn_interpolated) > 0:
            # Convert to numpy array for statistics
            dqn_matrix = np.array(dqn_interpolated)
            dqn_mean = np.mean(dqn_matrix, axis=0)
            dqn_std = np.std(dqn_matrix, axis=0)
            
            # Plot DQN mean with standard deviation
            plt.plot(dqn_steps, dqn_mean, color='blue', linewidth=3, label=f'DQN Mean (n={len(dqn_data)})')
            plt.fill_between(dqn_steps, dqn_mean - dqn_std, dqn_mean + dqn_std, 
                           color='blue', alpha=0.2, label=f'DQN ± 1 std')
            
            print(f"DQN final mean reward: {dqn_mean[-1]:.2f} ± {dqn_std[-1]:.2f}")
    
    # Process DDQN data
    if len(ddqn_data) > 0:
        print("\nProcessing DDQN data...")
        ddqn_steps, ddqn_interpolated = interpolate_data_to_common_steps(ddqn_data)
        
        if len(ddqn_interpolated) > 0:
            # Convert to numpy array for statistics
            ddqn_matrix = np.array(ddqn_interpolated)
            ddqn_mean = np.mean(ddqn_matrix, axis=0)
            ddqn_std = np.std(ddqn_matrix, axis=0)
            
            # Plot DDQN mean with standard deviation
            plt.plot(ddqn_steps, ddqn_mean, color='red', linewidth=3, label=f'DDQN Mean (n={len(ddqn_data)})')
            plt.fill_between(ddqn_steps, ddqn_mean - ddqn_std, ddqn_mean + ddqn_std, 
                           color='red', alpha=0.2, label=f'DDQN ± 1 std')
            
            print(f"DDQN final mean reward: {ddqn_mean[-1]:.2f} ± {ddqn_std[-1]:.2f}")
    
    # Configure the plot
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Mean Episode Reward (100-ep)', fontsize=12)
    plt.title('DQN vs DDQN Performance Comparison', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics text box
    stats_text = "Performance Summary:\\n"
    if len(dqn_data) > 0 and len(dqn_interpolated) > 0:
        stats_text += f"DQN: {dqn_mean[-1]:.2f} ± {dqn_std[-1]:.2f} (n={len(dqn_data)})\\n"
    if len(ddqn_data) > 0 and len(ddqn_interpolated) > 0:
        stats_text += f"DDQN: {ddqn_mean[-1]:.2f} ± {ddqn_std[-1]:.2f} (n={len(ddqn_data)})\\n"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    output_filename = f"{output_dir}/DQN_vs_DDQN_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\\nSaved comparison plot: {output_filename}")
    
    plt.show()

def main():
    """Main function"""
    print("=== DQN vs DDQN Mean Reward Comparison ===")
    compare_dqn_vs_ddqn()
    print("\\nComparison complete!")

if __name__ == "__main__":
    main()
