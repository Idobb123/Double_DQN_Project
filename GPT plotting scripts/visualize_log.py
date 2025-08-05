import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
from pathlib import Path

def load_tensorboard_data(log_dir):
    """
    Load data from TensorBoard log directory
    
    Args:
        log_dir: Path to the TensorBoard log directory
        
    Returns:
        dict: Dictionary containing the scalar data
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get available scalar tags
    scalar_tags = event_acc.Tags()['scalars']
    print(f"Available scalar tags in {log_dir}: {scalar_tags}")
    
    data = {}
    for tag in scalar_tags:
        scalar_events = event_acc.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    
    return data

def parse_log_directory_name(log_dir_name):
    """
    Parse the log directory name to extract environment, agent type, and index
    
    Args:
        log_dir_name: Name of the log directory
        
    Returns:
        dict: Parsed information
    """
    # Pattern: environment_agenttype_index_algorithm
    # Example: LunarLander-v2_regularagent_001_dqn
    parts = log_dir_name.split('_')
    
    if len(parts) >= 4:
        env_name = parts[0]
        agent_type = parts[1]
        index = parts[2]
        algorithm = parts[3]
        
        return {
            'environment': env_name,
            'agent_type': agent_type,
            'index': index,
            'algorithm': algorithm,
            'full_name': log_dir_name
        }
    else:
        return {
            'environment': 'unknown',
            'agent_type': 'unknown',
            'index': '000',
            'algorithm': 'unknown',
            'full_name': log_dir_name
        }

def create_individual_plots(logs_base_dir='logs', output_dir='output'):
    """
    Create individual plots for each log directory showing epsilon decay and mean episode reward
    
    Args:
        logs_base_dir: Base directory containing TensorBoard logs
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all log directories
    log_dirs = [d for d in os.listdir(logs_base_dir) 
                if os.path.isdir(os.path.join(logs_base_dir, d))]
    
    print(f"Found {len(log_dirs)} log directories: {log_dirs}")
    
    for log_dir_name in log_dirs:
        log_path = os.path.join(logs_base_dir, log_dir_name)
        
        try:
            print(f"\nProcessing: {log_dir_name}")
            
            # Load TensorBoard data
            data = load_tensorboard_data(log_path)
            
            # Parse directory name
            info = parse_log_directory_name(log_dir_name)
            
            # Create plot with two subplots: overlay plot and value function plot
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 8))
            
            # ===== LEFT SUBPLOT: Mean Episode Reward + Epsilon Overlay =====
            # Plot Mean Episode Reward on primary y-axis
            color1 = 'tab:blue'
            ax1.set_xlabel('Training Steps', fontsize=12)
            ax1.set_ylabel('Mean Episode Reward (100-ep)', color=color1, fontsize=12)
            
            reward_plotted = False
            if 'rollout/ep_rew_mean' in data:
                reward_data = data['rollout/ep_rew_mean']
                line1 = ax1.plot(reward_data['steps'], reward_data['values'], 
                        color=color1, linewidth=2, label='Mean Episode Reward (100-ep)')
                ax1.tick_params(axis='y', labelcolor=color1)
                reward_plotted = True
                
                # Add reward statistics
                final_reward = reward_data['values'][-1] if len(reward_data['values']) > 0 else 0
                max_reward = np.max(reward_data['values']) if len(reward_data['values']) > 0 else 0
            else:
                ax1.text(0.5, 0.5, 'No mean episode reward data found', 
                        ha='center', va='center', transform=ax1.transAxes, color=color1)
                final_reward = 0
                max_reward = 0
            
            # Plot Epsilon Decay on secondary y-axis
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Exploration Rate (Epsilon)', color=color2, fontsize=12)
            
            epsilon_plotted = False
            if 'rollout/exploration_rate' in data:
                epsilon_data = data['rollout/exploration_rate']
                line2 = ax2.plot(epsilon_data['steps'], epsilon_data['values'], 
                        color=color2, linewidth=2, alpha=0.8, label='Exploration Rate (Epsilon)')
                ax2.tick_params(axis='y', labelcolor=color2)
                epsilon_plotted = True
                
                # Add epsilon statistics
                initial_epsilon = epsilon_data['values'][0] if len(epsilon_data['values']) > 0 else 0
                final_epsilon = epsilon_data['values'][-1] if len(epsilon_data['values']) > 0 else 0
            else:
                ax2.text(0.5, 0.3, 'No epsilon data found', 
                        ha='center', va='center', transform=ax2.transAxes, color=color2)
                initial_epsilon = 0
                final_epsilon = 0
            
            # Add combined legend
            lines = []
            labels = []
            if reward_plotted and 'rollout/ep_rew_mean' in data:
                lines.extend(line1)
                labels.append('Mean Episode Reward (100-ep)')
            if epsilon_plotted and 'rollout/exploration_rate' in data:
                lines.extend(line2)
                labels.append('Exploration Rate (Epsilon)')
            
            if lines:
                ax1.legend(lines, labels, loc='upper right')
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'Reward - Final: {final_reward:.2f}, Max: {max_reward:.2f}\n'
            stats_text += f'Epsilon - Initial: {initial_epsilon:.4f}, Final: {final_epsilon:.4f}'
            ax1.text(0.02, 0.98, stats_text, 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax1.set_title('Reward & Exploration Rate', fontsize=14, fontweight='bold')
            
            # ===== RIGHT SUBPLOT: Value Function Estimates =====
            value_plotted = False
            if 'evaluation/avg_value_function' in data:
                value_data = data['evaluation/avg_value_function']
                color3 = 'tab:green'
                ax3.plot(value_data['steps'], value_data['values'], 
                        color=color3, linewidth=2, marker='o', markersize=4,
                        label='Average Value Function')
                ax3.set_xlabel('Training Steps', fontsize=12)
                ax3.set_ylabel('Average Value Function', color=color3, fontsize=12)
                ax3.tick_params(axis='y', labelcolor=color3)
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='lower right')
                value_plotted = True
                
                # Add value function statistics
                initial_value = value_data['values'][0] if len(value_data['values']) > 0 else 0
                final_value = value_data['values'][-1] if len(value_data['values']) > 0 else 0
                max_value = np.max(value_data['values']) if len(value_data['values']) > 0 else 0
                
                value_stats_text = f'Initial: {initial_value:.3f}\nFinal: {final_value:.3f}\nMax: {max_value:.3f}'
                ax3.text(0.02, 0.98, value_stats_text, 
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax3.text(0.5, 0.5, 'No value function data found', 
                        ha='center', va='center', transform=ax3.transAxes, 
                        fontsize=14, color='gray')
                initial_value = 0
                final_value = 0
                max_value = 0
            
            ax3.set_title('Value Function Estimates', fontsize=14, fontweight='bold')
            
            # Overall title
            fig.suptitle(f'Training Analysis: {info["environment"]} - {info["agent_type"]} - {info["algorithm"].upper()}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            output_filename = f"{output_dir}/{info['full_name']}_analysis.png"
            fig.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_filename}")
            
            # Show some statistics
            print(f"  Environment: {info['environment']}")
            print(f"  Agent Type: {info['agent_type']}")
            print(f"  Algorithm: {info['algorithm'].upper()}")
            print(f"  Index: {info['index']}")
            
            if reward_plotted:
                print(f"  Final mean reward: {final_reward:.2f}")
                print(f"  Max mean reward: {max_reward:.2f}")
            
            if epsilon_plotted:
                print(f"  Initial epsilon: {initial_epsilon:.4f}")
                print(f"  Final epsilon: {final_epsilon:.4f}")
            
            if value_plotted:
                print(f"  Initial value function: {initial_value:.3f}")
                print(f"  Final value function: {final_value:.3f}")
                print(f"  Max value function: {max_value:.3f}")
            
            plt.close(fig)  # Close to free memory
            
        except Exception as e:
            print(f"Error processing {log_dir_name}: {str(e)}")
            continue

def create_comparison_plots(logs_base_dir='logs', output_dir='output'):
    """
    Create comparison plots grouping by agent type and algorithm
    
    Args:
        logs_base_dir: Base directory containing TensorBoard logs
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all log directories and group them
    log_dirs = [d for d in os.listdir(logs_base_dir) 
                if os.path.isdir(os.path.join(logs_base_dir, d))]
    
    # Group logs by agent type and algorithm
    groups = {}
    for log_dir_name in log_dirs:
        info = parse_log_directory_name(log_dir_name)
        group_key = f"{info['agent_type']}_{info['algorithm']}"
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append((log_dir_name, info))
    
    # Create comparison plots for each group
    for group_key, group_logs in groups.items():
        if len(group_logs) <= 1:
            continue  # Skip groups with only one log
            
        print(f"\nCreating comparison plot for group: {group_key}")
        
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_logs)))
        
        # ===== LEFT SUBPLOT: Mean Episode Reward + Epsilon Overlay =====
        # Plot mean episode reward on primary y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Mean Episode Reward (100-ep)', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Create secondary y-axis for epsilon
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Exploration Rate (Epsilon)', color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        reward_lines = []
        epsilon_lines = []
        value_lines = []
        
        for i, (log_dir_name, info) in enumerate(group_logs):
            log_path = os.path.join(logs_base_dir, log_dir_name)
            
            try:
                data = load_tensorboard_data(log_path)
                color = colors[i]
                
                # Plot mean episode reward
                if 'rollout/ep_rew_mean' in data:
                    reward_data = data['rollout/ep_rew_mean']
                    line1 = ax1.plot(reward_data['steps'], reward_data['values'], 
                            color=color, linewidth=2, alpha=0.8,
                            label=f"Reward Run {info['index']}")
                    reward_lines.extend(line1)
                
                # Plot epsilon decay (with slightly different style)
                if 'rollout/exploration_rate' in data:
                    epsilon_data = data['rollout/exploration_rate']
                    line2 = ax2.plot(epsilon_data['steps'], epsilon_data['values'], 
                            color=color, linewidth=2, alpha=0.6, linestyle='--',
                            label=f"Epsilon Run {info['index']}")
                    epsilon_lines.extend(line2)
                
                # Plot value function estimates
                if 'evaluation/avg_value_function' in data:
                    value_data = data['evaluation/avg_value_function']
                    line3 = ax3.plot(value_data['steps'], value_data['values'], 
                            color=color, linewidth=2, alpha=0.8, marker='o', markersize=3,
                            label=f"Value Run {info['index']}")
                    value_lines.extend(line3)
                
            except Exception as e:
                print(f"Error loading {log_dir_name}: {str(e)}")
                continue
        
        # Configure left subplot (reward + epsilon)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Reward & Exploration Rate Comparison', fontsize=14, fontweight='bold')
        
        # Create combined legend for left subplot
        all_lines_left = reward_lines + epsilon_lines
        all_labels_left = [line.get_label() for line in all_lines_left]
        ax1.legend(all_lines_left, all_labels_left, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        # Configure right subplot (value function)
        ax3.set_xlabel('Training Steps', fontsize=12)
        ax3.set_ylabel('Average Value Function', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Value Function Comparison', fontsize=14, fontweight='bold')
        
        if value_lines:
            ax3.legend(value_lines, [line.get_label() for line in value_lines], loc='lower right')
        else:
            ax3.text(0.5, 0.5, 'No value function data found', 
                    ha='center', va='center', transform=ax3.transAxes, 
                    fontsize=14, color='gray')
        
        # Overall title
        fig.suptitle(f'Training Comparison: {group_key.replace("_", " ").title()} (Reward + Epsilon + Value)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_filename = f"{output_dir}/{group_key}_comparison.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {output_filename}")
        plt.close(fig)

def main():
    """
    Main function to create all visualizations
    """
    print("=== TensorBoard Log Visualization ===")
    print("Creating individual plots for each training run...")
    
    # Create individual plots
    create_individual_plots()
    
    print("\nCreating comparison plots by agent type and algorithm...")
    
    # Create comparison plots
    create_comparison_plots()
    
    print("\n=== Visualization Complete ===")
    print("Check the 'output' directory for generated plots.")

if __name__ == "__main__":
    main()
