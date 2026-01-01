import argparse
import numpy as np
import scipy.spatial.transform as st
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.visualization.render_mp4 import unpickle_data
from reverse.poc_naive_reverse.reverse_dataset import extract_ee_pose_from_obs


def compute_action_magnitude(action):
    """
    Compute the magnitude of a delta action.
    
    Args:
        action: numpy array of shape (8,) with [dx, dy, dz, dqx, dqy, dqz, dqw, gripper]
        
    Returns:
        dict with magnitudes for position, rotation, and total
    """
    # Position delta magnitude
    pos_delta = action[:3]
    pos_magnitude = np.linalg.norm(pos_delta)
    
    # Rotation delta magnitude (quaternion)
    rot_delta_quat = action[3:7]
    # Convert quaternion delta to rotation angle
    rot_delta = st.Rotation.from_quat(rot_delta_quat)
    rot_angle = rot_delta.magnitude()  # Returns rotation angle in radians
    
    # Gripper change magnitude
    gripper_delta = abs(action[7])
    
    # Total magnitude (weighted combination)
    # Position in meters, rotation in radians, gripper normalized
    total_magnitude = np.sqrt(pos_magnitude**2 + rot_angle**2 + gripper_delta**2)
    
    return {
        "position": pos_magnitude,
        "rotation": rot_angle,
        "gripper": gripper_delta,
        "total": total_magnitude
    }


def compute_state_difference(obs_t, obs_t_next):
    """
    Compute the actual state difference between two observations.
    
    Args:
        obs_t: Observation at timestep t
        obs_t_next: Observation at timestep t+1
        
    Returns:
        dict with magnitudes for position, rotation, and total differences
    """
    # Extract end-effector poses
    ee_pos_t, ee_quat_t, gripper_width_t = extract_ee_pose_from_obs(obs_t)
    ee_pos_next, ee_quat_next, gripper_width_next = extract_ee_pose_from_obs(obs_t_next)
    
    # Position difference
    pos_diff = ee_pos_next - ee_pos_t
    pos_magnitude = np.linalg.norm(pos_diff)
    
    # Rotation difference
    rot_t = st.Rotation.from_quat(ee_quat_t)
    rot_next = st.Rotation.from_quat(ee_quat_next)
    rot_diff = rot_t.inv() * rot_next  # Relative rotation
    rot_angle = rot_diff.magnitude()  # Rotation angle in radians
    
    # Gripper difference
    gripper_diff = abs(gripper_width_next - gripper_width_t)
    
    # Total magnitude
    total_magnitude = np.sqrt(pos_magnitude**2 + rot_angle**2 + gripper_diff**2)
    
    return {
        "position": pos_magnitude,
        "rotation": rot_angle,
        "gripper": gripper_diff,
        "total": total_magnitude
    }


def analyze_trajectory(data, label="trajectory"):
    """
    Analyze a trajectory and compute action magnitudes and state differences.
    
    Args:
        data: Dictionary with trajectory data
        label: Label for this trajectory (for plotting)
        
    Returns:
        dict with action magnitudes and state differences
    """
    observations = data["observations"]
    actions = data.get("actions", [])
    
    n_obs = len(observations)
    n_actions = len(actions)
    
    if n_obs < 2:
        raise ValueError(f"Trajectory must have at least 2 observations, got {n_obs}")
    
    # Compute action magnitudes
    action_magnitudes = {
        "position": [],
        "rotation": [],
        "gripper": [],
        "total": []
    }
    
    for action in actions:
        if action is None:
            continue
        action = np.array(action)
        if action.ndim > 1:
            action = action.flatten()
        if len(action) < 8:
            print(f"Warning: Action has unexpected shape {action.shape}, skipping")
            continue
        
        mags = compute_action_magnitude(action)
        for key in action_magnitudes:
            action_magnitudes[key].append(mags[key])
    
    # Compute state differences
    state_differences = {
        "position": [],
        "rotation": [],
        "gripper": [],
        "total": []
    }
    
    for t in range(n_obs - 1):
        obs_t = observations[t]
        obs_t_next = observations[t + 1]
        
        diffs = compute_state_difference(obs_t, obs_t_next)
        for key in state_differences:
            state_differences[key].append(diffs[key])
    
    # Convert to numpy arrays
    for key in action_magnitudes:
        action_magnitudes[key] = np.array(action_magnitudes[key])
    for key in state_differences:
        state_differences[key] = np.array(state_differences[key])
    
    return {
        "action_magnitudes": action_magnitudes,
        "state_differences": state_differences,
        "label": label
    }


def visualize_comparison(results_list, output_path):
    """
    Visualize comparison between action magnitudes and state differences.
    
    Args:
        results_list: List of analysis results (from analyze_trajectory)
        output_path: Path to save the visualization
    """
    n_trajectories = len(results_list)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define components to plot
    components = ["position", "rotation", "gripper", "total"]
    component_titles = {
        "position": "Position (m)",
        "rotation": "Rotation (rad)",
        "gripper": "Gripper Width",
        "total": "Total Magnitude"
    }
    
    for comp_idx, component in enumerate(components):
        # Action magnitude subplot
        ax1 = plt.subplot(4, 2, 2 * comp_idx + 1)
        for result in results_list:
            action_mags = result["action_magnitudes"][component]
            timesteps = np.arange(len(action_mags))
            ax1.plot(timesteps, action_mags, 'o-', label=f"{result['label']} (action)", alpha=0.7, markersize=3)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel(f'Action Magnitude ({component_titles[component]})')
        ax1.set_title(f'Action Magnitude: {component_titles[component]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # State difference subplot
        ax2 = plt.subplot(4, 2, 2 * comp_idx + 2)
        for result in results_list:
            state_diffs = result["state_differences"][component]
            timesteps = np.arange(len(state_diffs))
            ax2.plot(timesteps, state_diffs, 's-', label=f"{result['label']} (state)", alpha=0.7, markersize=3)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel(f'State Difference ({component_titles[component]})')
        ax2.set_title(f'State Difference: {component_titles[component]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def visualize_scatter_comparison(results_list, output_path):
    """
    Create scatter plots comparing action magnitudes vs state differences.
    
    Args:
        results_list: List of analysis results
        output_path: Path to save the visualization
    """
    components = ["position", "rotation", "gripper", "total"]
    component_titles = {
        "position": "Position (m)",
        "rotation": "Rotation (rad)",
        "gripper": "Gripper Width",
        "total": "Total Magnitude"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for comp_idx, component in enumerate(components):
        ax = axes[comp_idx]
        
        for result in results_list:
            action_mags = result["action_magnitudes"][component]
            state_diffs = result["state_differences"][component]
            
            # Ensure same length
            min_len = min(len(action_mags), len(state_diffs))
            action_mags = action_mags[:min_len]
            state_diffs = state_diffs[:min_len]
            
            ax.scatter(action_mags, state_diffs, label=result["label"], alpha=0.5, s=20)
        
        # Add diagonal line (y=x) for reference
        max_val = max(
            max(result["action_magnitudes"][component]) if len(result["action_magnitudes"][component]) > 0 else 0
            for result in results_list
        )
        max_val = max(max_val, max(
            max(result["state_differences"][component]) if len(result["state_differences"][component]) > 0 else 0
            for result in results_list
        ))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (perfect match)', alpha=0.7)
        
        ax.set_xlabel(f'Action Magnitude ({component_titles[component]})')
        ax.set_ylabel(f'State Difference ({component_titles[component]})')
        ax.set_title(f'Action vs State: {component_titles[component]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter comparison to {output_path}")


def visualize_statistics(results_list, output_path):
    """
    Create statistics comparison plots.
    
    Args:
        results_list: List of analysis results
        output_path: Path to save the visualization
    """
    components = ["position", "rotation", "gripper", "total"]
    component_titles = {
        "position": "Position (m)",
        "rotation": "Rotation (rad)",
        "gripper": "Gripper Width",
        "total": "Total Magnitude"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for comp_idx, component in enumerate(components):
        ax = axes[comp_idx]
        
        labels = []
        action_means = []
        action_stds = []
        state_means = []
        state_stds = []
        
        for result in results_list:
            action_mags = result["action_magnitudes"][component]
            state_diffs = result["state_differences"][component]
            
            if len(action_mags) > 0 and len(state_diffs) > 0:
                labels.append(result["label"])
                action_means.append(np.mean(action_mags))
                action_stds.append(np.std(action_mags))
                state_means.append(np.mean(state_diffs))
                state_stds.append(np.std(state_diffs))
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, action_means, width, yerr=action_stds, label='Action Magnitude', alpha=0.7)
        ax.bar(x + width/2, state_means, width, yerr=state_stds, label='State Difference', alpha=0.7)
        
        ax.set_xlabel('Trajectory')
        ax.set_ylabel(f'Mean Magnitude ({component_titles[component]})')
        ax.set_title(f'Statistics: {component_titles[component]}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize action and state comparison")
    parser.add_argument("--trajectory-path", type=str, required=True,
                        help="Path to trajectory pickle file")
    parser.add_argument("--reversed-trajectory-path", type=str, default=None,
                        help="Optional path to reversed trajectory pickle file for comparison")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualizations (default: same as trajectory directory)")
    
    args = parser.parse_args()
    
    trajectory_path = Path(args.trajectory_path)
    if not trajectory_path.exists():
        raise ValueError(f"Trajectory file does not exist: {trajectory_path}")
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = trajectory_path.parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory
    print(f"Loading trajectory from {trajectory_path}")
    data = unpickle_data(trajectory_path)
    
    # Analyze trajectory
    print("Analyzing trajectory...")
    results = [analyze_trajectory(data, label="original")]
    
    # Load and analyze reversed trajectory if provided
    if args.reversed_trajectory_path:
        reversed_path = Path(args.reversed_trajectory_path)
        if reversed_path.exists():
            print(f"Loading reversed trajectory from {reversed_path}")
            reversed_data = unpickle_data(reversed_path)
            reversed_results = analyze_trajectory(reversed_data, label="reversed")
            results.append(reversed_results)
        else:
            print(f"Warning: Reversed trajectory file does not exist: {reversed_path}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Time series comparison
    comparison_path = output_dir / "action_state_comparison.png"
    visualize_comparison(results, comparison_path)
    
    # Scatter plot comparison
    scatter_path = output_dir / "action_state_scatter.png"
    visualize_scatter_comparison(results, scatter_path)
    
    # Statistics comparison
    stats_path = output_dir / "action_state_statistics.png"
    visualize_statistics(results, stats_path)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    for result in results:
        print(f"\n{result['label'].upper()}:")
        for component in ["position", "rotation", "gripper", "total"]:
            action_mags = result["action_magnitudes"][component]
            state_diffs = result["state_differences"][component]
            
            if len(action_mags) > 0 and len(state_diffs) > 0:
                action_mean = np.mean(action_mags)
                action_std = np.std(action_mags)
                state_mean = np.mean(state_diffs)
                state_std = np.std(state_diffs)
                
                # Compute correlation
                min_len = min(len(action_mags), len(state_diffs))
                if min_len > 1:
                    correlation = np.corrcoef(action_mags[:min_len], state_diffs[:min_len])[0, 1]
                else:
                    correlation = np.nan
                
                print(f"  {component}:")
                print(f"    Action: mean={action_mean:.6f}, std={action_std:.6f}")
                print(f"    State:  mean={state_mean:.6f}, std={state_std:.6f}")
                print(f"    Correlation: {correlation:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

