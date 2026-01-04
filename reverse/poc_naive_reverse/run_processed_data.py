"""Replay trajectories from processed zarr dataset in FurnitureSim environment."""

import argparse
import zarr
from pathlib import Path
import numpy as np
import gym

from src.common.geometry import np_action_rot_6d_to_quat_xyzw
from src.gym.observation import FULL_OBS

import torch

def load_episode_from_zarr(zarr_path: Path, episode_idx: int):
    """
    Load a single episode from zarr dataset.
    
    Args:
        zarr_path: Path to zarr dataset
        episode_idx: Index of episode to load
        
    Returns:
        dict with episode data
    """
    z = zarr.open(str(zarr_path), mode='r')
    
    # Get episode boundaries
    episode_ends = np.array(z['episode_ends'])
    n_episodes = len(episode_ends)
    
    if episode_idx < 0 or episode_idx >= n_episodes:
        raise ValueError(f"Episode index {episode_idx} out of range [0, {n_episodes})")
    
    # Calculate start and end indices for this episode
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    
    print(f"Loading episode {episode_idx}/{n_episodes}")
    print(f"  Episode length: {end_idx - start_idx} timesteps")
    print(f"  Timestep range: [{start_idx}, {end_idx})")
    
    # Extract episode data
    episode_data = {
        'robot_state': np.array(z['robot_state'][start_idx:end_idx]),
        'action/delta': np.array(z['action/delta'][start_idx:end_idx]),
        'action/pos': np.array(z['action/pos'][start_idx:end_idx]),
        'reward': np.array(z['reward'][start_idx:end_idx]),
        'skill': np.array(z['skill'][start_idx:end_idx]),
        'task': z['task'][episode_idx],
        'success': z['success'][episode_idx],
        'pickle_file': z['pickle_file'][episode_idx] if 'pickle_file' in z else None,
    }
    
    # Load images if they exist
    if 'color_image1' in z:
        episode_data['color_image1'] = np.array(z['color_image1'][start_idx:end_idx])
    if 'color_image2' in z:
        episode_data['color_image2'] = np.array(z['color_image2'][start_idx:end_idx])
    
    # Load parts_poses if they exist
    if 'parts_poses' in z and z['parts_poses'].shape[0] > 0:
        episode_data['parts_poses'] = np.array(z['parts_poses'][start_idx:end_idx])
    
    # Print dataset metadata
    print(f"  Task: {episode_data['task']}")
    print(f"  Success: {episode_data['success']}")
    if episode_data['pickle_file']:
        print(f"  Source file: {episode_data['pickle_file']}")
    
    return episode_data


def convert_action_format(actions, from_format='rot_6d', to_format='quat'):
    """
    Convert action format.
    
    Args:
        actions: numpy array of actions
        from_format: 'rot_6d' or 'quat'
        to_format: 'rot_6d' or 'quat'
        
    Returns:
        Converted actions
    """
    if from_format == to_format:
        return actions
    
    if from_format == 'rot_6d' and to_format == 'quat':
        # Actions are [pos(3), rot_6d(6), gripper(1)] = 10D
        # Convert to [pos(3), quat(4), gripper(1)] = 8D
        return np_action_rot_6d_to_quat_xyzw(actions)
    else:
        raise NotImplementedError(f"Conversion from {from_format} to {to_format} not implemented")


def reconstruct_observations_from_zarr(episode_data):
    """
    Reconstruct observations from zarr episode data.
    
    This creates observation dictionaries similar to what the simulator returns,
    which can be used for resetting the simulator state.
    
    Args:
        episode_data: Dictionary with episode data from zarr
        
    Returns:
        List of observation dictionaries
    """
    n_timesteps = len(episode_data['robot_state'])
    observations = []
    
    print("episode data keys: ", episode_data.keys())
    for t in range(n_timesteps):
        obs = {
            'robot_state': episode_data['robot_state'][t],
        }
        
        # Add images if they exist
        if 'color_image1' in episode_data:
            obs['color_image1'] = episode_data['color_image1'][t]
        if 'color_image2' in episode_data:
            obs['color_image2'] = episode_data['color_image2'][t]
        
        # Add parts_poses if they exist
        if 'parts_poses' in episode_data:
            obs['parts_poses'] = episode_data['parts_poses'][t]
        
        observations.append(obs)
    
    return observations


def action_tensor(ac, device, num_envs=1):
    """Convert action to tensor format for environment."""
    if isinstance(ac, (list, np.ndarray)):
        ac = torch.tensor(ac).float().to(device)
    else:
        ac = ac.clone()
    
    if len(ac.shape) == 1:
        ac = ac[None]
    
    return ac.tile(num_envs, 1).float().to(device)


def main():
    parser = argparse.ArgumentParser(description="Replay trajectories from processed zarr dataset")
    parser.add_argument("--zarr-path", type=str, required=True,
                        help="Path to processed zarr dataset")
    parser.add_argument("--episode-idx", type=int, default=0,
                        help="Index of episode to replay (default: 0)")
    parser.add_argument("--furniture", type=str, default=None,
                        help="Furniture type (default: auto-detect from zarr)")
    parser.add_argument("--action-type", type=str, default="pos",
                        choices=["delta", "pos"],
                        help="Action type (default: pos)")
    parser.add_argument("--act-rot-repr", type=str, default="quat",
                        choices=["quat", "rot_6d"],
                        help="Action rotation representation (default: quat)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--compute-device-id", type=int, default=0,
                        help="GPU device ID for simulation")
    parser.add_argument("--graphics-device-id", type=int, default=0,
                        help="GPU device ID for rendering")
    parser.add_argument("--randomness", type=str, default="low",
                        help="Randomness level")
    parser.add_argument("--reset-to-initial", action="store_true",
                        help="Reset environment to initial state before each step (for debugging)")
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise ValueError(f"Zarr dataset does not exist: {zarr_path}")
    
    # Load episode from zarr
    print("=" * 60)
    print("Loading episode from zarr dataset...")
    print("=" * 60)
    episode_data = load_episode_from_zarr(zarr_path, args.episode_idx)
    
    # Determine furniture type
    if args.furniture is None:
        furniture = episode_data['task']
        print(f"Auto-detected furniture type: {furniture}")
    else:
        furniture = args.furniture
        print(f"Using specified furniture type: {furniture}")
    
    # Select actions based on action type
    if args.action_type == "pos":
        actions = episode_data['action/pos']
        print(f"Using position actions (shape: {actions.shape})")
    else:
        actions = episode_data['action/delta']
        print(f"Using delta actions (shape: {actions.shape})")
    
    # Convert action format if needed
    # Zarr stores actions in rot_6d format (10D for pos, 10D for delta)
    if args.act_rot_repr == "quat":
        print("Converting actions from rot_6d to quat format...")
        actions = convert_action_format(actions, from_format='rot_6d', to_format='quat')
        print(f"Converted actions shape: {actions.shape}")
    
    # Create environment
    print("\n" + "=" * 60)
    print("Creating FurnitureSim environment...")
    print("=" * 60)
    
    env = gym.make(
        "FurnitureSim-v0",
        furniture=furniture,
        num_envs=args.num_envs,
        resize_img=True,
        init_assembled=False,
        record=False,
        headless=args.headless,
        save_camera_input=False,
        randomness=args.randomness,
        action_type=args.action_type,
        obs_keys=FULL_OBS,
        ctrl_mode="diffik",
        act_rot_repr=args.act_rot_repr,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        concat_robot_state=False,
    )
    
    # Initialize environment
    print("\nResetting environment...")
    ob = env.reset()
    
    # Optionally reset to initial state from data
    if args.reset_to_initial:
        print("Resetting environment to initial state from zarr data...")
        observations = reconstruct_observations_from_zarr(episode_data)
        print("observation keys: ", observations[0].keys())
        print("observation robot state: ", observations[0]['robot_state'].shape)
        if hasattr(env, 'reset_to'):
            env.reset_to([observations[0]])
            print("Successfully reset to initial state")
        else:
            print("Warning: Environment does not support reset_to(), starting from default reset state")
    
    # Replay actions
    print("\n" + "=" * 60)
    print(f"Replaying {len(actions)} actions...")
    print("=" * 60)
    
    from tqdm import tqdm
    
    total_reward = 0
    for idx, ac in enumerate(tqdm(actions, desc="Executing actions")):
        # Optionally reset to exact state before each step (for debugging)
        if args.reset_to_initial and hasattr(env, 'reset_to'):
            observations = reconstruct_observations_from_zarr(episode_data)
            env.reset_to([observations[idx]])
        
        # Convert action to tensor
        ac_tensor = action_tensor(ac, env.device, args.num_envs)
        
        # Execute action
        ob, rew, done, info = env.step(ac_tensor)
        
        # Accumulate reward
        if isinstance(rew, torch.Tensor):
            rew_value = rew.item()
        else:
            rew_value = float(rew)
        total_reward += rew_value
        
        # Print progress for significant events
        if rew_value > 0:
            tqdm.write(f"  Step {idx}: Reward = {rew_value:.3f}, Total = {total_reward:.3f}")
        
        if done.any() if isinstance(done, torch.Tensor) else done:
            tqdm.write(f"  Episode finished at step {idx}")
            break
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("Replay completed!")
    print("=" * 60)
    print(f"Total reward: {total_reward:.3f}")
    print(f"Expected success: {episode_data['success']}")
    print(f"Steps executed: {idx + 1}/{len(actions)}")
    
    # Check if environment reports success
    if hasattr(env, 'is_success'):
        success_status = env.is_success()
        print(f"Environment success status: {success_status}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

