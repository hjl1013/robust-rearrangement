import argparse
import pickle
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import cv2

from src.behavior import get_actor
from src.behavior.base import Actor
from src.visualization.render_mp4 import unpickle_data
from src.common.geometry import np_proprioceptive_quat_xyzw_to_rot_6d

import torch


def get_cache_key(demo_dir: Path, max_demo_samples: Optional[int], observation_type: str) -> str:
    """Generate a cache key based on demo directory and parameters."""
    cache_str = f"{demo_dir}_{max_demo_samples}_{observation_type}"
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
    return cache_hash


def get_cache_path(cache_key: str) -> Path:
    """Get the cache file path for a given cache key."""
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"demo_latents_{cache_key}.npz"


def load_demo_latents_cache(cache_path: Path) -> Optional[np.ndarray]:
    """Load demo latents from cache if it exists."""
    if cache_path.exists():
        try:
            print(f"Loading demo latents from cache: {cache_path}")
            data = np.load(cache_path)
            demo_latents = data["demo_latents"]
            print(f"Loaded cached demo latents with shape: {demo_latents.shape}")
            return demo_latents
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
    return None


def save_demo_latents_cache(cache_path: Path, demo_latents: np.ndarray):
    """Save demo latents to cache."""
    try:
        print(f"Saving demo latents to cache: {cache_path}")
        np.savez_compressed(cache_path, demo_latents=demo_latents)
        print(f"Cache saved successfully")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def load_demo_trajectories(demo_dir: Path, max_samples: Optional[int] = None):
    """Load demonstration trajectories from pickle files."""
    pickle_files = list(demo_dir.rglob("*.pkl")) + list(demo_dir.rglob("*.pkl.xz")) + list(demo_dir.rglob("*.pkl.gz"))
    
    if len(pickle_files) == 0:
        raise ValueError(f"No pickle files found in {demo_dir}")
    
    print(f"Found {len(pickle_files)} pickle files")
    
    all_observations = []
    for pickle_path in tqdm(pickle_files, desc="Loading demo trajectories"):
        if max_samples is not None and len(all_observations) >= max_samples:
            break
        try:
            data = unpickle_data(pickle_path)
            if "observations" in data:
                all_observations.extend(data["observations"])
                if max_samples is not None and len(all_observations) >= max_samples:
                    all_observations = all_observations[:max_samples]
                    break
            else:
                print(f"Warning: {pickle_path} does not have 'observations' key, skipping")
        except Exception as e:
            print(f"Warning: Failed to load {pickle_path}: {e}, skipping")
            continue
    
    print(f"Loaded {len(all_observations)} demo observations")
    return all_observations


def filter_and_concat_robot_state_np(robot_state):
    """Filter and concatenate robot state, ensuring all arrays are 1D."""
    from src.common.robot_state import ROBOT_STATES
    
    current_robot_state = []
    for rs in ROBOT_STATES:
        if rs not in robot_state:
            continue
        
        value = robot_state[rs]
        # Ensure value is numpy array
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        # Squeeze to remove batch dimensions, then ensure 1D
        value = np.squeeze(value)
        if value.ndim == 0:
            value = value.reshape(1)
        elif value.ndim > 1:
            value = value.flatten()
        
        current_robot_state.append(value)
    
    return np.concatenate(current_robot_state, axis=-1)


def extract_encoder_features(
    actor: Actor,
    observations: List[dict],
    device: torch.device,
    observation_type: str,
) -> np.ndarray:
    """Extract latent features from observations using the encoder."""
    actor.eval()
    latents = []
    
    with torch.no_grad():
        for obs in tqdm(observations, desc="Extracting latents"):
            if observation_type == "state":
                robot_state = obs.get("robot_state", {})
                parts_poses = obs.get("parts_poses", np.zeros(0))
                
                if isinstance(robot_state, dict):
                    robot_state = filter_and_concat_robot_state_np(robot_state)
                
                robot_state = np.array(robot_state, dtype=np.float32)
                if robot_state.ndim > 1:
                    robot_state = robot_state.flatten()
                
                if robot_state.shape[-1] == 14:
                    robot_state = np_proprioceptive_quat_xyzw_to_rot_6d(robot_state)
                elif robot_state.shape[-1] != 16:
                    raise ValueError(
                        f"Unexpected robot_state dimension: {robot_state.shape[-1]}. "
                        f"Expected 14 (quaternion) or 16 (rot_6d)."
                    )
                
                robot_state = torch.from_numpy(robot_state).float().to(device)
                if robot_state.ndim == 1:
                    robot_state = robot_state.unsqueeze(0)
                
                parts_poses = np.array(parts_poses, dtype=np.float32)
                parts_poses = torch.from_numpy(parts_poses).float().to(device)
                if parts_poses.ndim == 1:
                    parts_poses = parts_poses.unsqueeze(0)
                
                robot_state[..., :3] *= int(actor.include_proprioceptive_pos)
                robot_state[..., 3:9] *= int(actor.include_proprioceptive_ori)
                
                robot_state_norm = actor.normalizer(robot_state, "robot_state", forward=True)
                parts_poses_norm = actor.normalizer(parts_poses, "parts_poses", forward=True)
                
                latent = torch.cat([robot_state_norm, parts_poses_norm], dim=-1)
            else:
                raise ValueError(f"Only 'state' observation type is supported for now")
            
            latent_np = latent.cpu().numpy()
            if latent_np.ndim > 1:
                latent_np = latent_np.flatten()
            latents.append(latent_np)
    
    return np.array(latents)


def calculate_min_distances(rollout_latents: np.ndarray, demo_latents: np.ndarray) -> np.ndarray:
    """Calculate minimum distance from each rollout latent to demo latents."""
    distances = []
    
    if rollout_latents.ndim == 1:
        rollout_latents = rollout_latents.reshape(1, -1)
    elif rollout_latents.ndim > 2:
        rollout_latents = rollout_latents.reshape(rollout_latents.shape[0], -1)
    
    if demo_latents.ndim == 1:
        demo_latents = demo_latents.reshape(1, -1)
    elif demo_latents.ndim > 2:
        demo_latents = demo_latents.reshape(demo_latents.shape[0], -1)
    
    if rollout_latents.shape[1] != demo_latents.shape[1]:
        min_dim = min(rollout_latents.shape[1], demo_latents.shape[1])
        print(f"Warning: Dimension mismatch. Truncating both to {min_dim} dimensions...")
        rollout_latents = rollout_latents[:, :min_dim]
        demo_latents = demo_latents[:, :min_dim]
    
    for rollout_latent in rollout_latents:
        rollout_latent = rollout_latent.flatten()
        dists = np.sqrt(np.sum((demo_latents - rollout_latent) ** 2, axis=1))
        min_dist = np.min(dists)
        distances.append(min_dist)
    
    return np.array(distances)


def rollout(env, policy, demo_dir: Path, max_demo_samples: int, observation_type: str, device: torch.device, n_rollouts: int):
    """Rollout with environment and policy using parallel environments, saving observations and out-of-distribution scores.
    
    Args:
        env: Environment with num_envs parallel environments
        policy: Policy to use for rollout
        demo_dir: Directory containing demonstration trajectories
        max_demo_samples: Maximum number of demo samples to use
        observation_type: Type of observation ("state" or "image")
        device: Device to use
        n_rollouts: Total number of rollouts to perform
    
    Returns:
        batched_observations: List of lists, where each inner list contains observations for one rollout
        batched_actions: List of lists, where each inner list contains actions for one rollout
        batched_rewards: List of lists, where each inner list contains rewards for one rollout
        batched_out_of_distribution_scores: List of arrays, where each array contains OOD scores for one rollout
    """
    from copy import deepcopy
    
    # Load demo latents (with caching)
    cache_key = get_cache_key(demo_dir, max_demo_samples, observation_type)
    cache_path = get_cache_path(cache_key)
    
    demo_latents = load_demo_latents_cache(cache_path)
    if demo_latents is None:
        print("Loading demo trajectories...")
        demo_observations = load_demo_trajectories(demo_dir, max_samples=max_demo_samples)
        
        print("Extracting demo latents...")
        demo_latents = extract_encoder_features(policy, demo_observations, device, observation_type)
        
        save_demo_latents_cache(cache_path, demo_latents)
    
    num_envs = env.num_envs
    n_batches = (n_rollouts + num_envs - 1) // num_envs  # Ceiling division
    
    # Storage for batched results
    batched_observations = []
    batched_actions = []
    batched_rewards = []
    batched_out_of_distribution_scores = []
    
    print(f"Running {n_rollouts} rollouts using {num_envs} parallel environments ({n_batches} batches)...")
    
    # Helper function to extract observation for a specific environment
    def extract_env_obs(obs_batch, env_idx):
        obs_copy = {}
        for k, v in obs_batch.items():
            if isinstance(v, dict):
                obs_copy[k] = {k2: np.squeeze(v2[env_idx].cpu().numpy()) if isinstance(v2, torch.Tensor) else np.squeeze(v2[env_idx])
                               for k2, v2 in v.items()}
            elif isinstance(v, torch.Tensor):
                obs_copy[k] = np.squeeze(v[env_idx].cpu().numpy())
            else:
                obs_copy[k] = v[env_idx] if hasattr(v, '__getitem__') else v
        return obs_copy
    
    rollout_idx = 0
    for batch_idx in range(n_batches):
        # Reset environments
        obs = env.reset()
        policy.reset()
        
        # Determine how many environments to use in this batch
        envs_in_batch = min(num_envs, n_rollouts - rollout_idx)
        
        # Storage for this batch (one list per environment)
        batch_observations = [[] for _ in range(envs_in_batch)]
        batch_actions = [[] for _ in range(envs_in_batch)]
        batch_rewards = [[] for _ in range(envs_in_batch)]
        
        # Store initial observations
        for env_idx in range(envs_in_batch):
            obs_copy = extract_env_obs(obs, env_idx)
            batch_observations[env_idx].append(obs_copy)
        
        # Process initial observation for policy (convert robot_state dict to tensor)
        if isinstance(obs["robot_state"], dict):
            obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"])
        
        done = torch.zeros((num_envs, 1), dtype=torch.bool, device=device)
        max_steps = 700  # Default max steps
        
        for step_idx in tqdm(range(max_steps), desc=f"Batch {batch_idx+1}/{n_batches}", leave=False):
            # Convert robot state dict to tensor if needed
            if isinstance(obs["robot_state"], dict):
                obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"])
            
            # Get action from policy
            action = policy.action(obs)
            
            # Store actions for this batch
            for env_idx in range(envs_in_batch):
                action_copy = action[env_idx].cpu().numpy() if isinstance(action, torch.Tensor) else action[env_idx]
                if action_copy.ndim > 1:
                    action_copy = np.squeeze(action_copy)
                batch_actions[env_idx].append(action_copy)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Store rewards for this batch
            for env_idx in range(envs_in_batch):
                if isinstance(reward, torch.Tensor):
                    reward_copy = reward[env_idx].cpu().item() if reward[env_idx].numel() == 1 else reward[env_idx].cpu().numpy()
                else:
                    reward_copy = reward[env_idx]
                batch_rewards[env_idx].append(reward_copy)
            
            # Store observations for this batch
            for env_idx in range(envs_in_batch):
                obs_copy = extract_env_obs(obs, env_idx)
                batch_observations[env_idx].append(obs_copy)
            
            # Check if all environments are done
            if done.all():
                break
        
        # Extract latents and calculate OOD scores for each rollout in this batch
        print(f"Extracting latents and calculating OOD scores for batch {batch_idx+1}...")
        for env_idx in range(envs_in_batch):
            rollout_latents = extract_encoder_features(policy, batch_observations[env_idx], device, observation_type)
            out_of_distribution_scores = calculate_min_distances(rollout_latents, demo_latents)
            
            batched_observations.append(batch_observations[env_idx])
            batched_actions.append(batch_actions[env_idx])
            batched_rewards.append(batch_rewards[env_idx])
            batched_out_of_distribution_scores.append(out_of_distribution_scores)
            
            rollout_idx += 1
            if rollout_idx >= n_rollouts:
                break
        
        if rollout_idx >= n_rollouts:
            break
    
    print(f"Completed {rollout_idx} rollouts")
    return batched_observations, batched_actions, batched_rewards, batched_out_of_distribution_scores


def calculate_first_out_of_distribution_state(
    observations: List[dict],
    out_of_distribution_scores: np.ndarray,
    threshold: float,
    consecutive_steps: int,
    timestep_diff: int,
) -> Tuple[int, int]:
    """Calculate the first state that is out of distribution.
    
    Args:
        observations: List of observation dictionaries
        out_of_distribution_scores: Array of OOD scores
        threshold: Threshold for OOD detection
        consecutive_steps: Number of consecutive steps above threshold required
        timestep_diff: Timestep difference between in-distribution and out-of-distribution states
    
    Returns:
        in_distribution_index: Index of in-distribution state (before first OOD state)
        out_of_distribution_index: Index of first out-of-distribution state
    """
    # Find consecutive steps above threshold
    above_threshold = out_of_distribution_scores > threshold
    
    # Find sequences of consecutive steps above threshold
    out_of_distribution_index = None
    consecutive_count = 0
    
    for i in range(len(out_of_distribution_scores)):
        if above_threshold[i]:
            consecutive_count += 1
            if consecutive_count >= consecutive_steps:
                out_of_distribution_index = i - consecutive_steps + 1  # Start of consecutive sequence
                break
        else:
            consecutive_count = 0
    
    if out_of_distribution_index is None:
        # No out-of-distribution state found
        return len(observations) - 1, len(observations) - 1
    
    # Calculate in-distribution index (before the first OOD state)
    in_distribution_index = max(0, out_of_distribution_index - timestep_diff)
    
    return in_distribution_index, out_of_distribution_index


def calculate_first_out_of_distribution_state_batched(
    batched_observations: List[List[dict]],
    batched_out_of_distribution_scores: List[np.ndarray],
    threshold: float,
    consecutive_steps: int,
    timestep_diff: int,
) -> List[Tuple[int, int]]:
    """Calculate the first out-of-distribution state for batched rollouts.
    
    Args:
        batched_observations: List of observation lists, one per rollout
        batched_out_of_distribution_scores: List of OOD score arrays, one per rollout
        threshold: Threshold for OOD detection
        consecutive_steps: Number of consecutive steps above threshold required
        timestep_diff: Timestep difference between in-distribution and out-of-distribution states
    
    Returns:
        List of (in_distribution_index, out_of_distribution_index) tuples, one per rollout
    """
    results = []
    for observations, scores in zip(batched_observations, batched_out_of_distribution_scores):
        in_idx, out_idx = calculate_first_out_of_distribution_state(
            observations, scores, threshold, consecutive_steps, timestep_diff
        )
        results.append((in_idx, out_idx))
    return results


def get_image_from_observation(obs: dict) -> np.ndarray:
    """Extract image from observation (combine color_image1 and color_image2)."""
    img1 = None
    img2 = None
    
    if "color_image1" in obs:
        img1 = obs["color_image1"]
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if len(img1.shape) == 4:
            img1 = img1[0]
        if len(img1.shape) == 3 and img1.shape[0] == 3:
            img1 = img1.transpose(1, 2, 0)
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
    
    if "color_image2" in obs:
        img2 = obs["color_image2"]
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        if len(img2.shape) == 4:
            img2 = img2[0]
        if len(img2.shape) == 3 and img2.shape[0] == 3:
            img2 = img2.transpose(1, 2, 0)
        if img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
    
    if img1 is not None and img2 is not None:
        # Resize to same height if needed
        if img1.shape[:2] != img2.shape[:2]:
            h = min(img1.shape[0], img2.shape[0])
            img1 = img1[:h]
            img2 = img2[:h]
        combined = np.concatenate([img1, img2], axis=1)
        return combined
    elif img1 is not None:
        return img1
    elif img2 is not None:
        return img2
    else:
        # Create dummy image
        dummy_img = np.zeros((240, 640, 3), dtype=np.uint8)
        if cv2 is not None:
            cv2.putText(dummy_img, "No Images Available", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return dummy_img


def visualize_rollout(
    observations: List[dict],
    out_of_distribution_scores: np.ndarray,
    in_distribution_index: int,
    out_of_distribution_index: int,
    threshold: float,
    consecutive_steps: int,
    output_path: Path,
    fps: int = 20,
):
    """Create a 2x2 grid visualization.
    
    Top-left: In-distribution state image (static)
    Top-right: Out-of-distribution state image (static)
    Bottom-left: Rollout video (sequence of colored images)
    Bottom-right: Out-of-distribution score graph with threshold and highlighted timesteps
    """
    # Ensure indices are valid
    in_distribution_index = max(0, min(in_distribution_index, len(observations) - 1))
    out_of_distribution_index = max(0, min(out_of_distribution_index, len(observations) - 1))
    
    # Get static images
    in_dist_img = get_image_from_observation(observations[in_distribution_index])
    out_dist_img = get_image_from_observation(observations[out_of_distribution_index])
    
    # Prepare rollout video frames
    rollout_frames = []
    for obs in observations:
        frame = get_image_from_observation(obs)
        rollout_frames.append(frame)
    
    # Create video frames
    frames = []
    for t in tqdm(range(len(rollout_frames)), desc="Creating visualization frames"):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top-left: In-distribution state (static)
        ax1.axis('off')
        ax1.imshow(in_dist_img)
        ax1.set_title(f'In-Distribution State (t={in_distribution_index})', fontsize=12, fontweight='bold')
        
        # Top-right: Out-of-distribution state (static)
        ax2.axis('off')
        ax2.imshow(out_dist_img)
        ax2.set_title(f'Out-of-Distribution State (t={out_of_distribution_index})', fontsize=12, fontweight='bold')
        
        # Bottom-left: Rollout video (current frame)
        ax3.axis('off')
        if t < len(rollout_frames):
            ax3.imshow(rollout_frames[t])
        ax3.set_title(f'Rollout Frame {t}', fontsize=12)
        
        # Bottom-right: OOD score graph
        ax4.set_xlabel('Timestep', fontsize=12)
        ax4.set_ylabel('Out-of-Distribution Score', fontsize=12)
        ax4.set_title('OOD Score Over Time', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Plot scores up to current timestep
        timesteps = np.arange(t + 1)
        current_scores = out_of_distribution_scores[:t + 1]
        
        ax4.plot(timesteps, current_scores, 'b-', linewidth=2, label='OOD Score')
        
        # Draw threshold line
        ax4.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
        
        # Highlight consecutive steps region
        if out_of_distribution_index is not None:
            start_idx = min(len(out_of_distribution_scores), out_of_distribution_index + 1)
            end_idx = max(0, out_of_distribution_index + consecutive_steps)
            if start_idx < len(out_of_distribution_scores):
                ax4.axvspan(start_idx, end_idx, alpha=0.3, color='yellow', label=f'Consecutive Steps ({consecutive_steps})')
        
        # Highlight detected timesteps
        if in_distribution_index < len(out_of_distribution_scores):
            ax4.plot(in_distribution_index, out_of_distribution_scores[in_distribution_index], 
                    'go', markersize=12, label='In-Distribution', zorder=5)
        if out_of_distribution_index < len(out_of_distribution_scores):
            ax4.plot(out_of_distribution_index, out_of_distribution_scores[out_of_distribution_index], 
                    'ro', markersize=12, label='Out-of-Distribution', zorder=5)
        
        ax4.set_xlim(0, len(out_of_distribution_scores))
        ax4.set_ylim(0, np.max(out_of_distribution_scores) * 1.1)
        ax4.legend(loc='upper right', fontsize=10)
        
        # Convert figure to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
    
    # Save video
    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"Saved visualization video to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rollout with policy and detect out-of-distribution states"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to policy checkpoint file (.pt)",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        required=True,
        help="Directory containing demonstration pickle files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reverse/data_collection/data",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--max-demo-samples",
        type=int,
        default=10000,
        help="Maximum number of demo samples to use",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2,
        help="Threshold for out-of-distribution detection",
    )
    parser.add_argument(
        "--consecutive-steps",
        type=int,
        default=20,
        help="Number of consecutive steps above threshold required",
    )
    parser.add_argument(
        "--timestep-diff",
        type=int,
        default=1,
        help="Timestep difference between in-distribution and out-of-distribution states",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization video",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for output video",
    )
    parser.add_argument(
        "--furniture",
        type=str,
        default="one_leg",
        help="Furniture type",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--randomness",
        type=str,
        default="low",
        help="Randomness level",
        choices=["low", "med", "high"],
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=1,
        help="Number of rollouts to perform",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = Path(args.demo_dir)
    
    # Create folders for rollouts and visualizations
    rollouts_dir = output_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    
    visualizations_dir = None
    if args.visualize:
        visualizations_dir = output_dir / "visualizations"
        visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing rollouts to determine starting index
    existing_rollouts = sorted(rollouts_dir.glob("*.pkl"))
    start_idx = 0
    if existing_rollouts:
        # Extract numbers from existing filenames (e.g., "000.pkl" -> 0)
        existing_indices = []
        for f in existing_rollouts:
            try:
                idx = int(f.stem)
                existing_indices.append(idx)
            except ValueError:
                continue
        if existing_indices:
            start_idx = max(existing_indices) + 1
        print(f"Found {len(existing_rollouts)} existing rollouts, starting from index {start_idx}")
    
    # Load policy
    print("Loading policy...")
    checkpoint_path = Path(args.policy_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" in checkpoint:
        cfg = OmegaConf.create(checkpoint["config"])
    else:
        raise ValueError("Config not found in checkpoint")
    
    # Apply config fixes
    if "base_policy" in cfg:
        cfg.action_dim = cfg.base_policy.action_dim
    if "student_policy" in cfg:
        cfg.action_dim = cfg.student_policy.action_dim
    if "critic" in cfg:
        cfg.actor.critic = cfg.critic
        cfg.actor.init_logstd = cfg.init_logstd
        cfg.discount = cfg.base_policy.discount
    
    policy: Actor = get_actor(cfg=cfg, device=device)
    
    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    policy.to(device)
    
    observation_type = cfg.get("observation_type", "state")
    print(f"Observation type: {observation_type}")
    
    # Create environment (using same setup as evaluate_model.py)
    print("Creating environment...")
    from src.gym import get_rl_env
    from src.gym.observation import FULL_OBS
    
    # Determine observation space from config (same as evaluate_model.py)
    observation_space = cfg.get("observation_space", "state")
    if observation_space not in ["image", "state"]:
        observation_space = "state"  # Default to state
    
    # Filter obs_keys for state observations (same as evaluate_model.py)
    obs_keys = FULL_OBS
    # if observation_space == "state":
    #     obs_keys = [key for key in FULL_OBS if "image" not in key]
    
    env = get_rl_env(
        gpu_id=args.gpu,
        task=args.furniture,
        num_envs=args.num_envs,
        randomness=args.randomness if hasattr(args, 'randomness') else "low",
        max_env_steps=700,
        resize_img=True,
        observation_space=observation_space,
        act_rot_repr="rot_6d",
        action_type="pos",  # Use "pos" like evaluate_model.py
        april_tags=False,
        verbose=False,
        headless=args.headless,
        concat_robot_state=False,  # Important: must be False, then manually concatenate like rollout.py
        ctrl_mode="diffik",
        obs_keys=obs_keys,
    )
    
    # Run batched rollouts
    batched_observations, batched_actions, batched_rewards, batched_out_of_distribution_scores = rollout(
        env, policy, demo_dir, args.max_demo_samples, observation_type, device, args.n_rollouts
    )
    
    # Calculate first out-of-distribution states for all rollouts (batched)
    print("\nCalculating out-of-distribution indices for all rollouts...")
    batched_indices = calculate_first_out_of_distribution_state_batched(
        batched_observations,
        batched_out_of_distribution_scores,
        args.threshold,
        args.consecutive_steps,
        args.timestep_diff,
    )
    
    # Save and visualize each rollout
    print(f"\nSaving {len(batched_observations)} rollouts...")
    for rollout_idx, (observations, actions, rewards, ood_scores, (in_idx, out_idx)) in enumerate(
        zip(batched_observations, batched_actions, batched_rewards, 
            batched_out_of_distribution_scores, batched_indices)
    ):
        file_idx = start_idx + rollout_idx
        filename = f"{file_idx:03d}"
        
        print(f"\nRollout {rollout_idx + 1}/{len(batched_observations)} (file: {filename}):")
        print(f"  In-distribution index: {in_idx}")
        print(f"  Out-of-distribution index: {out_idx}")
        print(f"  OOD score at in-dist: {ood_scores[in_idx]:.4f}")
        print(f"  OOD score at out-dist: {ood_scores[out_idx]:.4f}")
        
        # Save rollout results
        results = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "out_of_distribution_scores": ood_scores,
            "in_distribution_index": in_idx,
            "out_of_distribution_index": out_idx,
            "threshold": args.threshold,
            "consecutive_steps": args.consecutive_steps,
            "timestep_diff": args.timestep_diff,
        }

        rollout_path = rollouts_dir / f"{filename}.pkl"
        with open(rollout_path, "wb") as f:
            pickle.dump(results, f)

    # Visualize if requested
    if args.visualize:
        print(f"\nSaving {len(batched_observations)} visualizations...")
        for rollout_idx, (observations, actions, rewards, ood_scores, (in_idx, out_idx)) in enumerate(
            zip(batched_observations, batched_actions, batched_rewards, 
                batched_out_of_distribution_scores, batched_indices)
        ):
            file_idx = start_idx + rollout_idx
            filename = f"{file_idx:03d}"

            video_path = visualizations_dir / f"{filename}.mp4"
            visualize_rollout(
                observations,
                ood_scores,
                in_idx,
                out_idx,
                args.threshold,
                args.consecutive_steps,
                video_path,
                fps=args.fps,
            )
    
    print(f"\n{'='*60}")
    print(f"Done! Completed {len(batched_observations)} rollouts")
    print(f"Rollouts saved to: {rollouts_dir}")
    if args.visualize:
        print(f"Visualizations saved to: {visualizations_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
