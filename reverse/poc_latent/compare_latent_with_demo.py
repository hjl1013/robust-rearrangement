import argparse
import pickle
import lzma
import gzip
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import imageio
try:
    import cv2
except ImportError:
    cv2 = None

from src.behavior import get_actor
from src.behavior.base import Actor
from src.eval.eval_utils import get_model_from_api_or_cached, load_model_weights
from src.visualization.render_mp4 import unpickle_data
from src.data_processing.utils import resize, resize_crop
from src.common.robot_state import filter_and_concat_robot_state as filter_and_concat_robot_state_np
from src.common.geometry import np_proprioceptive_quat_xyzw_to_rot_6d


def load_rollout_trajectory(rollout_path: Path):
    """Load a rollout trajectory from a pickle file."""
    data = unpickle_data(rollout_path)
    return data


def get_cache_key(demo_dir: Path, max_demo_samples: Optional[int], observation_type: str) -> str:
    """Generate a cache key based on demo directory and parameters."""
    # Create a string representation of the cache parameters
    cache_str = f"{demo_dir}_{max_demo_samples}_{observation_type}"
    # Generate hash
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


def load_demo_trajectories(demo_dir: Path, max_episodes: Optional[int] = None):
    """Load demonstration trajectories from pickle files (same format as rollout)."""
    # Find all pickle files in the directory (including compressed ones)
    pickle_files = list(demo_dir.rglob("*.pkl")) + list(demo_dir.rglob("*.pkl.xz")) + list(demo_dir.rglob("*.pkl.gz"))
    
    if len(pickle_files) == 0:
        raise ValueError(f"No pickle files found in {demo_dir}")
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Load all trajectories
    all_observations = []
    loaded_episodes = 0
    
    for pickle_path in tqdm(pickle_files, desc="Loading demo trajectories"):
        if max_episodes is not None and loaded_episodes >= max_episodes:
            break
            
        try:
            data = unpickle_data(pickle_path)
            if "observations" in data:
                all_observations.extend(data["observations"])
                loaded_episodes += 1
            else:
                print(f"Warning: {pickle_path} does not have 'observations' key, skipping")
        except Exception as e:
            print(f"Warning: Failed to load {pickle_path}: {e}, skipping")
            continue
    
    print(f"Loaded {loaded_episodes} demo trajectories with {len(all_observations)} total observations")
    
    return all_observations


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
            # Prepare observation for the encoder
            if observation_type == "image":
                # Check if actor has encoders
                if not (hasattr(actor, "encoder1") and hasattr(actor, "encoder2") 
                        and actor.encoder1 is not None and actor.encoder2 is not None):
                    raise ValueError(
                        "Actor does not have encoders. "
                        "Make sure observation_type='image' and the model was trained with image observations."
                    )
                
                # Process images
                image1 = obs["color_image1"]
                image2 = obs["color_image2"]
                
                # Convert to torch tensor if numpy
                if isinstance(image1, np.ndarray):
                    image1 = torch.from_numpy(image1).to(device)
                    image2 = torch.from_numpy(image2).to(device)
                
                # Resize images if needed (similar to rollout.py)
                # Images should be (H, W, C), convert to (1, C, H, W)
                if len(image1.shape) == 3:
                    image1 = image1.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                    image2 = image2.permute(2, 0, 1).unsqueeze(0)
                elif len(image1.shape) == 4:
                    # Already batched, ensure it's (1, C, H, W)
                    if image1.shape[0] != 1:
                        image1 = image1.unsqueeze(0)
                        image2 = image2.unsqueeze(0)
                
                # Apply transforms (resize to 224x224)
                image1 = actor.camera1_transform(image1)
                image2 = actor.camera2_transform(image2)
                
                # Encode images - check if projection layers exist
                encoder1_out = actor.encoder1(image1)
                encoder2_out = actor.encoder2(image2)
                
                if hasattr(actor, "encoder1_proj") and actor.encoder1_proj is not None:
                    feature1 = actor.encoder1_proj(encoder1_out)
                else:
                    feature1 = encoder1_out
                
                if hasattr(actor, "encoder2_proj") and actor.encoder2_proj is not None:
                    feature2 = actor.encoder2_proj(encoder2_out)
                else:
                    feature2 = encoder2_out
                
                # Apply layernorm if enabled
                if hasattr(actor, "feature_layernorm") and actor.feature_layernorm:
                    if hasattr(actor, "layernorm1") and actor.layernorm1 is not None:
                        feature1 = actor.layernorm1(feature1)
                    if hasattr(actor, "layernorm2") and actor.layernorm2 is not None:
                        feature2 = actor.layernorm2(feature2)
                
                # For image observations, we can use just encoder features or include robot_state
                # Using just encoder features for now (pure encoder output)
                latent = torch.cat([feature1, feature2], dim=-1)  # (1, 2*encoding_dim)
                
            elif observation_type == "state":
                # For state observations, we need robot_state and parts_poses
                robot_state = obs["robot_state"]
                parts_poses = obs["parts_poses"]
                
                # Convert robot_state dict to numpy array if needed
                if isinstance(robot_state, dict):
                    robot_state = filter_and_concat_robot_state_np(robot_state)
                
                robot_state = np.array(robot_state, dtype=np.float32)
                
                # Ensure robot_state is 1D or 2D with last dim being the state dimension
                if robot_state.ndim == 0:
                    robot_state = robot_state.reshape(1)
                elif robot_state.ndim > 1:
                    robot_state = robot_state.flatten()
                
                # Convert from 14D (quaternion) to 16D (rot_6d) if needed
                # The normalizer expects 16D format
                if robot_state.shape[-1] == 14:
                    robot_state = np_proprioceptive_quat_xyzw_to_rot_6d(robot_state)
                elif robot_state.shape[-1] != 16:
                    raise ValueError(
                        f"Unexpected robot_state dimension: {robot_state.shape[-1]}. "
                        f"Expected 14 (quaternion) or 16 (rot_6d)."
                    )
                
                # Convert to torch tensor and ensure correct shape: (1, state_dim)
                robot_state = torch.from_numpy(robot_state).float().to(device)
                if robot_state.ndim == 1:
                    robot_state = robot_state.unsqueeze(0)  # (1, state_dim)
                
                parts_poses = np.array(parts_poses, dtype=np.float32)
                parts_poses = torch.from_numpy(parts_poses).float().to(device)
                if parts_poses.ndim == 1:
                    parts_poses = parts_poses.unsqueeze(0)  # (1, parts_poses_dim)
                
                # Apply proprioceptive flags (same as Actor does)
                robot_state[..., :3] *= int(actor.include_proprioceptive_pos)
                robot_state[..., 3:9] *= int(actor.include_proprioceptive_ori)
                
                # Normalize
                robot_state_norm = actor.normalizer(robot_state, "robot_state", forward=True)
                parts_poses_norm = actor.normalizer(parts_poses, "parts_poses", forward=True)
                
                # Concatenate (this is the full observation encoding, not just encoder output)
                latent = torch.cat([robot_state_norm, parts_poses_norm], dim=-1)
            else:
                raise ValueError(f"Unknown observation type: {observation_type}")
            
            # Ensure latent is properly shaped: flatten to 1D
            latent_np = latent.cpu().numpy()
            if latent_np.ndim > 1:
                latent_np = latent_np.flatten()
            latents.append(latent_np)
    
    return np.array(latents)




def calculate_min_distances(rollout_latents: np.ndarray, demo_latents: np.ndarray) -> tuple:
    """Calculate minimum distance from each rollout latent to demo latents."""
    distances = []
    
    # Ensure both arrays are 2D: (n_samples, latent_dim)
    if rollout_latents.ndim == 1:
        rollout_latents = rollout_latents.reshape(1, -1)
    elif rollout_latents.ndim > 2:
        rollout_latents = rollout_latents.reshape(rollout_latents.shape[0], -1)
    
    if demo_latents.ndim == 1:
        demo_latents = demo_latents.reshape(1, -1)
    elif demo_latents.ndim > 2:
        demo_latents = demo_latents.reshape(demo_latents.shape[0], -1)
    
    # Check that latent dimensions match
    if rollout_latents.shape[1] != demo_latents.shape[1]:
        print(f"\nWARNING: Latent dimension mismatch detected!")
        print(f"  Rollout latents: {rollout_latents.shape} ({rollout_latents.shape[1]} dims)")
        print(f"  Demo latents: {demo_latents.shape} ({demo_latents.shape[1]} dims)")
        
        # Try to fix: if rollout has more dims and starts with robot_state, use only robot_state
        if rollout_latents.shape[1] > demo_latents.shape[1] and demo_latents.shape[1] == 16:
            print(f"\nAttempting to fix: Using only robot_state (first {demo_latents.shape[1]} dims) from rollout...")
            rollout_latents = rollout_latents[:, :demo_latents.shape[1]]
            print(f"  Adjusted rollout latents shape: {rollout_latents.shape}")
        elif demo_latents.shape[1] > rollout_latents.shape[1] and rollout_latents.shape[1] == 16:
            print(f"\nAttempting to fix: Using only robot_state (first {rollout_latents.shape[1]} dims) from demo...")
            demo_latents = demo_latents[:, :rollout_latents.shape[1]]
            print(f"  Adjusted demo latents shape: {demo_latents.shape}")
        else:
            raise ValueError(
                f"Latent dimension mismatch: rollout latents have {rollout_latents.shape[1]} dims, "
                f"but demo latents have {demo_latents.shape[1]} dims. "
                f"Rollout shape: {rollout_latents.shape}, Demo shape: {demo_latents.shape}\n"
                f"This usually means:\n"
                f"  - Rollout has parts_poses but demo trajectories don't (or vice versa)\n"
                f"  - Different observation types are being used\n"
                f"  - Trajectory structure mismatch\n"
                f"Ensure both rollout and demo pickle files have the same observation structure."
            )
    
    # Use vectorized computation for efficiency
    for rollout_latent in rollout_latents:
        # Ensure rollout_latent is 1D: (latent_dim,)
        rollout_latent = rollout_latent.flatten()
        
        # Calculate L2 distance to all demo latents (vectorized)
        # demo_latents: (n_demo, latent_dim), rollout_latent: (latent_dim,)
        dists = np.sqrt(np.sum((demo_latents - rollout_latent) ** 2, axis=1))
        min_dist = np.min(dists)
        distances.append(min_dist)
    
    return np.array(distances), rollout_latents, demo_latents


def create_distance_video(
    rollout_video: np.ndarray,
    distances: np.ndarray,
    output_path: Path,
    fps: int = 20,
):
    """Create a video with rollout on left and distance graph on right."""
    frames = []
    
    for t in tqdm(range(len(rollout_video)), desc="Creating video frames"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left side: video frame
        ax1.axis('off')
        if t < len(rollout_video) and rollout_video[t].size > 0:
            try:
                ax1.imshow(rollout_video[t])
            except Exception as e:
                print(f"Warning: Could not display frame {t}: {e}")
                blank = np.zeros((240, 640, 3), dtype=np.uint8)
                ax1.imshow(blank)
        else:
            blank = np.zeros((240, 640, 3), dtype=np.uint8)
            ax1.imshow(blank)
        ax1.set_title(f'Rollout Frame {t}', fontsize=12)
        
        # Right side: distance graph
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Min Distance to Demo Latents', fontsize=12)
        ax2.set_title('Latent Space Distance Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Plot distance up to current timestep
        timesteps = np.arange(t + 1)
        current_distances = distances[:t + 1]
        
        ax2.plot(timesteps, current_distances, 'b-', linewidth=2, label='Distance')
        ax2.plot(t, distances[t], 'ro', markersize=10, label='Current')
        ax2.set_xlim(0, len(distances))
        ax2.set_ylim(0, np.max(distances) * 1.1)
        ax2.legend()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
    
    # Save video
    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"Saved distance video to {output_path}")


def visualize_latent_space_2d(
    rollout_latents: np.ndarray,
    demo_latents: np.ndarray,
    output_path: Path,
    method: str = "tsne",  # "tsne" or "pca"
):
    """Visualize latent space in 2D with rollout and demo trajectories."""
    # Ensure dimensions match
    if rollout_latents.shape[1] != demo_latents.shape[1]:
        print(f"Warning: Dimension mismatch in 2D visualization. Rollout: {rollout_latents.shape}, Demo: {demo_latents.shape}")
        min_dim = min(rollout_latents.shape[1], demo_latents.shape[1])
        print(f"Truncating both to {min_dim} dimensions...")
        rollout_latents = rollout_latents[:, :min_dim]
        demo_latents = demo_latents[:, :min_dim]
    
    # Combine all latents for dimensionality reduction
    all_latents = np.vstack([demo_latents, rollout_latents])
    
    print(f"Reducing dimensionality using {method}...")
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    latents_2d = reducer.fit_transform(all_latents)
    
    # Split back
    demo_latents_2d = latents_2d[:len(demo_latents)]
    rollout_latents_2d = latents_2d[len(demo_latents):]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot demo trajectories (red, opacity by timestep)
    # We need to group demo latents by episode
    # For simplicity, plot all demo points with varying opacity
    n_demo = len(demo_latents_2d)
    demo_opacities = np.linspace(0.3, 1.0, n_demo)
    
    ax.scatter(
        demo_latents_2d[:, 0],
        demo_latents_2d[:, 1],
        c='red',
        alpha=demo_opacities,
        s=20,
        label='Demonstrations',
        edgecolors='none',
    )
    
    # Plot rollout trajectory (blue, opacity by timestep)
    n_rollout = len(rollout_latents_2d)
    rollout_opacities = np.linspace(0.3, 1.0, n_rollout)
    
    ax.scatter(
        rollout_latents_2d[:, 0],
        rollout_latents_2d[:, 1],
        c='blue',
        alpha=rollout_opacities,
        s=30,
        label='Rollout',
        edgecolors='none',
    )
    
    # Draw lines connecting rollout points
    ax.plot(
        rollout_latents_2d[:, 0],
        rollout_latents_2d[:, 1],
        'b-',
        alpha=0.3,
        linewidth=1,
    )
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title('Latent Space Visualization', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 2D latent visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize latent space distances between rollout and demonstrations"
    )
    parser.add_argument(
        "--wt-path",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pt)",
    )
    parser.add_argument(
        "--rollout-path",
        type=str,
        required=True,
        help="Path to rollout pickle file",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        required=False,
        help="Directory containing demonstration pickle files (.pkl, .pkl.xz, .pkl.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./latent_visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--max-demo-episodes",
        type=int,
        default=None,
        help="Maximum number of demo episodes to load",
    )
    parser.add_argument(
        "--max-demo-samples",
        type=int,
        default=10000,
        help="Maximum number of demo samples to use for distance calculation",
    )
    parser.add_argument(
        "--reduction-method",
        type=str,
        choices=["tsne", "pca"],
        default="tsne",
        help="Method for 2D dimensionality reduction",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for output video",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load model and extract encoder
    print("=" * 60)
    print("Step 1: Loading model...")
    print("=" * 60)
    
    checkpoint_path = Path(args.wt_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    if "config" in checkpoint:
        cfg = OmegaConf.create(checkpoint["config"])
    else:
        raise ValueError("Config not found in checkpoint. Please provide a checkpoint with config.")

    # Temporary fix for residual missing field
    if "base_policy" in cfg:
        print("Applying residual field hotfix")
        cfg.action_dim = cfg.base_policy.action_dim

    # Temporary fix for dagger missing field
    if "student_policy" in cfg:
        print("Applying dagger field hotfix")
        cfg.action_dim = cfg.student_policy.action_dim

    # Temporary fix for critic missing field in actor config
    if "critic" in cfg:
        print("Applying critic field hotfix")
        cfg.actor.critic = cfg.critic
        cfg.actor.init_logstd = cfg.init_logstd
        cfg.discount = cfg.base_policy.discount
    
    # Create actor and load weights
    actor: Actor = get_actor(cfg=cfg, device=device)
    
    if "model_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["model_state_dict"])
    else:
        actor.load_state_dict(checkpoint)
    
    actor.eval()
    actor.to(device)
    
    # Determine observation type
    observation_type = cfg.get("observation_type", "image")
    print(f"Observation type: {observation_type}")
    
    # Verify if using image encoders or only proprioceptive
    has_encoders = (
        hasattr(actor, "encoder1") 
        and hasattr(actor, "encoder2") 
        and actor.encoder1 is not None 
        and actor.encoder2 is not None
    )
    
    print("\n" + "-" * 60)
    print("VERIFICATION: What information is being used?")
    print("-" * 60)
    if observation_type == "image" and has_encoders:
        print(f"✓ Using IMAGE observations with encoders")
        print(f"  Encoder 1: {type(actor.encoder1).__name__}")
        print(f"  Encoder 2: {type(actor.encoder2).__name__}")
        if hasattr(actor, "encoding_dim"):
            print(f"  Encoding dimension: {actor.encoding_dim}")
        print(f"  Latents will include: encoder features from images")
    elif observation_type == "state":
        print(f"✓ Using STATE observations (robot_state + parts_poses)")
        print(f"  Robot state dim: {getattr(actor, 'robot_state_dim', 'unknown')}")
        print(f"  Using ONLY proprioceptive information (no images)")
        print(f"  Latents will include: normalized robot_state + parts_poses")
    else:
        print(f"⚠ Warning: observation_type={observation_type} but encoders available: {has_encoders}")
        if not has_encoders:
            print(f"  This suggests the model uses only proprioceptive information")
    print("-" * 60 + "\n")
    
    # Step 2: Load rollout trajectory
    print("\n" + "=" * 60)
    print("Step 2: Loading rollout trajectory...")
    print("=" * 60)
    
    rollout_path = Path(args.rollout_path)
    rollout_data = load_rollout_trajectory(rollout_path)
    rollout_observations = rollout_data["observations"]
    print(f"Loaded rollout with {len(rollout_observations)} timesteps")
    
    # Extract rollout latents
    print("\nExtracting rollout latents...")
    
    # Check what's actually in the rollout observations
    if len(rollout_observations) > 0:
        first_obs = rollout_observations[0]
        print(f"First observation keys: {list(first_obs.keys())}")
        if "color_image1" in first_obs:
            print(f"  Has color_image1: shape {np.array(first_obs['color_image1']).shape}")
        if "color_image2" in first_obs:
            print(f"  Has color_image2: shape {np.array(first_obs['color_image2']).shape}")
        if "robot_state" in first_obs:
            rs = first_obs["robot_state"]
            if isinstance(rs, dict):
                print(f"  Has robot_state (dict) with keys: {list(rs.keys())}")
            else:
                print(f"  Has robot_state: shape {np.array(rs).shape}")
        if "parts_poses" in first_obs:
            print(f"  Has parts_poses: shape {np.array(first_obs['parts_poses']).shape}")
    
    rollout_latents = extract_encoder_features(
        actor, rollout_observations, device, observation_type
    )
    print(f"Rollout latents shape: {rollout_latents.shape}")
    print(f"  Latent dimension: {rollout_latents.shape[1] if rollout_latents.ndim > 1 else len(rollout_latents)}")
    
    # Step 3: Load demonstration trajectories and extract latents
    print("\n" + "=" * 60)
    print("Step 3: Loading demonstration trajectories...")
    print("=" * 60)
    
    demo_latents = None
    cache_path = None

    # Demo dir specified - always recalculate (don't use cache when --demo-dir is specified)
    demo_dir = Path(args.demo_dir)
    
    cache_key = get_cache_key(demo_dir, args.max_demo_samples, observation_type)
    cache_path = get_cache_path(cache_key)

    if cache_path.exists():
        print(f"Using cached demo latents from: {cache_path.name}")
        demo_latents = load_demo_latents_cache(cache_path)
    else:
        print("No cached demo latents found, calculating...")
        demo_observations = load_demo_trajectories(
            demo_dir, max_episodes=args.max_demo_episodes
        )
        
        # Limit the number of observations if specified
        if args.max_demo_samples is not None and len(demo_observations) > args.max_demo_samples:
            print(f"Limiting demo observations to {args.max_demo_samples} (from {len(demo_observations)})")
            demo_observations = demo_observations[:args.max_demo_samples]
        
        # Check what's in the demo observations
        if len(demo_observations) > 0:
            first_obs = demo_observations[0]
            print(f"First demo observation keys: {list(first_obs.keys())}")
            if "color_image1" in first_obs:
                print(f"  Has color_image1: shape {np.array(first_obs['color_image1']).shape}")
            if "color_image2" in first_obs:
                print(f"  Has color_image2: shape {np.array(first_obs['color_image2']).shape}")
            if "robot_state" in first_obs:
                rs = first_obs["robot_state"]
                if isinstance(rs, dict):
                    print(f"  Has robot_state (dict) with keys: {list(rs.keys())}")
                else:
                    print(f"  Has robot_state: shape {np.array(rs).shape}")
            if "parts_poses" in first_obs:
                print(f"  Has parts_poses: shape {np.array(first_obs['parts_poses']).shape}")
        
        # Extract demo latents using the same function as rollout
        print("\nExtracting demo latents...")
        demo_latents = extract_encoder_features(
            actor, demo_observations, device, observation_type
        )
        print(f"Demo latents shape: {demo_latents.shape}")
        print(f"  Latent dimension: {demo_latents.shape[1] if demo_latents.ndim > 1 else len(demo_latents)}")
        
        # Save to cache
        save_demo_latents_cache(cache_path, demo_latents)
    
    if demo_latents is None:
        raise ValueError("Failed to load demo latents. Please specify --demo-dir to generate them.")
    
    # Step 4: Calculate distances
    print("\n" + "=" * 60)
    print("Step 4: Calculating distances...")
    print("=" * 60)
    
    # Calculate distances (this may modify the latents to fix dimension mismatch)
    distances, rollout_latents_fixed, demo_latents_fixed = calculate_min_distances(rollout_latents, demo_latents)
    print(f"Distance statistics:")
    print(f"  Mean: {np.mean(distances):.4f}")
    print(f"  Std: {np.std(distances):.4f}")
    print(f"  Min: {np.min(distances):.4f}")
    print(f"  Max: {np.max(distances):.4f}")
    
    # Use the fixed latents for visualization
    rollout_latents = rollout_latents_fixed
    demo_latents = demo_latents_fixed
    
    # Step 5: Create visualizations
    print("\n" + "=" * 60)
    print("Step 5: Creating visualizations...")
    print("=" * 60)
    
    # Create rollout video frames
    print("Preparing rollout video...")
    rollout_video = []
    has_images_in_rollout = False
    
    for obs in rollout_observations:
        # Check if images exist in observation (regardless of observation_type)
        if "color_image1" in obs and "color_image2" in obs:
            has_images_in_rollout = True
            img1 = obs["color_image1"]
            img2 = obs["color_image2"]
            
            # Ensure images are uint8 and have correct shape
            if isinstance(img1, torch.Tensor):
                img1 = img1.cpu().numpy()
            if isinstance(img2, torch.Tensor):
                img2 = img2.cpu().numpy()
            
            # Handle different image formats
            if len(img1.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
                img1 = img1[0]
                img2 = img2[0]
            if len(img1.shape) == 3 and img1.shape[0] == 3:  # (C, H, W)
                img1 = img1.transpose(1, 2, 0)
                img2 = img2.transpose(1, 2, 0)
            
            # Ensure uint8
            if img1.dtype != np.uint8:
                img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
            
            # Resize if needed to match
            if img1.shape[:2] != img2.shape[:2]:
                # Resize to same height
                h = min(img1.shape[0], img2.shape[0])
                img1 = img1[:h]
                img2 = img2[:h]
            
            # Concatenate side by side
            combined = np.concatenate([img1, img2], axis=1)
            rollout_video.append(combined)
        else:
            # For state observations or when images don't exist, create a dummy image with text
            dummy_img = np.zeros((240, 640, 3), dtype=np.uint8)
            # Add text indicating no images
            if cv2 is not None:
                cv2.putText(dummy_img, "No Images Available", (50, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(dummy_img, f"State Observation Only", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            rollout_video.append(dummy_img)
    
    rollout_video = np.array(rollout_video)
    print(f"Rollout video shape: {rollout_video.shape}")
    print(f"Has images in rollout: {has_images_in_rollout}")
    
    # Create distance video
    print("Creating distance video...")
    distance_video_path = output_dir / "distance_video.mp4"
    create_distance_video(rollout_video, distances, distance_video_path, fps=args.fps)
    
    # Create 2D latent space visualization
    print("Creating 2D latent space visualization...")
    latent_2d_path = output_dir / f"latent_space_2d_{args.reduction_method}.png"
    visualize_latent_space_2d(
        rollout_latents,
        demo_latents,
        latent_2d_path,
        method=args.reduction_method,
    )
    
    # Save distance plot as static image
    print("Creating static distance plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, 'b-', linewidth=2)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Min Distance to Demo Latents', fontsize=12)
    ax.set_title('Latent Space Distance Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    distance_plot_path = output_dir / "distance_plot.png"
    plt.savefig(distance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance plot to {distance_plot_path}")
    
    print("\n" + "=" * 60)
    print("Done! Visualizations saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

