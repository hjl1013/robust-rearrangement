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

def calculate_adjacent_distances(rollout_latents: np.ndarray) -> np.ndarray:
    """Calculate distances between adjacent observations in the rollout."""
    distances = np.linalg.norm(np.diff(rollout_latents, axis=0), axis=1)

    # print statistics of distances
    print(f"Distance statistics:")
    print(f"  Mean: {np.mean(distances):.4f}")
    print(f"  Std: {np.std(distances):.4f}")
    print(f"  Min: {np.min(distances):.4f}")
    print(f"  Max: {np.max(distances):.4f}")

    return distances

def create_adjacent_distance_video(
    rollout_video: np.ndarray, 
    adjacent_distances: np.ndarray, 
    rollout_latents_2d: np.ndarray,
    output_path: Path, 
    fps: int = 20
):
    """Create a video with rollout on left, adjacent distances in middle, and t-SNE on right."""
    frames = []
    for i in tqdm(range(len(rollout_video)-1), desc="Creating video frames"):
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], hspace=0.3)
        
        # Left: video frame
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')
        if i < len(rollout_video) and rollout_video[i].size > 0:
            try:
                ax1.imshow(rollout_video[i])
            except Exception as e:
                print(f"Warning: Could not display frame {i}: {e}")
                blank = np.zeros((240, 640, 3), dtype=np.uint8)
                ax1.imshow(blank)
        else:
            blank = np.zeros((240, 640, 3), dtype=np.uint8)
            ax1.imshow(blank)
        ax1.set_title(f'Rollout Frame {i}', fontsize=12)
        
        # Middle: distance graph
        ax2 = fig.add_subplot(gs[1])
        timesteps = np.arange(i + 1)
        current_distances = adjacent_distances[:i + 1]
        ax2.plot(timesteps, current_distances, 'b-', linewidth=2, label='Distance')
        if i < len(adjacent_distances):
            ax2.plot(i, adjacent_distances[i], 'ro', markersize=10, label='Current')
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Adjacent Distance', fontsize=12)
        ax2.set_title('Adjacent Distances Over Time', fontsize=14)
        ax2.set_xlim(0, len(adjacent_distances))
        ax2.set_ylim(0, np.max(adjacent_distances) * 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Right: t-SNE visualization
        ax3 = fig.add_subplot(gs[2])
        # Plot trajectory up to current timestep
        if i + 1 <= len(rollout_latents_2d):
            current_latents_2d = rollout_latents_2d[:i + 1]
            n_current = len(current_latents_2d)
            opacities = np.linspace(0.3, 1.0, n_current)
            
            # Plot all points up to current timestep
            ax3.scatter(
                current_latents_2d[:, 0],
                current_latents_2d[:, 1],
                c='blue',
                alpha=opacities,
                s=30,
                edgecolors='none',
            )
            
            # Draw line connecting points
            if len(current_latents_2d) > 1:
                ax3.plot(
                    current_latents_2d[:, 0],
                    current_latents_2d[:, 1],
                    'b-',
                    alpha=0.3,
                    linewidth=1,
                )
            
            # Highlight current point
            if i < len(rollout_latents_2d):
                ax3.scatter(
                    rollout_latents_2d[i, 0],
                    rollout_latents_2d[i, 1],
                    c='red',
                    s=100,
                    marker='*',
                    edgecolors='black',
                    linewidths=2,
                    label='Current',
                    zorder=10,
                )
        
        ax3.set_xlabel('t-SNE Component 1', fontsize=12)
        ax3.set_ylabel('t-SNE Component 2', fontsize=12)
        ax3.set_title('Latent Space (t-SNE)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        if i < len(rollout_latents_2d):
            ax3.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
    
    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"Saved adjacent distance video to {output_path} with fps {fps}")


def visualize_latent_space_2d_rollout(
    rollout_latents: np.ndarray,
    output_path: Path,
    method: str = "tsne",  # "tsne" or "pca"
):
    """Visualize rollout latent space in 2D."""
    # Ensure latents are 2D: (n_samples, latent_dim)
    if rollout_latents.ndim == 1:
        rollout_latents = rollout_latents.reshape(1, -1)
    elif rollout_latents.ndim > 2:
        rollout_latents = rollout_latents.reshape(rollout_latents.shape[0], -1)
    
    print(f"Reducing dimensionality using {method}...")
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(rollout_latents) - 1))
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    rollout_latents_2d = reducer.fit_transform(rollout_latents)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
    
    # Mark start and end points
    ax.scatter(
        rollout_latents_2d[0, 0],
        rollout_latents_2d[0, 1],
        c='green',
        s=150,
        marker='s',
        edgecolors='black',
        linewidths=2,
        label='Start',
        zorder=10,
    )
    ax.scatter(
        rollout_latents_2d[-1, 0],
        rollout_latents_2d[-1, 1],
        c='red',
        s=150,
        marker='s',
        edgecolors='black',
        linewidths=2,
        label='End',
        zorder=10,
    )
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title('Rollout Latent Space Visualization', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 2D latent visualization to {output_path}")
    
    return rollout_latents_2d

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
    print("Step 3: Calculating distances between adjacent observations...")
    print("=" * 60)
    
    adjacent_distances = calculate_adjacent_distances(rollout_latents)
    
    # Step 4: Create visualizations
    print("\n" + "=" * 60)
    print("Step 4: Creating visualizations...")
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
    
    # Create 2D latent space visualization (t-SNE)
    print("Creating 2D latent space visualization (t-SNE)...")
    latent_2d_path = output_dir / "latent_space_2d_tsne.png"
    rollout_latents_2d = visualize_latent_space_2d_rollout(
        rollout_latents,
        latent_2d_path,
        method="tsne",
    )
    
    # Save static distance plot
    print("Creating static distance plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(adjacent_distances, 'b-', linewidth=2)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Adjacent Distance', fontsize=12)
    ax.set_title('Adjacent Distances Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    distance_plot_path = output_dir / "adjacent_distance_plot.png"
    plt.savefig(distance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance plot to {distance_plot_path}")
    
    # Create adjacent distance video with t-SNE visualization
    print("Creating adjacent distance video with t-SNE visualization...")
    adjacent_distance_video_path = output_dir / "adjacent_distance_video.mp4"
    create_adjacent_distance_video(
        rollout_video, 
        adjacent_distances, 
        rollout_latents_2d,
        adjacent_distance_video_path, 
        fps=args.fps
    )
    
    print("\n" + "=" * 60)
    print("Done! Visualizations saved to:", output_dir)
    print("=" * 60)

if __name__ == "__main__":
    main()

