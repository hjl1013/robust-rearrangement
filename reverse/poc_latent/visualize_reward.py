"""Instantiate FurnitureSim-v0 and test various functionalities."""

import argparse
import pickle
import matplotlib.pyplot as plt
import furniture_bench
from pathlib import Path
from tqdm import tqdm
import imageio

import gym
import cv2
import torch
import numpy as np


def create_reward_video(
    rollout_video: np.ndarray,
    rewards: np.ndarray,
    ac_rewards: np.ndarray,
    output_path: Path,
    fps: int = 20,
):
    """Create a video with rollout on left and reward graph on right."""
    frames = []
    
    for t in tqdm(range(len(rollout_video)), desc="Creating video frames"):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
        
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
        
        # Right side: reward graph
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title('Reward Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax3.set_xlabel('Timestep', fontsize=12)
        ax3.set_ylabel('Action Reward', fontsize=12)
        ax3.set_title('Accumulated Reward Over Time', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Plot reward up to current timestep
        timesteps = np.arange(t)
        current_rewards = rewards[:t]
        current_ac_rewards = ac_rewards[:t]
        
        ax2.plot(timesteps, current_rewards, 'b-', linewidth=2, label='Reward')
        ax2.set_xlim(0, len(rewards))
        ax3.plot(timesteps, current_ac_rewards, 'b-', linewidth=2, label='Accumulated Reward')
        ax3.set_xlim(0, len(rewards))
        # Set ylim based on all rewards, not just current
        reward_min = np.min(rewards)
        reward_max = np.max(rewards)
        reward_range = reward_max - reward_min
        ac_reward_min = np.min(ac_rewards)
        ac_reward_max = np.max(ac_rewards)
        ac_reward_range = ac_reward_max - ac_reward_min
        if reward_range > 0:
            ax2.set_ylim(reward_min - 0.1 * reward_range, reward_max + 0.1 * reward_range)
            ax3.set_ylim(ac_reward_min - 0.1 * ac_reward_range, ac_reward_max + 0.1 * ac_reward_range)
        else:
            ax2.set_ylim(reward_min - 0.1, reward_max + 0.1)
            ax3.set_ylim(ac_reward_min - 0.1, ac_reward_max + 0.1)
        ax2.legend()
        ax3.legend()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
    
    # Save video
    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"Saved reward video to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="square_table")
    parser.add_argument(
        "--file-path", help="Demo path to replay (data directory or pickle)"
    )
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--init-assembled",
        action="store_true",
        help="Initialize the environment with the assembled furniture.",
    )
    parser.add_argument(
        "--save-camera-input",
        action="store_true",
        help="Save camera input of the simulator at the beginning of the episode.",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record the video of the simulator."
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input.",
    )
    parser.add_argument(
        "--randomness",
        default="low",
        help="Randomness level of the environment.",
    )
    parser.add_argument(
        "--high-random-idx",
        default=0,
        type=int,
        help="The index of high_randomness.",
    )
    parser.add_argument(
        "--env-id",
        default="FurnitureSim-v0",
        help="Environment id of FurnitureSim",
    )
    parser.add_argument(
        "--replay-path", type=str, help="Path to the saved data to replay action."
    )

    parser.add_argument(
        "--act-rot-repr",
        type=str,
        help="Rotation representation for action space.",
        choices=["quat", "axis", "rot_6d"],
        default="quat",
    )

    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation.",
    )

    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering.",
    )

    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reverse/poc_latent/visualization_reward",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for output video",
    )
    parser.add_argument(
        "--dense-reward",
        action="store_true",
        help="Use dense reward.",
    )
    parser.add_argument(
        "--latent-model",
        type=str,
        default=None,
        help="Path to latent model to use for dense reward.",
    )
    args = parser.parse_args()

    if args.dense_reward:
        # load latent model
        device = torch.device(f"cuda:{args.compute_device_id}" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(args.latent_model)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get config from checkpoint
        from omegaconf import DictConfig, OmegaConf
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
        from src.behavior import get_actor
        from src.behavior.base import Actor
        actor: Actor = get_actor(cfg=cfg, device=device)
        
        if "model_state_dict" in checkpoint:
            actor.load_state_dict(checkpoint["model_state_dict"])
        else:
            actor.load_state_dict(checkpoint)
        
        actor.eval()
        actor.to(device)
    else:
        actor = None

    # Create FurnitureSim environment.
    from src.gym.observation import FULL_OBS
    env = gym.make(
        args.env_id,
        furniture=args.furniture,
        num_envs=args.num_envs,
        resize_img=not args.high_res,
        init_assembled=args.init_assembled,
        record=args.record,
        headless=args.headless,
        save_camera_input=args.save_camera_input,
        randomness=args.randomness,
        high_random_idx=args.high_random_idx,
        # action_type="pos",
        obs_keys=FULL_OBS,
        ctrl_mode="diffik",
        act_rot_repr=args.act_rot_repr,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        dense_reward=args.dense_reward,
        latent_model=actor,
    )

    # Initialize FurnitureSim.
    ob = env.reset()
    done = False

    def action_tensor(ac):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(args.num_envs, 1).float().to(env.device)

    # Replay the trajectory.
    with open(args.replay_path, "rb") as f:
        data = pickle.load(f)

    rewards = []
    ac_rewards = []
    ac_reward = 0
    for obs, ac in tqdm(zip(data["observations"], data["actions"]), total=len(data["observations"]), desc="Calculating rewards"):
        env.reset_to([obs])
        ac = action_tensor(ac)
        ob, rew, done, _ = env.step(ac)
        ac_reward += rew.cpu().item()
        rewards.append(rew.cpu().item())
        ac_rewards.append(ac_reward)

    rewards = np.array(rewards)
    ac_rewards = np.array(ac_rewards)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract video frames from observations
    print("Preparing rollout video...")
    rollout_video = []
    has_images_in_rollout = False
    
    for obs in data["observations"]:
        # Check if images exist in observation
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
    
    # Create reward video
    print("Creating reward video...")
    reward_video_path = output_dir / "reward_video.mp4"
    create_reward_video(rollout_video, rewards, ac_rewards, reward_video_path, fps=args.fps)
    
    # Save static reward plot
    print("Creating static reward plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rewards, 'b-', linewidth=2)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_plot_path = output_dir / "reward_plot.png"
    plt.savefig(reward_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reward plot to {reward_plot_path}")
    
    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
