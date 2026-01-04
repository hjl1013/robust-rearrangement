import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pickle

import gym
import numpy as np
import scipy.spatial.transform as st
from pynput.keyboard import Key, Listener
from multiprocessing.managers import SharedMemoryManager
import matplotlib.pyplot as plt
import matplotlib

from furniture_bench.config import config
from furniture_bench.envs.furniture_recovery_sim_env import FurnitureRecoverySimEnv
from furniture_bench.envs.observation import FULL_OBS
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.device.spacemouse.spacemouse_shared_memory import Spacemouse
from src.data_collection.collect_enum import CollectEnum
from src.data_collection.data_collector_sm import precise_wait
from src.data_collection.keyboard_interface import KeyboardInterface
from src.data_processing.utils import resize, resize_crop
from src.gym import turn_off_april_tags
from src.visualization.render_mp4 import pickle_data
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

import torch

# Use non-interactive backend for matplotlib to allow updates
matplotlib.use('TkAgg')


class RecoveryKeyboardInterface(KeyboardInterface):
    """Extended keyboard interface for recovery teleoperation with save/discard keys."""
    
    def __init__(self):
        super().__init__()
        self.save_pressed = False
        self.discard_pressed = False
        self.save_failure_pressed = False  # 'f' key
        # Navigation commands
        self.ghost_prev_pressed = False  # 'u' key
        self.ghost_next_pressed = False  # 'i' key
        self.real_prev_pressed = False   # 'o' key
        self.real_next_pressed = False   # 'p' key
        # Graph display command
        self.show_graph_pressed = False  # 'g' key
        # Set indices command
        self.set_indices_pressed = False  # 'y' key
        # Undo command
        self.undo_pressed = False  # 'b' key
    
    def on_press(self, k):
        try:
            k_char = k.char
            
            # Ignore digit keys (don't record rewards in recovery mode)
            # if k_char.isdigit():
            #     gym.logger.info(f"Digit key '{k_char}' pressed but ignored (no reward recording in recovery mode)")
            #     return
            
            # Handle save and discard keys first
            if k_char == "s":
                gym.logger.info("Save pressed")
                self.save_pressed = True
                self.key_enum = CollectEnum.SUCCESS  # Use SUCCESS for save
                return
            elif k_char == "f":
                gym.logger.info("Save failure pressed")
                self.save_failure_pressed = True
                return
            elif k_char == "d":
                gym.logger.info("Discard pressed")
                self.discard_pressed = True
                self.key_enum = CollectEnum.FAIL  # Use FAIL for discard
                return
            # Handle graph display
            elif k_char == "g":
                gym.logger.info("Show graph pressed")
                self.show_graph_pressed = True
                return
            # Handle set indices (override parent's 'y' for SUCCESS_RECORD)
            elif k_char == "y":
                gym.logger.info("Set indices pressed")
                self.set_indices_pressed = True
                return
            # Handle undo
            elif k_char == "b":
                gym.logger.info("Undo pressed")
                self.undo_pressed = True
                return
            # Handle navigation keys
            elif k_char == "u":
                gym.logger.info("Ghost previous pressed")
                self.ghost_prev_pressed = True
                return
            elif k_char == "i":
                gym.logger.info("Ghost next pressed")
                self.ghost_next_pressed = True
                return
            elif k_char == "o":
                gym.logger.info("Real previous pressed")
                self.real_prev_pressed = True
                return
            elif k_char == "p":
                gym.logger.info("Real next pressed")
                self.real_next_pressed = True
                return
            elif k_char == "r":
                gym.logger.info("Reset pressed")
                self.key_enum = CollectEnum.RESET
                return

            # Call parent's on_press for other keys (pass original key object)
            # super().on_press(k)
        except AttributeError:
            # Handle special keys (like ESC)
            if k == Key.esc:
                self.key_enum = CollectEnum.TERMINATE
            else:
                # Pass to parent for other special keys
                super().on_press(k)
            pass
    
    def reset(self):
        super().reset()
        self.save_pressed = False
        self.discard_pressed = False
        self.save_failure_pressed = False
        self.ghost_prev_pressed = False
        self.ghost_next_pressed = False
        self.real_prev_pressed = False
        self.real_next_pressed = False
        self.show_graph_pressed = False
        self.set_indices_pressed = False
        self.undo_pressed = False


class RecoveryTeleopCollector:
    """Data collector for recovery teleoperation."""
    
    def __init__(
        self,
        env: FurnitureRecoverySimEnv,
        device_interface: DeviceInterface,
        output_dir: Path,
        furniture: str,
        ctrl_mode: str = "diffik",
    ):
        self.env = env
        self.device_interface = device_interface
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.furniture = furniture
        self.ctrl_mode = ctrl_mode
        
        # Collection state
        self.transitions = []
        self.recording = True
        self.robot_settled = False
        self.starttime = None
        
        # OOD graph state
        self.ood_fig = None
        self.ood_ax = None
        self.trajectory_data_cache = None
        
        # Timing parameters
        self.frequency = 10
        self.dt = 1 / self.frequency
        self.command_latency = 0.01
        self.start_delay = 2
        
        # SpaceMouse speed parameters
        self.max_pos_speed = 0.3 if ctrl_mode == "diffik" else 0.8
        self.max_rot_speed = 0.7 if ctrl_mode == "diffik" else 4.0
        self.deadzone = 0.05
        
        # SpaceMouse parameters
        self.sm_dpos_scalar = np.array([1.8] * 3)
        self.sm_drot_scalar = np.array([4] * 3)
        self.use_spacemouse = True  # Enable SpaceMouse by default
        
        # Action bounds
        self.pos_bounds_m = 0.025
        self.ori_bounds_deg = 20
        
        # Metadata
        self.metadata = {
            "max_force_magnitude": self.env.max_force_magnitude,
            "max_torque_magnitude": self.env.max_torque_magnitude,
            "max_obstacle_offset": self.env.max_obstacle_offset,
            "franka_joint_rand_lim_deg": self.env.franka_joint_rand_lim_deg,
            "ctrl_mode": ctrl_mode,
            "pos_bounds_m": self.pos_bounds_m,
            "ori_bounds_deg": self.ori_bounds_deg,
            "frequency": self.frequency,
            "command_latency": self.command_latency,
            "deadzone": self.deadzone,
            "max_pos_speed": self.max_pos_speed,
            "max_rot_speed": self.max_rot_speed,
            "sm_dpos_scalar": self.sm_dpos_scalar.tolist(),
            "sm_drot_scalar": self.sm_drot_scalar.tolist(),
        }
    
    def _reset_collector_buffer(self):
        """Reset the collection buffer."""
        self.transitions = []
        self.recording = True
        self.robot_settled = False
    
    def _squeeze_and_numpy(self, d):
        """Recursively squeeze and convert tensors to numpy arrays."""
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self._squeeze_and_numpy(v)
            elif v is None:
                continue
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                d[k] = v.squeeze()
            elif k == "rewards":
                d[k] = float(v)
            elif k == "skills":
                d[k] = int(v)
        return d
    
    def store_transition(self, obs, action=None, rew=None, skill_complete=None, setup_phase=False):
        """Store a transition."""
        if (not setup_phase) and (not self.robot_settled or not self.recording):
            return
        
        n_ob = {}
        n_ob["color_image1"] = resize(obs["color_image1"])
        n_ob["color_image2"] = resize_crop(obs["color_image2"])
        n_ob["robot_state"] = obs["robot_state"]
        n_ob["parts_poses"] = obs["parts_poses"]
        
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.squeeze().cpu().numpy()
            elif isinstance(action, np.ndarray):
                action = action.squeeze()
        
        if rew is not None:
            if isinstance(rew, torch.Tensor):
                rew = rew.item()
            elif isinstance(rew, np.ndarray):
                rew = rew.item()
            elif isinstance(rew, (float, int)):
                rew = float(rew)
        
        transition = {
            "observations": n_ob,
            "actions": action,
            "rewards": rew,
            "skills": skill_complete,
        }
        
        transition = self._squeeze_and_numpy(transition)
        self.transitions.append(transition)
    
    def save(self, trajectory_name: str, success: bool = True):
        """Save collected trajectory."""
        print(f"Saving trajectory: {trajectory_name} (success: {success})")
        print(f"Length of trajectory: {len(self.transitions)}")
        
        if len(self.transitions) == 0:
            print("Warning: No transitions to save!")
            return
        
        # Prepare data
        data = {}
        data["observations"] = [t["observations"] for t in self.transitions]
        data["actions"] = [t["actions"] for t in self.transitions][:-1]  # Actions are one less than observations
        data["rewards"] = [t["rewards"] for t in self.transitions][:-1]
        data["skills"] = [t["skills"] for t in self.transitions][:-1]
        data["success"] = success
        data["furniture"] = self.furniture
        data["metadata"] = self.metadata
        
        # Save to recovery directory with same name as trajectory
        if success:
            output_dir = self.output_dir / "success"
            save_path = output_dir / f"{trajectory_name}.pkl"
        else:
            output_dir = self.output_dir / "failure"
            save_path = output_dir / f"{trajectory_name}.pkl"
        output_dir.mkdir(parents=True, exist_ok=True)
        pickle_data(data, save_path)
        print(f"Data saved at {save_path}")
    
    def show_ood_graph(self, trajectory_data: dict):
        """Display OOD score graph with current indices."""
        if "out_of_distribution_scores" not in trajectory_data:
            print("Warning: No OOD scores in trajectory data")
            return
        
        ood_scores = trajectory_data["out_of_distribution_scores"]
        in_dist_idx = self.env.in_dist_state_idx if hasattr(self.env, 'in_dist_state_idx') else 0
        ood_idx = self.env.ood_state_idx if hasattr(self.env, 'ood_state_idx') else 0
        
        # Get threshold and consecutive steps if available
        threshold = trajectory_data.get("threshold", 0.0)
        consecutive_steps = trajectory_data.get("consecutive_steps", 1)
        
        # Create or update figure
        if not hasattr(self, 'ood_fig') or self.ood_fig is None:
            plt.ion()  # Enable interactive mode
            self.ood_fig, self.ood_ax = plt.subplots(figsize=(12, 6))
            self.ood_fig.canvas.manager.set_window_title('OOD Score Graph')
        else:
            self.ood_ax.clear()
        
        # Plot OOD scores
        timesteps = np.arange(len(ood_scores))
        self.ood_ax.plot(timesteps, ood_scores, 'b-', linewidth=2, label='OOD Score')
        
        # Draw threshold line
        if threshold > 0:
            self.ood_ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                               label=f'Threshold ({threshold:.3f})')
        
        # Highlight consecutive steps region
        if ood_idx is not None and consecutive_steps > 1:
            start_idx = max(0, ood_idx + 1)
            end_idx = ood_idx + consecutive_steps
            self.ood_ax.axvspan(start_idx, end_idx, alpha=0.3, color='yellow', 
                               label=f'Consecutive Steps ({consecutive_steps})')
        
        # Highlight current indices
        if in_dist_idx < len(ood_scores):
            self.ood_ax.plot(in_dist_idx, ood_scores[in_dist_idx], 
                           'go', markersize=15, label=f'In-Distribution (t={in_dist_idx})', zorder=5)
            self.ood_ax.axvline(x=in_dist_idx, color='g', linestyle=':', alpha=0.5)
        
        if ood_idx < len(ood_scores):
            self.ood_ax.plot(ood_idx, ood_scores[ood_idx], 
                           'ro', markersize=15, label=f'Out-of-Distribution (t={ood_idx})', zorder=5)
            self.ood_ax.axvline(x=ood_idx, color='r', linestyle=':', alpha=0.5)
        
        # Set labels and title
        self.ood_ax.set_xlabel('Timestep', fontsize=12)
        self.ood_ax.set_ylabel('Out-of-Distribution Score', fontsize=12)
        self.ood_ax.set_title('OOD Score Over Time', fontsize=14, fontweight='bold')
        self.ood_ax.grid(True, alpha=0.3)
        
        # Set limits
        self.ood_ax.set_xlim(0, len(ood_scores) - 1)
        y_max = np.max(ood_scores) * 1.1 if len(ood_scores) > 0 else 1.0
        self.ood_ax.set_ylim(0, max(y_max, threshold * 1.2))
        
        # Add legend
        self.ood_ax.legend(loc='upper right', fontsize=10)
        
        # Update display
        self.ood_fig.canvas.draw()
        self.ood_fig.canvas.flush_events()
        plt.show(block=False)
        
        print(f"OOD Graph displayed: In-dist={in_dist_idx}, OOD={ood_idx}")
    
    def set_indices_manually(self):
        """Manually set in-distribution and OOD indices."""
        if self.trajectory_data_cache is None or "observations" not in self.trajectory_data_cache:
            print("Warning: No trajectory data available")
            return False
        
        max_idx = len(self.trajectory_data_cache["observations"]) - 1
        
        try:
            print(f"\nCurrent indices:")
            print(f"  In-distribution: {self.env.in_dist_state_idx if hasattr(self.env, 'in_dist_state_idx') else 'N/A'}")
            print(f"  Out-of-distribution: {self.env.ood_state_idx if hasattr(self.env, 'ood_state_idx') else 'N/A'}")
            print(f"  Valid range: [0, {max_idx}]")
            
            in_dist_input = input("Enter in-distribution index: ").strip()
            ood_input = input("Enter out-of-distribution index: ").strip()
            
            in_dist_idx = int(in_dist_input)
            ood_idx = int(ood_input)
            
            # Validate indices
            if in_dist_idx < 0 or in_dist_idx > max_idx:
                print(f"Error: In-distribution index {in_dist_idx} out of range [0, {max_idx}]")
                return False
            
            if ood_idx < 0 or ood_idx > max_idx:
                print(f"Error: Out-of-distribution index {ood_idx} out of range [0, {max_idx}]")
                return False
            
            # Update environment indices
            self.env.in_dist_state_idx = in_dist_idx
            self.env.ood_state_idx = ood_idx
            
            print(f"Updated indices: In-dist={in_dist_idx}, OOD={ood_idx}")
            
            # Update ghost objects to new in-distribution state
            observations = self.trajectory_data_cache["observations"]
            in_dist_state = observations[in_dist_idx]
            env_idx = 0  # Only support single environment
            
            if "robot_state" in in_dist_state:
                self.env._update_ghost_robot_pose(env_idx, in_dist_state["robot_state"])
            
            if "parts_poses" in in_dist_state:
                parts_poses = in_dist_state["parts_poses"]
                if isinstance(parts_poses, torch.Tensor):
                    parts_poses = parts_poses.cpu().numpy()
                self.env._update_ghost_parts_poses(env_idx, parts_poses)
            
            # Update real objects to new OOD state
            ood_state = observations[ood_idx]
            self.env.reset_env_to(env_idx, ood_state)
            self.env.refresh()
            
            # Update graph if it's open
            if self.ood_fig is not None and plt.fignum_exists(self.ood_fig.number):
                self.show_ood_graph(self.trajectory_data_cache)
            
            return True
            
        except ValueError as e:
            print(f"Error: Invalid input - {e}")
            return False
        except Exception as e:
            print(f"Error setting indices: {e}")
            return False
    
    def undo_last_action(self):
        """Undo the last recorded action."""
        if len(self.transitions) > 1:  # Keep at least the initial observation
            removed = self.transitions.pop()
            print(f"Undone last action. Transitions: {len(self.transitions)}")
            return True
        else:
            print("Cannot undo: only initial observation remains")
            return False
    
    def collect_recovery_trajectory(self, trajectory_path: Path):
        """Collect recovery teleoperation for a single trajectory."""
        trajectory_name = trajectory_path.stem
        
        print(f"\n{'='*60}")
        print(f"Starting recovery teleoperation for: {trajectory_name}")
        print(f"{'='*60}")
        
        # Load and cache trajectory data for graph display
        with open(trajectory_path, 'rb') as f:
            self.trajectory_data_cache = pickle.load(f)
        
        # Reset environment with trajectory
        obs = self.env.reset(trajectory_path=str(trajectory_path))
        self._reset_collector_buffer()
        
        # Store initial observation
        self.store_transition(obs, setup_phase=True)
        
        # Wait for robot to settle
        print("Waiting for robot to settle...")
        self.starttime = datetime.now()
        self.robot_settled = False
        
        # Main teleoperation loop
        t_start = time.monotonic()
        iter_idx = 0
        done = False
        
        # Set target pose
        translation, quat_xyzw = self.env.get_ee_pose()
        translation, quat_xyzw = (
            translation.cpu().numpy().squeeze(),
            quat_xyzw.cpu().numpy().squeeze(),
        )
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        target_pose_rv = np.array([*translation, *rotvec])
        target_pose_last_action_rv = None
        
        gripper_width = self.env.gripper_width()
        gripper_open = gripper_width >= 0.06
        grasp_flag = torch.from_numpy(np.array([-1 if gripper_open else 1])).to(self.env.device)
        
        def pose_rv2mat(pose_rv):
            pose_mat = np.eye(4)
            pose_mat[:-1, -1] = pose_rv[:3]
            pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_rv[3:]).as_matrix()
            return pose_mat
        
        def to_isaac_dpose_from_abs(current_pose_mat, goal_pose_mat, grasp_flag, device, rm=True):
            """Convert from absolute current and desired pose to delta pose."""
            if rm:
                delta_rot_mat = (
                    np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
                )
            else:
                delta_rot_mat = goal_pose_mat[:-1:-1] @ np.linalg.inv(
                    current_pose_mat[:-1, :-1]
                )
            
            dpos = goal_pose_mat[:-1, -1] - current_pose_mat[:-1, -1]
            target_translation = torch.from_numpy(dpos).float().to(device)
            
            target_rot = st.Rotation.from_matrix(delta_rot_mat)
            target_quat_xyzw = torch.from_numpy(target_rot.as_quat()).float().to(device)
            target_dpose = torch.cat(
                (target_translation, target_quat_xyzw, grasp_flag), dim=-1
            ).reshape(1, -1)
            return target_dpose
        
        # SpaceMouse setup
        with SharedMemoryManager() as shm_manager:
            sm = None
            if self.use_spacemouse:
                try:
                    sm = Spacemouse(shm_manager=shm_manager, deadzone=self.deadzone)
                    sm.__enter__()
                    print("SpaceMouse connected")
                except Exception as e:
                    print(f"Warning: Could not connect to SpaceMouse: {e}")
                    print("Continuing with keyboard-only teleoperation")
                    sm = None
            
            prev_keyboard_gripper = -1
            ready_to_grasp = True
            steps_since_grasp = 0
            record_latency_when_grasping = 8
            
            while not done:
                # Check if robot has settled
                if (not self.robot_settled) and (
                    (datetime.now() - self.starttime).seconds > self.start_delay
                ):
                    self.robot_settled = True
                    print("\nRecovery Teleoperation Controls:")
                    print("  'r' - Reset to current trajectory (retry)")
                    print("  's' - Save recovery trajectory (success)")
                    print("  'f' - Save recovery trajectory (failure)")
                    print("  'd' - Discard and move to next trajectory")
                    print("  'g' - Show/update OOD score graph")
                    print("  'y' - Set in-distribution and OOD indices manually")
                    print("  'u' - Move ghost to previous state")
                    print("  'i' - Move ghost to next state")
                    print("  'o' - Move real to previous state")
                    print("  'p' - Move real to next state")
                    print("  'b' - Undo last recorded action")
                    print("  ESC - Terminate program")
                
                # Calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * self.dt
                t_sample = t_cycle_end - self.command_latency
                precise_wait(t_sample)
                
                # Get SpaceMouse input if available
                dpos = np.zeros(3)
                drot_xyz = np.zeros(3)
                drot = st.Rotation.identity()
                if sm is not None:
                    try:
                        sm_state = sm.get_motion_state_transformed()
                        # Scale position command
                        dpos = (
                            sm_state[:3]
                            * (self.max_pos_speed / self.frequency)
                            * self.sm_dpos_scalar
                        )
                        
                        # Convert and scale rotation command
                        drot_xyz = sm_state[3:] * (self.max_rot_speed / self.frequency)
                        drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
                        drot_rotvec *= self.sm_drot_scalar
                        drot = st.Rotation.from_rotvec(drot_rotvec)
                    except Exception as e:
                        print(f"Warning: SpaceMouse error: {e}")
                        drot = st.Rotation.identity()
                        dpos = np.zeros(3)
                        drot_xyz = np.zeros(3)
                
                # Get keyboard action
                keyboard_action, collect_enum = self.device_interface.get_action()
                
                # Handle graph display
                if hasattr(self.device_interface, 'show_graph_pressed') and self.device_interface.show_graph_pressed:
                    if self.trajectory_data_cache is not None:
                        self.show_ood_graph(self.trajectory_data_cache)
                    else:
                        print("Warning: No trajectory data available for graph")
                    self.device_interface.show_graph_pressed = False
                
                # Handle set indices
                if hasattr(self.device_interface, 'set_indices_pressed') and self.device_interface.set_indices_pressed:
                    if self.set_indices_manually():
                        # Update target pose after resetting real objects
                        translation, quat_xyzw = self.env.get_ee_pose()
                        translation, quat_xyzw = (
                            translation.cpu().numpy().squeeze(),
                            quat_xyzw.cpu().numpy().squeeze(),
                        )
                        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                        target_pose_rv = np.array([*translation, *rotvec])
                        target_pose_last_action_rv = None
                    self.device_interface.set_indices_pressed = False
                
                # Handle undo
                if hasattr(self.device_interface, 'undo_pressed') and self.device_interface.undo_pressed:
                    self.undo_last_action()
                    self.device_interface.undo_pressed = False
                
                # Handle navigation keys first (they don't affect collect_enum)
                graph_needs_update = False
                
                if hasattr(self.device_interface, 'ghost_prev_pressed') and self.device_interface.ghost_prev_pressed:
                    self.env.navigate_ghost_prev()
                    self.device_interface.ghost_prev_pressed = False
                    graph_needs_update = True
                
                if hasattr(self.device_interface, 'ghost_next_pressed') and self.device_interface.ghost_next_pressed:
                    self.env.navigate_ghost_next()
                    self.device_interface.ghost_next_pressed = False
                    graph_needs_update = True
                
                if hasattr(self.device_interface, 'real_prev_pressed') and self.device_interface.real_prev_pressed:
                    self.env.navigate_real_prev()
                    self.device_interface.real_prev_pressed = False
                    graph_needs_update = True
                    # Update target pose after resetting real objects
                    translation, quat_xyzw = self.env.get_ee_pose()
                    translation, quat_xyzw = (
                        translation.cpu().numpy().squeeze(),
                        quat_xyzw.cpu().numpy().squeeze(),
                    )
                    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                    target_pose_rv = np.array([*translation, *rotvec])
                    target_pose_last_action_rv = None
                
                if hasattr(self.device_interface, 'real_next_pressed') and self.device_interface.real_next_pressed:
                    self.env.navigate_real_next()
                    self.device_interface.real_next_pressed = False
                    graph_needs_update = True
                    # Update target pose after resetting real objects
                    translation, quat_xyzw = self.env.get_ee_pose()
                    translation, quat_xyzw = (
                        translation.cpu().numpy().squeeze(),
                        quat_xyzw.cpu().numpy().squeeze(),
                    )
                    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                    target_pose_rv = np.array([*translation, *rotvec])
                    target_pose_last_action_rv = None
                
                # Update graph if navigation keys were pressed and graph is open
                if graph_needs_update and self.ood_fig is not None and plt.fignum_exists(self.ood_fig.number):
                    if self.trajectory_data_cache is not None:
                        self.show_ood_graph(self.trajectory_data_cache)
                
                # Handle special keys
                if collect_enum == CollectEnum.PAUSE:
                    self.recording = False
                    print("Paused recording")
                elif collect_enum == CollectEnum.CONTINUE:
                    self.recording = True
                    print("Continued recording")
                elif collect_enum == CollectEnum.RESET:
                    # Reset to current trajectory again
                    print("Resetting to current trajectory...")
                    obs = self.env.reset(trajectory_path=str(trajectory_path))
                    self._reset_collector_buffer()
                    self.store_transition(obs, setup_phase=True)
                    self.starttime = datetime.now()
                    self.robot_settled = False
                    
                    # Reset target pose
                    translation, quat_xyzw = self.env.get_ee_pose()
                    translation, quat_xyzw = (
                        translation.cpu().numpy().squeeze(),
                        quat_xyzw.cpu().numpy().squeeze(),
                    )
                    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                    target_pose_rv = np.array([*translation, *rotvec])
                    target_pose_last_action_rv = None
                    continue
                elif collect_enum == CollectEnum.TERMINATE:
                    print("Terminating program")
                    # Cleanup SpaceMouse before returning
                    if sm is not None:
                        try:
                            sm.__exit__(None, None, None)
                        except Exception:
                            pass
                    return None
                elif collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                    # Save or discard
                    if collect_enum == CollectEnum.SUCCESS:
                        # Save
                        self.save(trajectory_name, success=True)
                    else:
                        # Discard
                        print(f"Discarding trajectory: {trajectory_name}")
                    
                    # Cleanup SpaceMouse before returning
                    if sm is not None:
                        try:
                            sm.__exit__(None, None, None)
                        except Exception:
                            pass
                    # Return status
                    return collect_enum == CollectEnum.SUCCESS
                
                # Handle save failure
                if hasattr(self.device_interface, 'save_failure_pressed') and self.device_interface.save_failure_pressed:
                    # Save as failure
                    self.save(trajectory_name, success=False)
                    self.device_interface.save_failure_pressed = False
                    
                    # Cleanup SpaceMouse before returning
                    if sm is not None:
                        try:
                            sm.__exit__(None, None, None)
                        except Exception:
                            pass
                    # Return status
                    return True
                
                # Handle gripper actions
                steps_since_grasp += 1
                if steps_since_grasp > record_latency_when_grasping:
                    ready_to_grasp = True
                if steps_since_grasp < record_latency_when_grasping:
                    action_taken = True
                
                kb_grasp = prev_keyboard_gripper != keyboard_action[-1]
                sm_grasp = False
                if sm is not None:
                    try:
                        sm_grasp = (
                            sm.is_button_pressed(0) or sm.is_button_pressed(1)
                        ) and ready_to_grasp
                    except Exception:
                        pass
                
                if kb_grasp or sm_grasp:
                    grasp_flag = -1 * grasp_flag
                    gripper_open = not gripper_open
                    ready_to_grasp = False
                    steps_since_grasp = 0
                prev_keyboard_gripper = keyboard_action[-1]
                
                # Apply SpaceMouse delta to target pose
                new_target_pose_rv = target_pose_rv.copy()
                new_target_pose_rv[:3] += dpos
                new_target_pose_rv[3:] = (
                    drot * st.Rotation.from_rotvec(target_pose_rv[3:])
                ).as_rotvec()
                
                # Update target pose based on keyboard action
                if not np.allclose(keyboard_action[:6], 0.0):
                    # Keyboard action is delta, so update target pose
                    kb_dpos = keyboard_action[:3]
                    kb_dquat = keyboard_action[3:7]
                    
                    # Convert quaternion delta to rotation
                    kb_dquat_rot = st.Rotation.from_quat(kb_dquat)
                    new_rotvec = (kb_dquat_rot * st.Rotation.from_rotvec(new_target_pose_rv[3:])).as_rotvec()
                    
                    new_target_pose_rv[:3] += kb_dpos
                    new_target_pose_rv[3:] = new_rotvec
                
                # Determine if action was taken
                action_taken = False
                if np.allclose(dpos, 0.0) and np.allclose(drot_xyz, 0.0):
                    if target_pose_last_action_rv is None:
                        translation, quat_xyzw = self.env.get_ee_pose()
                        translation, quat_xyzw = (
                            translation.cpu().numpy().squeeze(),
                            quat_xyzw.cpu().numpy().squeeze(),
                        )
                        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                        target_pose_last_action_rv = np.array([*translation, *rotvec])
                else:
                    action_taken = True
                    target_pose_last_action_rv = None
                
                if not np.allclose(keyboard_action[:6], 0.0):
                    action_taken = True
                    target_pose_last_action_rv = None
            
                # Convert to action
                # Use current target_pose_rv as current pose, and new_target_pose_rv or target_pose_last_action_rv as goal
                current_pose_mat = pose_rv2mat(target_pose_rv)
                if target_pose_last_action_rv is not None:
                    goal_pose_mat = pose_rv2mat(target_pose_last_action_rv)
                else:
                    goal_pose_mat = pose_rv2mat(new_target_pose_rv)
                
                # Update target_pose_rv after computing action
                target_pose_rv = new_target_pose_rv
                
                action = to_isaac_dpose_from_abs(
                    current_pose_mat=current_pose_mat,
                    goal_pose_mat=goal_pose_mat,
                    grasp_flag=grasp_flag,
                    device=self.env.device,
                    rm=True,
                )
                
                # Apply keyboard action if present (overrides SpaceMouse)
                if not np.allclose(keyboard_action[:6], 0.0):
                    action[0, :7] = (
                        torch.from_numpy(keyboard_action[:7])
                        .float()
                        .to(action.device)
                    )
                    action_taken = True
                    target_pose_last_action_rv = None
            
                # Scale action
                action = scale_scripted_action(
                    action.detach().cpu().clone(),
                    pos_bounds_m=self.pos_bounds_m,
                    ori_bounds_deg=self.ori_bounds_deg,
                    device=self.env.device,
                )
                
                # Execute action
                next_obs, rew, done, info = self.env.step(action)
                
                # Store transition
                if self.robot_settled and self.recording and action_taken:
                    if info.get("action_success", True):
                        print(f"[Recovery Teleoperation] Storing action {action}")
                        self.store_transition(obs, action, rew, skill_complete=0)
                
                obs = next_obs
                
                # Update target pose from current pose
                translation, quat_xyzw = self.env.get_ee_pose()
                translation, quat_xyzw = (
                    translation.cpu().numpy().squeeze(),
                    quat_xyzw.cpu().numpy().squeeze(),
                )
                rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                target_pose_rv = np.array([*translation, *rotvec])
                
                # Wait for cycle end
                precise_wait(t_cycle_end)
                iter_idx += 1
            
            # Cleanup SpaceMouse
            if sm is not None:
                try:
                    sm.__exit__(None, None, None)
                except Exception:
                    pass
        
        return None


def main():
    parser = argparse.ArgumentParser(description="Recovery teleoperation data collection")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing rollout trajectories",
    )
    parser.add_argument(
        "--furniture",
        help="Name of the furniture",
        choices=list(config["furniture"].keys()),
        required=True,
    )
    parser.add_argument("--randomness", default="low", choices=["low", "med", "high"])
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument(
        "--ctrl-mode",
        type=str,
        help="Type of low level controller to use.",
        choices=["osc", "diffik"],
        default="diffik",
    )
    parser.add_argument(
        "--draw-marker",
        action="store_true",
        help="If set, will draw an AprilTag marker on the furniture",
    )
    parser.add_argument(
        "--no-ee-laser",
        action="store_false",
        help="If set, will not show the laser coming from the end effector",
        dest="ee_laser",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reverse/data_collection/data/recovery",
        help="Output directory for recovery trajectories",
    )
    
    args = parser.parse_args()
    
    if not args.draw_marker:
        turn_off_april_tags()
    
    # Setup keyboard interface
    keyboard_device_interface = RecoveryKeyboardInterface()
    keyboard_device_interface.print_usage()
    
    # Ensure valid randomness
    randomness = Randomness.str_to_enum(args.randomness)
    
    # Create recovery environment
    print("Creating recovery environment...")
    env = FurnitureRecoverySimEnv(
        furniture=args.furniture,
        obs_keys=FULL_OBS,
        headless=False,
        max_env_steps=3_000,
        num_envs=1,
        act_rot_repr="quat",
        action_type="delta",
        manual_done=True,
        resize_img=False,
        np_step_out=False,
        channel_first=False,
        randomness=randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser,
    )
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create collector
    collector = RecoveryTeleopCollector(
        env=env,
        device_interface=keyboard_device_interface,
        output_dir=output_dir,
        furniture=args.furniture,
        ctrl_mode=args.ctrl_mode,
    )
    
    # Load trajectories
    trajectory_dir = Path(args.trajectory_dir)
    trajectory_paths = sorted(trajectory_dir.glob("*.pkl"))
    
    if len(trajectory_paths) == 0:
        print(f"No trajectory files found in {trajectory_dir}")
        return
    
    print(f"Found {len(trajectory_paths)} trajectories")
    
    # Iterate through trajectories
    for traj_idx, trajectory_path in enumerate(trajectory_paths):
        print(f"\nTrajectory {traj_idx + 1}/{len(trajectory_paths)}: {trajectory_path.name}")
        
        try:
            result = collector.collect_recovery_trajectory(trajectory_path)
            
            if result is None:
                # User terminated
                print("Program terminated by user")
                break
            elif result:
                # Saved successfully
                print(f"Saved recovery trajectory for {trajectory_path.name}")
            else:
                # Discarded
                print(f"Discarded trajectory {trajectory_path.name}, moving to next")
        except Exception as e:
            print(f"Error processing trajectory {trajectory_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nRecovery teleoperation complete!")
    keyboard_device_interface.close()


if __name__ == "__main__":
    main()

