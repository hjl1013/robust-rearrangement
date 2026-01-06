import argparse
import numpy as np
import scipy.spatial.transform as st
from pathlib import Path
from tqdm import tqdm
import copy

from src.visualization.render_mp4 import unpickle_data, pickle_data


def pose_rv2mat(pose_rv):
    """Convert pose from [x, y, z, rx, ry, rz] (rotvec) to 4x4 matrix."""
    pose_mat = np.eye(4)
    pose_mat[:-1, -1] = pose_rv[:3]
    pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_rv[3:]).as_matrix()
    return pose_mat


def mat2pose_rv(pose_mat):
    """Convert 4x4 matrix to [x, y, z, rx, ry, rz] (rotvec)."""
    pos = pose_mat[:-1, -1]
    rotvec = st.Rotation.from_matrix(pose_mat[:-1, :-1]).as_rotvec()
    return np.concatenate([pos, rotvec])


def to_isaac_dpose_from_abs(current_pose_mat, goal_pose_mat, grasp_flag, rm=True):
    """
    Convert from absolute current and desired pose to delta pose.
    
    Args:
        current_pose_mat: 4x4 matrix of current pose
        goal_pose_mat: 4x4 matrix of goal pose
        grasp_flag: gripper flag (-1 for open, 1 for closed)
        rm (bool): 'rm' stands for 'right multiplication' - If True, assume commands send as right multiply (local rotations)
    
    Returns:
        numpy array of shape (8,) with [dx, dy, dz, dqx, dqy, dqz, dqw, gripper]
    """
    if rm:
        delta_rot_mat = (
            np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
        )
    else:
        delta_rot_mat = goal_pose_mat[:-1, :-1] @ np.linalg.inv(
            current_pose_mat[:-1, :-1]
        )

    dpos = goal_pose_mat[:-1, -1] - current_pose_mat[:-1, -1]
    
    target_rot = st.Rotation.from_matrix(delta_rot_mat)
    target_quat_xyzw = target_rot.as_quat()
    
    # Concatenate: [dx, dy, dz, dqx, dqy, dqz, dqw, gripper]
    target_dpose = np.concatenate([dpos, target_quat_xyzw, np.array([grasp_flag])])
    return target_dpose


def extract_ee_pose_from_obs(obs):
    """
    Extract end-effector pose from observation.
    
    Args:
        obs: Observation dict with 'robot_state' key
        
    Returns:
        tuple: (ee_pos, ee_quat, gripper_width)
            - ee_pos: numpy array of shape (3,)
            - ee_quat: numpy array of shape (4,) in xyzw format
            - gripper_width: float
    """
    robot_state = obs["robot_state"]
    
    if isinstance(robot_state, dict):
        ee_pos = np.array(robot_state["ee_pos"], dtype=np.float32)
        ee_quat = np.array(robot_state["ee_quat"], dtype=np.float32)
        gripper_width = float(robot_state.get("gripper_width", 0.0))
    else:
        # If robot_state is already concatenated, extract based on ROBOT_STATES order
        # ROBOT_STATES = ["ee_pos", "ee_quat", "ee_pos_vel", "ee_ori_vel", "gripper_width"]
        # Dimensions: 3 + 4 + 3 + 3 + 1 = 14
        robot_state = np.array(robot_state, dtype=np.float32)
        if robot_state.ndim > 1:
            robot_state = robot_state.flatten()
        
        ee_pos = robot_state[0:3]
        ee_quat = robot_state[3:7]
        gripper_width = float(robot_state[13]) if len(robot_state) > 14 else 0.0
    
    return ee_pos, ee_quat, gripper_width


def compute_action(obs_t, obs_t_plus_1):
    """
    Compute reversed delta action from two observations.
    
    Original: action[t-1] moves from obs[t-1] to obs[t]
    Reversed: reversed_action[t] should move from obs[t] to obs[t-1]
    
    Args:
        obs_t: Observation at timestep t (current state in reversed trajectory)
        obs_t_plus_1: Observation at timestep t+1 (next state in reversed trajectory)
        
    Returns:
        numpy array of shape (8,) with reversed delta action [dx, dy, dz, dqx, dqy, dqz, dqw, gripper]
    """
    # Extract end-effector poses
    ee_pos_t, ee_quat_t, gripper_width_t = extract_ee_pose_from_obs(obs_t)
    ee_pos_t_plus_1, ee_quat_t_plus_1, gripper_width_t_plus_1 = extract_ee_pose_from_obs(obs_t_plus_1)
    
    # Convert to 4x4 matrices
    pose_mat_t = np.eye(4)
    pose_mat_t[:-1, -1] = ee_pos_t
    pose_mat_t[:-1, :-1] = st.Rotation.from_quat(ee_quat_t).as_matrix()
    
    pose_mat_t_plus_1 = np.eye(4)
    pose_mat_t_plus_1[:-1, -1] = ee_pos_t_plus_1
    pose_mat_t_plus_1[:-1, :-1] = st.Rotation.from_quat(ee_quat_t_plus_1).as_matrix()
    
    # Determine gripper flag from gripper width
    # Use the gripper state from the "current" observation (obs_t) in reversed trajectory
    # Gripper open if width >= 0.05 (threshold from data_collector_sm.py)
    grasp_flag = -1 if gripper_width_t >= 0.05 else 1
    
    # Compute delta action: from obs_t to obs_t_minus_1
    reversed_action = to_isaac_dpose_from_abs(
        current_pose_mat=pose_mat_t,
        goal_pose_mat=pose_mat_t_plus_1,
        grasp_flag=grasp_flag,
        rm=True  # Right multiply as used in data_collector_sm.py
    )
    
    return reversed_action


def reverse_trajectory(data):
    """
    Reverse a trajectory by reversing observations and computing reversed actions.
    
    Args:
        data: Dictionary with trajectory data containing:
            - observations: List of observation dicts
            - actions: List of action arrays (not used, computed from observations)
            - rewards: List of rewards (reversed as dummy data)
            - skills: List of skill flags (reversed as dummy data)
            - success: Boolean (copied as-is)
            - furniture: String (copied as-is)
            - error: Boolean (copied as-is)
            - error_description: String (copied as-is)
            - metadata: Dict (copied as-is)
    
    Returns:
        Dictionary with reversed trajectory data
    """
    observations = data["observations"]
    n_obs = len(observations)
    
    if n_obs < 2:
        raise ValueError(f"Trajectory must have at least 2 observations, got {n_obs}")
    
    # Reverse observations
    reversed_observations = list(reversed(observations))
    
    # Compute reversed actions from observations
    # User instruction: "subtract the state of timestep t from state of timestep t-1 to get the reversed action of timestep t"
    # Interpretation: reversed_action[t] = state[t-1] - state[t] (in terms of delta action computation)
    # This means: state[t] + reversed_action[t] = state[t-1]
    # So reversed_action[t] moves from state[t] to state[t-1] in reversed trajectory
    #
    # Original trajectory: obs[0], obs[1], ..., obs[N]
    # Reversed trajectory: obs_rev[0]=obs[N], obs_rev[1]=obs[N-1], ..., obs_rev[N]=obs[0]
    # For reversed_action[t]: move from obs_rev[t] to obs_rev[t-1]
    reversed_actions = []
    for t in range(n_obs-1):  # Start from t=1 since we need t-1
        # In reversed trajectory:
        # obs_rev[t] is current state (which is obs[N-t] in original)
        # obs_rev[t-1] is previous state (which is obs[N-t+1] in original)
        # reversed_action[t] should move from obs_rev[t] to obs_rev[t-1]
        obs_t = reversed_observations[t]  # Current state at timestep t in reversed trajectory
        obs_t_plus_1 = reversed_observations[t + 1]  # Previous state at timestep t-1 in reversed trajectory
        
        reversed_action = compute_action(obs_t, obs_t_plus_1)
        reversed_actions.append(reversed_action)
    
    # Reverse other keys as dummy data
    reversed_rewards = list(reversed(data.get("rewards", [])))[:-1] if "rewards" in data else []
    reversed_skills = list(reversed(data.get("skills", [])))[:-1] if "skills" in data else []
    
    # Ensure lengths match
    if len(reversed_rewards) != len(reversed_actions):
        reversed_rewards = [0.0] * len(reversed_actions)
    if len(reversed_skills) != len(reversed_actions):
        reversed_skills = [0] * len(reversed_actions)
    
    # Create reversed data dictionary
    reversed_data = {
        "observations": reversed_observations,
        "actions": reversed_actions,
        "rewards": reversed_rewards,
        "skills": reversed_skills,
        "success": data.get("success", False),
        "furniture": data.get("furniture", ""),
        "error": data.get("error", False),
        "error_description": data.get("error_description", ""),
    }
    
    # Copy metadata if present
    if "metadata" in data:
        reversed_data["metadata"] = copy.deepcopy(data["metadata"])
    
    return reversed_data


def main():
    parser = argparse.ArgumentParser(description="Reverse trajectories in pickle files")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input pickle files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save reversed pickle files")
    parser.add_argument("--pattern", type=str, default="*.pkl*",
                        help="File pattern to match (default: *.pkl*)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files (including compressed ones)
    pickle_files = (
        list(input_dir.rglob("*.pkl")) +
        list(input_dir.rglob("*.pkl.xz")) +
        list(input_dir.rglob("*.pkl.gz"))
    )
    
    if len(pickle_files) == 0:
        raise ValueError(f"No pickle files found in {input_dir}")
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Process each pickle file
    for pickle_path in tqdm(pickle_files, desc="Reversing trajectories"):
        try:
            # Load trajectory
            data = unpickle_data(pickle_path)
            
            # Verify required keys
            if "observations" not in data:
                print(f"Warning: {pickle_path} missing 'observations' key, skipping")
                continue
            
            # Reverse trajectory
            reversed_data = reverse_trajectory(data)
            
            # Determine output path (preserve relative structure)
            rel_path = pickle_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save reversed trajectory
            pickle_data(reversed_data, output_path)
            
        except Exception as e:
            print(f"Error processing {pickle_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Done! Reversed trajectories saved to {output_dir}")


if __name__ == "__main__":
    main()
