import pickle
from pathlib import Path
import argparse

def fix_recovery_trajectory(pickle_path):
    """Remove the first None action from a recovery trajectory."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    if len(data["actions"]) > 0 and data["actions"][0] is None:
        print(f"Fixing {pickle_path}: Removing first None action")
        data["observations"] = data["observations"][1:]
        data["actions"] = data["actions"][1:]
        data["rewards"] = data["rewards"][1:] if len(data["rewards"]) > 0 else []
        data["skills"] = data["skills"][1:] if len(data["skills"]) > 0 else []
        
        # Save back
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Fixed: {len(data['actions'])} actions remaining")
    else:
        print(f"No fix needed for {pickle_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-dir", type=str, required=True)
    args = parser.parse_args()
    # Apply to all recovery trajectories
    recovery_dir = Path(args.recovery_dir)
    for pickle_file in recovery_dir.rglob("*.pkl"):
        fix_recovery_trajectory(pickle_file)