import argparse
from pathlib import Path
import pickle

from forward_rollout import visualize_rollout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    
    rollout_dir = Path(args.data_dir) / 'rollouts'
    rollout_files = sorted(rollout_dir.glob("*.pkl"))

    visualizations_dir = Path(args.data_dir) / 'visualizations'
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(rollout_files)} visualizations...")
    for rollout_file in rollout_files:
        print(f"Visualizing {rollout_file}...")
        with open(rollout_file, "rb") as f:
            data = pickle.load(f)
            observations = data["observations"]
            actions = data["actions"]
            rewards = data["rewards"]
            out_of_distribution_scores = data["out_of_distribution_scores"]
            in_distribution_index = data["in_distribution_index"]
            out_of_distribution_index = data["out_of_distribution_index"]
            threshold = data["threshold"]
            consecutive_steps = data["consecutive_steps"]
            timestep_diff = data["timestep_diff"]

        filename = rollout_file.stem

        video_path = visualizations_dir / f"{filename}.mp4"
        visualize_rollout(
            observations,
            out_of_distribution_scores,
            in_distribution_index,
            out_of_distribution_index,
            threshold,
            consecutive_steps,
            video_path,
            fps=args.fps,
        )

    print(f"\n{'='*60}")
    print(f"Done! Completed {len(rollout_files)} rollouts")
    print(f"Visualizations saved to: {visualizations_dir}")
    print(f"{'='*60}")