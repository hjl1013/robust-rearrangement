from datetime import datetime
from pathlib import Path
import zarr
import numpy as np
from collections import defaultdict
from tqdm import tqdm, trange
import argparse
from numcodecs import Blosc, JSON

def combine_datasets(data_paths, output_path, chunksize=1000):
    """
    Combine multiple zarr datasets into a single dataset with the same format as process_pickles.py
    
    Args:
        data_paths: List of paths to zarr datasets to combine
        output_path: Path to save the combined dataset
        chunksize: Chunk size for writing data (default: 1000)
    """
    last_episode_end = 0

    # Initialize data structure - collect all possible keys from all datasets
    new_data = {}
    all_keys = set()
    
    # First pass: collect all keys from all datasets
    for path in data_paths:
        dataset = zarr.open(path, "r")
        for key in dataset.keys():
            all_keys.add(key)
    
    # Initialize data structure for all keys
    for key in all_keys:
        if key in ["action/delta", "action/pos"]:
            new_data[key] = []
        elif key == "action":
            # Handle legacy format with action group
            if "action/delta" not in new_data:
                new_data["action/delta"] = []
            if "action/pos" not in new_data:
                new_data["action/pos"] = []
        else:
            new_data[key] = []

    # Collect data from all datasets
    for path in tqdm(data_paths, desc="Combining datasets"):
        dataset = zarr.open(path, "r")
        dataset_keys = list(dataset.keys())
        for key in tqdm(dataset_keys, leave=False, desc=f"Processing {Path(path).name}"):
            if key == "episode_ends":
                # Increment episode_ends values and append
                incremented_ends = dataset[key][:] + last_episode_end
                new_data[key].extend(incremented_ends.tolist())
                last_episode_end = incremented_ends[-1]
            elif key == "action/delta":
                # Handle action/delta directly
                new_data["action/delta"].append(dataset[key][:])
            elif key == "action/pos":
                # Handle action/pos directly
                new_data["action/pos"].append(dataset[key][:])
            elif key == "action":
                # Handle legacy format with action group
                if "delta" in dataset[key]:
                    new_data["action/delta"].append(dataset[key]["delta"][:])
                if "pos" in dataset[key]:
                    new_data["action/pos"].append(dataset[key]["pos"][:])
            else:
                # Simply append for other keys
                if key in new_data:
                    data = dataset[key][:]
                    if isinstance(data, np.ndarray):
                        new_data[key].append(data)
                    else:
                        # Handle object arrays (like strings)
                        new_data[key].extend(data.tolist() if hasattr(data, 'tolist') else list(data))

    # Concatenate arrays
    print("Concatenating arrays...")
    concatenated_data = {}
    for key in tqdm(new_data.keys(), desc="Concatenating"):
        if len(new_data[key]) == 0:
            continue
        
        # Check if first element is an array (meaning we need to concatenate)
        # or a scalar/list (meaning we can use np.array directly)
        first_item = new_data[key][0]
        is_list_of_arrays = isinstance(first_item, np.ndarray)
        
        if key == "episode_ends":
            # episode_ends is always a flat list of integers (from extend())
            concatenated_data[key] = np.array(new_data[key], dtype=np.uint32)
        elif key in ["task", "pickle_file"]:
            # String arrays - handle both cases
            if is_list_of_arrays:
                # List of arrays - flatten first, then create object array
                flattened = []
                for item in new_data[key]:
                    if isinstance(item, np.ndarray):
                        flattened.extend(item.tolist())
                    else:
                        flattened.append(item)
                concatenated_data[key] = np.array(flattened, dtype=object)
            else:
                # Flat list
                concatenated_data[key] = np.array(new_data[key], dtype=object)
        elif is_list_of_arrays:
            # All other arrays that are lists of arrays - concatenate
            concatenated_data[key] = np.concatenate(new_data[key], axis=0)
            # Ensure correct dtype for specific keys
            if key == "success":
                concatenated_data[key] = concatenated_data[key].astype(np.uint8)
        else:
            # Flat list - convert to array (shouldn't happen for most keys, but handle it)
            if key in ["action/delta", "action/pos"]:
                # These should always be arrays, but handle edge case
                concatenated_data[key] = np.array(new_data[key])
            else:
                concatenated_data[key] = np.array(new_data[key])

    # Create output zarr store with same format as process_pickles.py
    print("Creating output zarr store...")
    out_zarr = zarr.open(str(output_path), mode="w")
    out_zarr.attrs["time_created"] = datetime.now().astimezone().isoformat()

    # Define compressor (same as process_pickles.py)
    compressor = Blosc(cname="lz4", clevel=5)

    # Create datasets with proper compression and chunking
    for key, data in tqdm(concatenated_data.items(), desc="Creating datasets"):
        if "color_image" in key:
            # Apply compression to image data
            out_zarr.create_dataset(
                key,
                data=data,
                chunks=(chunksize,) + data.shape[1:],
                compressor=compressor,
            )
        elif data.dtype == object:
            # Object arrays (strings)
            out_zarr.create_dataset(
                key,
                data=data,
                chunks=data.shape,
                object_codec=JSON(),
            )
        else:
            # Other numeric arrays
            out_zarr.create_dataset(
                key,
                data=data,
                chunks=data.shape,
            )

    # Set final metadata (same format as process_pickles.py)
    out_zarr.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    out_zarr.attrs["chunksize"] = chunksize
    out_zarr.attrs["rotation_mode"] = "rot_6d"
    out_zarr.attrs["n_episodes"] = len(out_zarr["episode_ends"])
    out_zarr.attrs["n_timesteps"] = len(out_zarr["action/delta"])
    out_zarr.attrs["mean_episode_length"] = round(
        len(out_zarr["action/delta"]) / len(out_zarr["episode_ends"])
    )
    out_zarr.attrs["calculated_pos_action_from_delta"] = True
    
    # Copy compatible metadata from first dataset (excluding computed values)
    print("Copying metadata...")
    exclude_attrs = {
        "n_episodes", "n_timesteps", "mean_episode_length", 
        "time_created", "time_finished", "chunksize"
    }
    if len(data_paths) > 0:
        first_dataset = zarr.open(data_paths[0], "r")
        for attr_key, attr_value in first_dataset.attrs.items():
            if attr_key not in exclude_attrs:
                try:
                    out_zarr.attrs[attr_key] = attr_value
                except Exception as e:
                    print(f"Warning: Could not copy attribute {attr_key}: {e}")

    print(f"Combined dataset saved to {output_path}")
    print(f"Total episodes: {out_zarr.attrs['n_episodes']}")
    print(f"Total timesteps: {out_zarr.attrs['n_timesteps']}")
    print(f"Mean episode length: {out_zarr.attrs['mean_episode_length']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple zarr datasets into one")
    parser.add_argument("--data-paths", type=str, required=True, nargs="+",
                        help="Paths to zarr datasets to combine")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the combined dataset")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Chunk size for writing data (default: 1000)")
    args = parser.parse_args()
    combine_datasets(args.data_paths, args.output_path, chunksize=args.chunksize)