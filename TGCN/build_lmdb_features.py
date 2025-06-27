import lmdb
import torch
import os
import json
import tqdm
import argparse
from multiprocessing import Pool

# Import necessary functions/classes from existing files
from configs import create_arg_parser, get_config_from_args # Keep these
from pose_utils import extract_pose_features_from_content # Changed import
import shutil # For removing existing LMDB

def prepare_instance_data(args):
    # This function now *prepares* the data for a single instance,
    # but does not write to LMDB directly. It returns the key and the processed tensor.
    instance_entry, pose_root = args # Removed lmdb_path and features_dir from args
    video_id, gloss_cat, frame_start, frame_end = instance_entry

    video_json_path = os.path.join(pose_root, f"{video_id}.json")
    
    try:
        with open(video_json_path, 'r') as f:
            video_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Video JSON file not found for {video_id} at {video_json_path}. Skipping.")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Malformed JSON for {video_id} at {video_json_path}. Skipping.")
        return None

    pose_seq = []
    # Create a mapping for quick lookup of frame data
    frame_data_map = {entry["frame_index"]: entry["data"] for entry in video_data}

    for frame_idx in range(frame_start, frame_end + 1):
        frame_content = frame_data_map.get(frame_idx)
        if frame_content:
            try:
                # The 'data' field contains the original OpenPose JSON content
                person_data = frame_content["people"][0]
                xy = extract_pose_features_from_content(person_data)
            except (KeyError, IndexError):
                xy = None # Handle cases where "people" or "people[0]" is missing
        else:
            xy = None # Frame index not found in the merged JSON

        if xy is not None:
            pose_seq.append(xy)
        elif pose_seq:
            pose_seq.append(pose_seq[-1])
        else:
            pose_seq.append(torch.zeros((55, 2)))

    if not pose_seq: # After trying to collect all frames, if still empty
        print(f"Warning: No valid poses extracted for {video_id}, frames {frame_start}-{frame_end}. Skipping LMDB entry.")
        return

    total_frames = len(pose_seq)
    target_frames = 50

    if total_frames >= target_frames:
        # Randomly select a consecutive 50-frame segment
        start_idx = torch.randint(0, total_frames - target_frames + 1, (1,)).item()
        pose_seq = pose_seq[start_idx:start_idx + target_frames]
    else:
        # Pad the sequence with the last frame to reach 50
        last_frame = pose_seq[-1]
        pose_seq += [last_frame] * (target_frames - total_frames)

    # Convert to tensor shape: (50, 55, 2)
    pose_tensor = torch.stack(pose_seq)

    key = f"{video_id}_{frame_start}_{frame_end}".encode("ascii") # Key for LMDB
    
    return key, pose_tensor.numpy().astype('float32').tobytes()


def build_lmdb_database(configs, args):
    lmdb_path = configs.lmdb_path
    pose_data_root = configs.pose_data_root
    # features_dir = configs.features_dir # Not used in this function anymore

    # Ensure LMDB directory exists
    # lmdb.open will create the LMDB file/directory itself, but its parent directory must exist.
    lmdb_dir = os.path.dirname(lmdb_path) if os.path.basename(lmdb_path) else lmdb_path
    os.makedirs(lmdb_dir, exist_ok=True)

    # Clear existing LMDB if it exists, to ensure a fresh build
    if os.path.exists(lmdb_path):
        print(f"Removing existing LMDB at {lmdb_path}...")
        # LMDB is a directory, so remove it recursively
        shutil.rmtree(lmdb_path)
        print("Existing LMDB removed.")

    index_file_path = os.path.join(configs.splits_dir, f'{args.subset}.json')

    with open(index_file_path, 'r') as f:
        content = json.load(f)

    # Prepare a list of all instances to process
    all_instance_args = []
    for gloss_entry in content:
        # gloss = gloss_entry['gloss'] # Not directly used in prepare_instance_data args
        for instance in gloss_entry['instances']:
            # Only process instances from the 'train', 'val', 'test' splits
            if instance['split'] in ['train', 'val', 'test']:
                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']
                # The gloss is part of the instance_entry tuple, but not directly used in prepare_instance_data
                # It's used in LMDBSignDataset to create the key.
                # The key is video_id_frame_start_frame_end, so gloss is not needed here.
                all_instance_args.append(( (video_id, None, frame_start, frame_end), # gloss_cat is not needed here
                                           pose_data_root ))

    print(f"Total instances to prepare: {len(all_instance_args)}")

    # Use multiprocessing to prepare data
    prepared_data = []
    with Pool(processes=args.num_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(prepare_instance_data, all_instance_args), total=len(all_instance_args)):
            if result is not None:
                prepared_data.append(result)

    print(f"Prepared {len(prepared_data)} instances. Starting LMDB write.")

    # Now, write all prepared data to LMDB in a single process
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=1) # max_readers=1 for single writer
    with env.begin(write=True) as txn:
        for key, value in tqdm.tqdm(prepared_data, desc="Writing to LMDB"):
            txn.put(key, value)
    env.close()

    print(f"LMDB database built at: {lmdb_path}")


if __name__ == '__main__':
    parser = create_arg_parser()
    parser.add_argument('--num_processes', type=int, default=os.cpu_count(),
                        help='Number of processes for multiprocessing LMDB build')
    args = parser.parse_args()

    configs = get_config_from_args(args) 

    print(f"Building LMDB for subset: {args.subset}")
    build_lmdb_database(configs, args)