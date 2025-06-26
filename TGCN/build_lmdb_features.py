import lmdb
import torch
import os
import json
import tqdm
import argparse
from multiprocessing import Pool

# Import necessary functions/classes from existing files
from configs import Config, create_arg_parser
from sign_dataset import read_pose_file

def build_lmdb_for_instance(args):
    """Helper function to process a single instance for multiprocessing."""
    instance_entry, pose_root, features_dir, lmdb_path = args
    video_id, gloss_cat, frame_start, frame_end = instance_entry

    # Open LMDB environment in each process (important for multiprocessing)
    # map_size should be large enough to hold the entire dataset
    # max_readers is important for multiprocessing writes
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=126)
    txn = env.begin(write=True)

    pose_seq = []
    # Iterate through all frames for the instance
    for i in range(frame_start, frame_end + 1):
        pose_path = os.path.join(pose_root, video_id, f"image_{str(i).zfill(5)}_keypoints.json")
        # read_pose_file handles .pt caching and on-the-fly feature extraction
        xy = read_pose_file(pose_path, features_dir)
        if xy is not None:
            pose_seq.append(xy)
        elif pose_seq: # If previous frames exist, use the last valid pose
            pose_seq.append(pose_seq[-1])
        else: # If no valid pose yet, append a zero tensor (should be rare for start of video)
            # Assuming 55 keypoints, 2 coordinates (x, y)
            pose_seq.append(torch.zeros((55, 2)))

    if not pose_seq: # Handle cases where no valid frames were found for an instance
        print(f"Warning: No valid poses found for video_id: {video_id}, frames {frame_start}-{frame_end}")
        txn.abort() # Abort transaction if no data to write
        env.close()
        return

    # Stack all collected poses for the instance
    # The shape will be (num_frames_in_instance, 55, 2)
    pose_tensor = torch.stack(pose_seq)

    # Store the full sequence in LMDB
    # Key should uniquely identify the instance
    key = f"{video_id}_{frame_start}_{frame_end}".encode("ascii")
    txn.put(key, pose_tensor.numpy().tobytes())

    txn.commit()
    env.close()


def build_lmdb_database(configs, args):
    lmdb_path = configs.lmdb_path
    pose_data_root = configs.pose_data_root
    features_dir = configs.features_dir

    os.makedirs(lmdb_path, exist_ok=True)

    index_file_path = os.path.join(configs.splits_dir, f'{args.subset}.json')

    with open(index_file_path, 'r') as f:
        content = json.load(f)

    # Prepare a list of all instances to process
    all_instances = []
    for gloss_entry in content:
        gloss = gloss_entry['gloss']
        for instance in gloss_entry['instances']:
            # Only process instances from the 'train', 'val', 'test' splits
            if instance['split'] in ['train', 'val', 'test']:
                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']
                all_instances.append(( (video_id, gloss, frame_start, frame_end),
                                       pose_data_root, features_dir, lmdb_path ))

    print(f"Total instances to process: {len(all_instances)}")

    with Pool(processes=args.num_processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(build_lmdb_for_instance, all_instances), total=len(all_instances)))

    print(f"LMDB database built at: {lmdb_path}")


if __name__ == '__main__':
    parser = create_arg_parser()
    parser.add_argument('--num_processes', type=int, default=os.cpu_count(),
                        help='Number of processes for multiprocessing LMDB build')
    args = parser.parse_args()

    # Use subset-specific config if available, otherwise use default
    if os.path.exists(os.path.join('configs', f'{args.subset}.ini')):
        config_file = os.path.join('configs', f'{args.subset}.ini')
    else:
        config_file = args.config

    configs = Config(config_file)

    print(f"Building LMDB for subset: {args.subset}")
    build_lmdb_database(configs, args)