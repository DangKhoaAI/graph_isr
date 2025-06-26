import lmdb
import torch
import os
import json
import tqdm
import argparse
from multiprocessing import Pool

# Import necessary functions/classes from existing files
from configs import Config, create_arg_parser
from pose_utils import read_pose_file

def build_lmdb_for_instance(args):
    instance_entry, pose_root, features_dir, lmdb_path = args
    video_id, gloss_cat, frame_start, frame_end = instance_entry

    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=126)
    txn = env.begin(write=True)

    pose_seq = []
    for i in range(frame_start, frame_end + 1):
        pose_path = os.path.join(pose_root, video_id, f"image_{str(i).zfill(5)}_keypoints.json")
        xy = read_pose_file(pose_path)
        if xy is not None:
            pose_seq.append(xy)
        elif pose_seq:
            pose_seq.append(pose_seq[-1])
        else:
            pose_seq.append(torch.zeros((55, 2)))

    if not pose_seq:
        print(f"Warning: No valid poses for {video_id}, frames {frame_start}-{frame_end}")
        txn.abort()
        env.close()
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

    key = f"{video_id}_{frame_start}_{frame_end}".encode("ascii")
    txn.put(key, pose_tensor.numpy().astype('float32').tobytes())

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