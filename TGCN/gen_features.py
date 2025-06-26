import json
import os
import time
import argparse
from multiprocessing import Pool
import torch
from configs import Config, create_arg_parser


def compute_difference(x):
    diff = []

    for i, xx in enumerate(x):
        temp = []
        for j, xxx in enumerate(x):
            if i != j:
                temp.append(xx - xxx)

        diff.append(temp)

    return diff


def gen(entry_list, pose_data_root, features_dir, body_pose_exclude):
    for i, entry in enumerate(entry_list):
        for instance in entry['instances']:
            vid = instance['video_id']

            frame_start = instance['frame_start']
            frame_end = instance['frame_end']

            save_to = os.path.join(features_dir, vid)

            for frame_id in range(frame_start, frame_end + 1):
                frame_id = 'image_{}'.format(str(frame_id).zfill(5))

                ft_path = os.path.join(save_to, frame_id + '_ft.pt')
                if not os.path.exists(ft_path):
                    try:
                        pose_content = json.load(open(os.path.join(pose_data_root,
                                                                   vid, frame_id + '_keypoints.json')))["people"][0]
                    except IndexError:
                        continue

                    body_pose = pose_content["pose_keypoints_2d"]
                    left_hand_pose = pose_content["hand_left_keypoints_2d"]
                    right_hand_pose = pose_content["hand_right_keypoints_2d"]

                    body_pose.extend(left_hand_pose)
                    body_pose.extend(right_hand_pose)

                    x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
                    y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]

                    x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
                    y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)

                    x_diff = torch.FloatTensor(compute_difference(x)) / 2
                    y_diff = torch.FloatTensor(compute_difference(y)) / 2

                    zero_indices = (x_diff == 0).nonzero()
                    orient = y_diff / x_diff
                    orient[zero_indices] = 0

                    xy = torch.stack([x, y]).transpose_(0, 1)
                    ft = torch.cat([xy, x_diff, y_diff, orient], dim=1)

                    torch.save(ft, ft_path)

        print('Finish {}-th entry'.format(i))


if __name__ == '__main__':
    parser = create_arg_parser()
    parser.add_argument('--num_processes', type=int, default=3,
                        help='Number of processes for multiprocessing')
    args = parser.parse_args()
    
    # Use subset-specific config if available, otherwise use default
    if os.path.exists(os.path.join('configs', f'{args.subset}.ini')):
        config_file = os.path.join('configs', f'{args.subset}.ini')
    else:
        config_file = args.config
        
    configs = Config(config_file)
    
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
    index_file_path = os.path.join(configs.splits_dir, f'{args.subset}.json')

    with open(index_file_path, 'r') as f:
        content = json.load(f)

    # Before Pool: create all necessary directories to avoid race conditions
    print("Pre-creating all feature directories...")
    for entry in content:
        for instance in entry['instances']:
            vid = instance['video_id']
            save_to = os.path.join(configs.features_dir, vid)
            os.makedirs(save_to, exist_ok=True)

    start_time = time.time()
    
    # Split entries for multiprocessing
    total_entries = len(content)
    entries_per_process = total_entries // args.num_processes
    
    entry_splits = []
    for i in range(args.num_processes):
        start_idx = i * entries_per_process
        if i == args.num_processes - 1:  # Last process takes remaining entries
            end_idx = total_entries
        else:
            end_idx = (i + 1) * entries_per_process
        entry_splits.append(content[start_idx:end_idx])

    # Create worker arguments
    worker_args = [(split, configs.pose_data_root, configs.features_dir, body_pose_exclude) 
                   for split in entry_splits]

    p = Pool(args.num_processes)
    print("Starting feature generation with {} processes...".format(args.num_processes))
    p.starmap(gen, worker_args)
    p.close()
    p.join()
    
    print("Feature generation completed in {:.2f} seconds".format(time.time() - start_time))
