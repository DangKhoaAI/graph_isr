import json
import math
import os
import random

import numpy as np
import lmdb

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import utils # Assuming utils.py contains labels2cat
from sampling_utils import rand_start_sampling, sequential_sampling, k_copies_fixed_length_sequential_sampling # Reuse sampling functions


class LMDBSignDataset(Dataset):
    def __init__(self, index_file_path, split, lmdb_path, sample_strategy='rnd_start', num_samples=25, num_copies=4, return_video_id=False):
        assert os.path.exists(index_file_path), f"Non-existent indexing file path: {index_file_path}"
        assert os.path.exists(lmdb_path), f"LMDB path does not exist: {lmdb_path}"

        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        if isinstance(split, str):
            split = [split]

        self._make_dataset(index_file_path, split)

        self.lmdb_path = lmdb_path
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples
        self.num_copies = num_copies
        self.return_video_id = return_video_id

        # Open LMDB environment in __init__
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin() # Keep a transaction open for faster reads in __getitem__

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end = self.data[index]

        # Construct the key used during LMDB build
        key = f"{video_id}_{frame_start}_{frame_end}".encode("ascii")

        # Retrieve data from LMDB
        data_bytes = self.txn.get(key)
        if data_bytes is None:
            # This should ideally not happen if LMDB is built correctly
            # Fallback: return a zero tensor or handle error
            print(f"Warning: Key {key.decode('ascii')} not found in LMDB. Returning zero tensor.")
            x = torch.zeros((55, self.num_samples * 2))
            if self.return_video_id:
                return x, gloss_cat, video_id
            else:
                return x, gloss_cat

        # Deserialize the numpy array and convert to torch tensor
        # The stored data is (num_frames_in_instance, 55, 2)
        full_pose_sequence = torch.from_numpy( np.frombuffer(data_bytes, dtype=np.float32).copy().reshape(-1, 55, 2))
        num_frames_in_instance = full_pose_sequence.shape[0]

        # Apply sampling strategy
        if self.sample_strategy == 'rnd_start':
            frames_to_sample_indices = rand_start_sampling(0, num_frames_in_instance - 1, self.num_samples)
        elif self.sample_strategy == 'seq':
            frames_to_sample_indices = sequential_sampling(0, num_frames_in_instance - 1, self.num_samples)
        elif self.sample_strategy == 'k_copies':
            frames_to_sample_indices = k_copies_fixed_length_sequential_sampling(0, num_frames_in_instance - 1, self.num_samples, self.num_copies)
        else:
            raise NotImplementedError(f'Unimplemented sample strategy found: {self.sample_strategy}.')

        # Select frames based on sampled indices
        sampled_poses = []
        for idx in frames_to_sample_indices:
            if idx < num_frames_in_instance:
                sampled_poses.append(full_pose_sequence[idx])
            else:
                # If index is out of bounds (due to padding in sampling functions), repeat the last frame
                sampled_poses.append(full_pose_sequence[-1])

        # Stack the sampled poses
        x = torch.stack(sampled_poses) # Shape: (num_samples, 55, 2)

        # Reshape for the model: (num_samples, 55, 2) -> (55, num_samples * 2)
        # The model expects (batch_size, num_keypoints, num_samples * 2)
        # So, for a single item, it should be (55, num_samples * 2)
        x = x.permute(1, 0, 2).reshape(55, -1) # (55, num_samples * 2)

        y = gloss_cat

        if self.return_video_id:
            return x, y, video_id
        else:
            return x, y

    def _make_dataset(self, index_file_path, split):
        with open(index_file_path, 'r') as f:
            content = json.load(f)

        # create label encoder
        glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])

        self.label_encoder.fit(glosses)
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))

        # make dataset
        for gloss_entry in content:
            gloss, instances = gloss_entry['gloss'], gloss_entry['instances']
            gloss_cat = utils.labels2cat(self.label_encoder, [gloss])[0]

            for instance in instances:
                if instance['split'] not in split:
                    continue

                frame_end = instance['frame_end']
                frame_start = instance['frame_start']
                video_id = instance['video_id']

                instance_entry = video_id, gloss_cat, frame_start, frame_end
                self.data.append(instance_entry)

    def __del__(self):
        if hasattr(self, 'env') and self.env:
            self.env.close()