import json
import os
import torch


def read_pose_file(filepath):
    """
    Reads a pose JSON file, computes features on-the-fly, and returns only the XY coordinates.
    This version is independent of pre-generated .pt files and does not create them.
    """
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

    try:
        with open(filepath, 'r') as f:
            content = json.load(f)["people"][0]
    except (FileNotFoundError, IndexError, KeyError):
        # Return None if file is not found, or JSON is malformed
        return None

    try:
        body_pose = content["pose_keypoints_2d"]
        left_hand_pose = content["hand_left_keypoints_2d"]
        right_hand_pose = content["hand_right_keypoints_2d"]
    except KeyError:
        # Return None if keypoints are missing
        return None

    body_pose.extend(left_hand_pose)
    body_pose.extend(right_hand_pose)

    x = [v for i, v in enumerate(body_pose) if i % 3 == 0 and i // 3 not in body_pose_exclude]
    y = [v for i, v in enumerate(body_pose) if i % 3 == 1 and i // 3 not in body_pose_exclude]

    # Normalize coordinates
    x = 2 * ((torch.FloatTensor(x) / 256.0) - 0.5)
    y = 2 * ((torch.FloatTensor(y) / 256.0) - 0.5)

    # The rest of the data pipeline (LMDB build, dataset loader, model)
    # is designed to work with only the XY coordinates.
    xy = torch.stack([x, y]).transpose_(0, 1)  # Shape: (55, 2)

    return xy