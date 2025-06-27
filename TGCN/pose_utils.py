import json
import os
import torch


def extract_pose_features_from_content(pose_content):
    """
    Extracts pose features (XY coordinates) from the 'data' content of a merged JSON entry.
    This version is independent of file paths and directly processes the dictionary content.
    """
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

    try:
        # The 'pose_content' here is already the dictionary that was previously
        # json.load(open(...))["people"][0]
        # So we directly access its keys.
        body_pose = pose_content["pose_keypoints_2d"]
        left_hand_pose = pose_content["hand_left_keypoints_2d"]
        right_hand_pose = pose_content["hand_right_keypoints_2d"]
    except (KeyError, IndexError): # Handle cases where "people" or "people[0]" is missing or keypoints are missing
        # Return None if keypoints are missing or structure is unexpected
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