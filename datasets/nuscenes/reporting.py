'''
Taken and adapted from https://github.com/aleksandrkim61/EagerMOT
Check out the corresponding Paper https://arxiv.org/abs/2104.14682
This is serves as inspiration for our own code
'''
import os
import ujson as json
import time
from typing import IO, Any, Dict, Iterable, List

import numpy as np
from pyquaternion import Quaternion

# from objects.fused_instance import FusedInstance

from datasets.nuscenes.classes import name_from_id
# from transform.nuscenes import convert_kitti_bbox_coordinates_to_nu

'''
    FROM: https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any
    We define a standardized tracking result format that serves as an input to the evaluation code.
    Results are evaluated for each 2Hz keyframe, also known as sample.
    The tracking results for a particular evaluation set (train/val/test) are 
        stored in a single JSON file. For the train and val sets the evaluation can be 
        performed by the user on their local machine. 
    For the test set the user needs to zip the single JSON result file and submit it to 
        the official evaluation server (see above).
    The JSON file includes meta data meta on the type of inputs used for this method.
    Furthermore it includes a dictionary results that maps each sample_token to 
        a list of sample_result entries.
    Each sample_token from the current evaluation set must be included in results, although the list of predictions
        may be empty if no object is tracked.
'''
def build_results_dict(frame_token: str,
                            translation: List[float],
                            size : List[float],
                            rotation: List[float],
                            velocity : List[float],
                            tracking_id : str,
                            tracking_name : str,
                            tracking_score : float
                            ) -> Dict[str, Any]:

    track_dict: Dict[str, Any] = {"sample_token": frame_token}
    track_dict["translation"] = translation
    track_dict["size"] = size
    track_dict["rotation"] = rotation
    track_dict["velocity"] = velocity
    track_dict["tracking_id"] = tracking_id
    track_dict["tracking_name"] = tracking_name # name_from_id(instance.class_id)
    track_dict["tracking_score"] = tracking_score # confidence 

    return track_dict


def add_results_to_submit( submission: Dict[str, Dict[str, Any]], 
                            frame_token: str,
                            predicted_instance_dicts: Iterable[Dict[str, Any]] ) -> None:
    assert frame_token not in submission["results"], submission["results"][frame_token]
    submission["results"][frame_token] = []

    for instance_dict in predicted_instance_dicts:
        submission["results"][frame_token].append(instance_dict)

    if len(submission["results"][frame_token]) == 0:
        print(f"Nothing tracked for {frame_token}")


def save_to_json_file(submission: Dict[str, Dict[str, Any]],
                             folder_name: str, version: str) -> None:
    print(f"Frames tracked: {len(submission['results'].keys())}")
    results_file = os.path.join(folder_name, (version + "_tracking.json"))
    with open(results_file, 'w') as f:
        json.dump(submission, f, indent=4)
