import os
from typing import Any, Dict, List
import numpy as np

import ujson as json
from nuscenes.nuscenes import Box
import datasets.nuscenes.classes as nu_classes
from pyquaternion import Quaternion

def load_detections_nuscenes_detection_submission_file(path_to_detection_file) -> Dict[str, List[Box]]:
    '''
    Taken and adapted from https://github.com/aleksandrkim61/EagerMOT
    Check out the corresponding Paper https://arxiv.org/abs/2104.14682
    This is serves as inspiration for our own code
    '''
    filepath = os.path.join(path_to_detection_file)
    
    print(f"Parsing {filepath}")
    with open(filepath, 'r') as f:
        full_results_json = json.load(f)

    all_detections = full_results_json["results"]
    all_frames_to_bboxes: Dict[str, List[Box]] = {}
    for frame_token, frame_dets in all_detections.items():
        assert frame_token not in all_frames_to_bboxes
        # all_frames_to_bboxes[frame_token] = [Bbox3d.from_nu_det(det) for det in frame_dets
        #                                      if det["detection_name"] in nu_classes.ALL_NUSCENES_CLASS_NAMES]
        all_frames_to_bboxes[frame_token] = [build_box_from_detection_submission_file(det) for det in frame_dets
                                             if det["detection_name"] in nu_classes.ALL_NUSCENES_CLASS_NAMES]
    return all_frames_to_bboxes

def build_box_from_detection_submission_file(sample_result: Dict[str,Any]):
    """
    Build box from the detection dictionary file.
    sample_result {
    "sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":        <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
    "size":               <float> [3]   -- Estimated bounding box size in m: width, length, height.
    "rotation":           <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":           <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
    "detection_score":    <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
    "attribute_name":     <str>         -- Name of the predicted attribute or empty string for classes without attributes.
                                           See table below for valid attributes for each class, e.g. cycle.with_rider.
                                           Attributes are ignored for classes without attributes.
                                           There are a few cases (0.4%) where attributes are missing also for classes
                                           that should have them. We ignore the predicted attributes for these cases.
    }
    """
    center : List[float] = sample_result["translation"]
    size : List[float] = sample_result["size"]
    rotation: List[float] = sample_result["rotation"]
    orientation:Quaternion = Quaternion(rotation)
    # Input 2d velocity vector 
    velocity: List[float] = sample_result["velocity"]
    if velocity: # check if it is not empty
        # Add additional velocity in z-direction 
        velocity.append(0.0)
        velocity = tuple(velocity)
    else:
        velocity = (np.nan, np.nan, np.nan)
    detection_name:str = sample_result["detection_name"]
    label = nu_classes.id_from_name(detection_name)
    detection_score:float = sample_result["detection_score"]
    attribute_name:str = sample_result["attribute_name"]

    box = Box( center,
                size= size,
                orientation= orientation,
                label= label,
                score= detection_score,
                velocity=velocity,
                name=attribute_name,
                token= None)
    return box
