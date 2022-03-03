from typing import List
from xmlrpc.client import Boolean

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box


def get_box_centers(boxes : List[Box]):
    """
    Returns only the centers of a List of Box objects

    :param boxes :type: list of box objects
    :return: A numpy array of centers with corresponding indices as the input list
    """
    # print("test")
    centers = []
    for box in boxes:
        centers.append(box.center)
    centers = np.array(centers)
    return centers

def is_same_instance(nuscenes_handle:NuScenes, sample_annotation_token_i: str, sample_annotation_token_j: str)-> Boolean:
    """
    Compares 3D bounding boxes if they belong to the same object instance 
    :param nuscenes_handle: nuscenes wrapper that allows to look for instance token
    :param sample_annotation_token_i: Unique sample_annotation identifier. Identifies Bounding Boxes
    :param sample_annotation_token_j: Unique sample_annotation identifier. Identifies Bounding Boxes
    """
    record_i = nuscenes_handle.get('sample_annotation', sample_annotation_token_i)
    record_j = nuscenes_handle.get('sample_annotation', sample_annotation_token_j)

    if record_i['instance_token'] == record_j['instance_token']:
        return True
    else:
        return False

