from typing import List, Union
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
    centers = []
    for box in boxes:
        centers.append(box.center)
    centers = np.array(centers)
    return centers

def is_same_instance(nuscenes_handle:NuScenes, \
                    sample_annotation_token_i: str,\
                    sample_annotation_token_j: str) -> Boolean:
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

def filter_boxes(nuscenes_handle:NuScenes, 
                        boxes:List[Box],
                        categoryQuery: Union[str,List[str]]) -> List[Box]:
    """
    Returns a list of Bounding Boxes that belong to a specified class or subclass

    :param nuscenes_handle: nuscenes wrapper that allows to look for instance token
    """
    if not isinstance(categoryQuery, list):
        categoryQuery = [categoryQuery]
    
    filtered_boxes = []
    for box in boxes:
        category = nuscenes_handle.get('sample_annotation',box.token)['category_name']
        if any(object_categorie in category for object_categorie in categoryQuery):
            filtered_boxes.append(box)
    return filtered_boxes
