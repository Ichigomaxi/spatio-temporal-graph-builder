import numpy as np
from nuscenes.utils.data_classes import Box
from typing import List

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