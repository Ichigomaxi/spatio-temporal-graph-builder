from typing import List
from unittest import TestProgram
from cv2 import writeOpticalFlow
from matplotlib.pyplot import box
from nuscenes.nuscenes import NuScenes
from utility import get_box_centers, filter_boxes

from utility import is_same_instance
from nuscenes.utils.data_classes import Box

import numpy as np

def get_filtered_centers(nusc:NuScenes,sample_token:str):
    sample = nusc.get('sample', sample_token)

    # Get LIDAR sample data token
    sensor = 'LIDAR_TOP'
    lidar_top_sample_data = nusc.get('sample_data', sample['data'][sensor])
    lidar_top_sample_data_token =lidar_top_sample_data['token']
    # Get all Boxes as List of Box objects
    _ , boxes, _= nusc.get_sample_data(lidar_top_sample_data_token, selected_anntokens=None, use_flat_vehicle_coordinates =False)
    # Only Retain Car boxes
    car_boxes = filter_boxes(nusc, boxes = boxes, categoryQuery= 'vehicle.car')
    centers = get_box_centers(car_boxes)
    return centers

#TrackID = InstanceID
def assign_track_ids():
    pass

#Convert TrackIds into one_hot-encoded vector
def convert_into_one_hot_encoding():
    pass

def generate_flow_labels(nuscenes_handle:NuScenes,
                        temporal_pointpairs:List[List[int]],\
                         car_box_list:List[List[Box]],centers):
    '''
    Returns a set of flow parameters corresponding to the temporal connections. 
    A Parameter is 1 if the connected nodes belong to the same object instance.
    Otherwise the parameter is 0.
    args:
    nuscenes_handle:NuScenes
    temporal_pointpairs: List(n,2) contains index pairs in the global indexing (corresponding to centers)
    car_box_list:
    centers: numpy.array(#number of detected objects, 2) contain the object centers from 
            different time frames stacked over eachother 
    return: 
    flow_labels: List(n) List of Flow parameters/labels
    '''

    def get_box(car_box_list, center):
        for box_list in car_box_list:
            for box in box_list:
                if np.equal(box.center,center).all():
                    return box
        print('box not found!')
        return 0
                    
    
    flow_labels = []
    for point_pair in temporal_pointpairs:
        node_a_center = centers[point_pair[0]]
        node_b_center = centers[point_pair[1]]
        print(node_a_center)
        print(node_b_center)

        node_a_box = get_box(car_box_list, node_a_center)
        node_b_box = get_box(car_box_list, node_b_center)
        if(node_a_box == -1 | node_b_box == -1):
            print('box not found!')

        str_node_a_sample_annotation = node_a_box.token
        str_node_b_sample_annotation = node_b_box.token

        if (is_same_instance(nuscenes_handle,str_node_a_sample_annotation \
                                ,str_node_b_sample_annotation)):
            flow_labels.append(1)
        else:
            flow_labels.append(0)
    