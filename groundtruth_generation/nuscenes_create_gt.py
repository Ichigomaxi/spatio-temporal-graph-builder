from enum import Enum, IntEnum
from typing import List

import numpy as np
from cv2 import writeOpticalFlow
from matplotlib.pyplot import box
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from torch import affine_grid_generator
from zmq import device
from utility import filter_boxes, get_box_centers, is_same_instance
from utils.nuscenes_helper_functions import is_valid_box
import torch

class edge_label_classes(IntEnum):
    different_instance = 0
    same_instance = 1
    new_instance = 2

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
def convert_into_one_hot_encoding(
        label:edge_label_classes, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )-> torch.Tensor:

    label_one_hot = torch.zeros(len(edge_label_classes), dtype=torch.uint8, device = device)

    if( label == edge_label_classes.different_instance ):
        label_one_hot[edge_label_classes.different_instance] = 1
    elif( label== edge_label_classes.same_instance ):
        label_one_hot[edge_label_classes.same_instance] = 1
    elif( label== edge_label_classes.new_instance ):
        label_one_hot[edge_label_classes.new_instance] = 1

    return label_one_hot

def generate_edge_label_one_hot(nuscenes_handle:NuScenes,
            sample_annotation_token_a:str,
            sample_annotation_token_b:str,
            new_instances_token_list:List[str] = [],
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )->torch.Tensor:

    label = None
    if (is_same_instance(nuscenes_handle,sample_annotation_token_a \
                                ,sample_annotation_token_b)):
        label = edge_label_classes.same_instance
    elif (is_new_instance_in_graph_scene(nuscenes_handle,sample_annotation_token_a,
                sample_annotation_token_b, new_instances_token_list)):
        label = edge_label_classes.new_instance
    else:
        label = edge_label_classes.different_instance

    return convert_into_one_hot_encoding(label, device=device)

def is_new_instance_in_graph_scene(
            nuscenes_handle:NuScenes,
            sample_annotation_token_a:str,
            sample_annotation_token_b:str,
            new_instances_token_list:List[str]):
    '''

    '''
    sample_annotation_a = nuscenes_handle.get('sample_annotation', sample_annotation_token_a)
    sample_annotation_b = nuscenes_handle.get('sample_annotation', sample_annotation_token_b)

    instance_token_a = sample_annotation_a['instance_token']
    instance_token_b = sample_annotation_b['instance_token']

    if ((instance_token_a in new_instances_token_list)
        or 
        (instance_token_b in new_instances_token_list)):
        # print("instance_token_a: ",instance_token_a)
        # print("instance_token_b: ",instance_token_b)
        # print("new_instances_token_list: ",new_instances_token_list)
        return True
    else:
        return False



def generate_flow_labels(nuscenes_handle:NuScenes,
                        temporal_pointpairs:List[List[int]],\
                         car_box_list:List[List[Box]], centers:np.ndarray):
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

    # def get_box(car_box_list, center):
        
    #     for box_list in car_box_list:
    #         for box in box_list:
    #             if np.equal(box.center,center)==[True,True,False]:
    #                 return box
    #     print('box not found!')
    #     return 0
    
    flow_labels = []
    for point_pair in temporal_pointpairs:
        node_a_center = centers[point_pair[0]]
        node_b_center = centers[point_pair[1]]
        # print(node_a_center)
        # print(node_b_center)

        # node_a_box = get_box(car_box_list, node_a_center)
        # node_b_box = get_box(car_box_list, node_b_center)
        

        node_a_box = car_box_list[point_pair[0]]
        node_b_box = car_box_list[point_pair[1]]

        if not (is_valid_box(node_a_box,node_a_center) and is_valid_box(node_b_box,node_b_center)):
            print('invalid boxes!!!')
            raise ValueError('A box does not correspond to a selected center')

        str_node_a_sample_annotation = node_a_box.token
        str_node_b_sample_annotation = node_b_box.token

        if (is_same_instance(nuscenes_handle,str_node_a_sample_annotation \
                                ,str_node_b_sample_annotation)):
            flow_labels.append(1)
        else:
            flow_labels.append(0)

    return flow_labels
