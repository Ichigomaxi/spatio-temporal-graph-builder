from typing import Any, List
import numpy as np
import torch
import sys

from torchmetrics import BLEUScore
# sys.path.append("datasets")
# import datasets.NuscenesDataset
from datasets.NuscenesDataset import NuscenesDataset
from datasets.mot_graph import Graph
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes,Box
from utils.nuscenes_helper_functions import get_sample_data_table,get_lidar_pointlcoud_path
from pyquaternion import Quaternion

import open3d as o3d
from open3d import geometry
BLACK = np.array([0,0,0])
LIGHTGREY = np.array([1,1,1]) * 0.75
GREY = np.array([1,1,1]) * 0.5
GREEN = np.array([0, 1, 0])
RED = np.array([1, 0, 0])
BLUE = np.array([0, 0, 1])
WHITE = np.asarray([1,1,1])
YELLOW = (RED + GREEN)/np.linalg.norm((RED + GREEN))
def prepare_single_color_array(color:np.ndarray, array_length:int) -> np.ndarray:
    
    if color is None:
        color_array = [[0.5, 0.5, 0.5] for i in range(array_length)]
        color_array = np.asarray(color_array)
    else:
        assert color.shape == np.zeros(3).shape, "color is not of np.shape == (3) "
        color_array = np.tile(color, (array_length, 1))

    return color_array

def add_pointcloud(points:torch.Tensor, color:np.ndarray =None):
    """
    Inputs:
    points.shape = (n,3)
    color.shape = (3)
    """
    line_set_sequences = [] 
    # if color is None:
    #     color = [[0.5, 0.5, 0.5] for i in range(points.shape[0])]
    # else: 
    #     color = np.tile(color, (points.shape[0], 1))
    color = prepare_single_color_array(color,points.shape[0] )

    pcd = o3d.geometry.PointCloud()
    np_points = points.cpu().numpy().reshape(-1,3)
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(color)

    line_set_sequences.append(pcd)
    return line_set_sequences

def add_line_set(nodes:torch.Tensor, edge_indices: torch.Tensor,color:np.ndarray =None)-> List[o3d.geometry.LineSet]:
    '''
    nodes.shape = (N, 3) only xyz values 
    edge_indices.shape = (2, E)
    color.shape = (3, 1)
    '''
    line_set_sequences = []

    colors: np.ndarray = prepare_single_color_array(color, edge_indices.shape[1] )

    # make sure that edge_labels has same length as edge indices
    assert len(colors) == edge_indices.shape[1]

    # #Transport onto CPU and Transform into numpy array
    # Vector3dVector only takes in up to 3 dimensions. Get only xyz from node_features
    np_nodes = nodes.cpu().numpy().reshape(-1, 3)
    np_edge_indices = edge_indices.cpu().numpy().reshape(2, -1)
    np_edge_indices.astype(np.int32) #Vector2iVector takes in int32 type
    # Transpose if Graph connectivity in COO format with shape :obj:`[2, num_edges]
    if np_edge_indices.shape[0] == 2:
        np_edge_indices = np_edge_indices.T

    line_set = geometry.LineSet(points=o3d.utility.Vector3dVector(np_nodes),
        lines=o3d.utility.Vector2iVector(np_edge_indices))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    line_set_sequences.append(line_set) 
    return line_set_sequences

def add_line_set_labeled(nodes:torch.Tensor, edge_indices: torch.Tensor,edge_labels: torch.Tensor,
                        true_color: np.ndarray = GREEN,
                        false_color: np.ndarray = GREY):
    '''
    nodes.shape = (N, 3) only xyz values 
    edge_indices.shape = (2, E)
    edge_labels.shape = (E, 1)
    '''
    line_set_sequences = []
    mask_edge_label = edge_labels == 1
    
    colors = []
    for edge_label in edge_labels:
        if edge_label == 0:
            colors.append(false_color)
        else:
            colors.append(true_color)
    colors: np.ndarray = np.stack(colors)
    
    # make sure that edge_labels has same length as edge indices
    assert len(colors) == edge_indices.shape[1]

    # #Transport onto CPU and Transform into numpy array
    # Vector3dVector only takes in up to 3 dimensions. Get only xyz from node_features
    np_nodes: np.ndarray = nodes.cpu().numpy().reshape(-1, 3)
    np_edge_indices: np.ndarray = edge_indices.cpu().numpy().reshape(2, -1)
    np_edge_indices.astype(np.int32) #Vector2iVector takes in int32 type
    # Transpose if Graph connectivity in COO format with shape :obj:`[2, num_edges]
    if np_edge_indices.shape[0] == 2:
        np_edge_indices = np_edge_indices.T

    line_set = geometry.LineSet(points=o3d.utility.Vector3dVector(np_nodes),
        lines=o3d.utility.Vector2iVector(np_edge_indices))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    line_set_sequences.append(line_set) 

    return line_set_sequences

def add_bounding_boxes(boxes:List[Box], bbox_color:np.ndarray = GREEN, offset:int = 0):
    """
    Only for nuscenes detections with respect to LIDAR_TOP frame
    """
    line_set_bounding_boxes= []

    for box in boxes:
        center:np.ndarray = box.center
        dim:np.ndarray = box.wlh
        orientation:Quaternion = box.orientation
        rotation_matrix_respect_2_LIDAR = orientation.rotation_matrix
        # center_o3d = o3d.utility.Vector3dVector(center)
        # rotation_matrix_respect_2_LIDAR_o3d = o3d.utility.Matrix3dVector(rotation_matrix_respect_2_LIDAR)
        # dim_o3d = o3d.utility.Vector3dVector(dim)
        # box3d = o3d.geometry.OrientedBoundingBox(center_o3d, 
        #         rotation_matrix_respect_2_LIDAR_o3d, 
        #         dim_o3d)
        box3d = o3d.geometry.OrientedBoundingBox(center, 
                rotation_matrix_respect_2_LIDAR, 
                dim)

        line_set_bounding_box = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set_bounding_box.paint_uniform_color(bbox_color)
        line_set_bounding_box.translate(np.array([0,0,offset]))
        # Rotate by 90 degrees in z- axis
        rot_axis = 2 # Z-axis
        yaw = np.zeros(3)
        yaw[rot_axis] = np.pi*0.5
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        line_set_bounding_box.rotate(rot_mat)
        # Append to list
        line_set_bounding_boxes.append(line_set_bounding_box)
    
    return line_set_bounding_boxes

def add_nuscenes_pointcloud(pointcloud_path:str, point_color:np.ndarray = GREY, offset:int = 0):
    #Load Pointclouds from Nuscenes
    pointcloud_nusc:LidarPointCloud = LidarPointCloud.from_file(pointcloud_path)
    # Transpose Points
    pointcloud_nusc_transposed = np.transpose(pointcloud_nusc.points)
    # Init Open3D pointcloud object
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud_nusc_transposed[:, :3])
    # Translate Pointcloud by given offset in z direction
    pointcloud_o3d.translate(np.array([0,0,offset]))
    #Add chosen color
    point_color= point_color
    points_colors = np.tile(np.array(point_color), (pointcloud_nusc_transposed.shape[0], 1))
    pointcloud_o3d.colors = o3d.utility.Vector3dVector(points_colors)

    return [pointcloud_o3d]

def build_geometries_input_graph_w_pointcloud_w_Bboxes(mot_graph:NuscenesMotGraph, nuscenes_handle: NuScenes):
    geometry_list = []

    geometry_list.extend(
        build_geometries_input_graph_w_pointcloud(mot_graph, nuscenes_handle)
    )

    offset = mot_graph.SPATIAL_SHIFT_TIMEFRAMES
    for i in range(mot_graph.max_frame_dist):
        boxes = mot_graph.graph_dataframe["boxes_dict"][i]
        line_set_bounding_boxes:List[Any] = \
            add_bounding_boxes(boxes, bbox_color=GREEN, offset=offset*i)
        geometry_list.extend(line_set_bounding_boxes)

    return geometry_list

def build_geometries_input_graph_w_pointcloud(mot_graph:NuscenesMotGraph, nuscenes_handle: NuScenes):
    geometry_list = []

    geometry_list_graph = visualize_input_graph_new(mot_graph)
    geometry_list.extend(geometry_list_graph)

    # Read in paths to pointcloud files
    lidar_pcl_path_list = []
    for i in range(mot_graph.max_frame_dist):
        sample_token = mot_graph.graph_dataframe["available_sample_tokens"][i]
        lidar_pcl_path = get_lidar_pointlcoud_path(sample_token, nuscenes_handle)
        lidar_pcl_path_list.append(lidar_pcl_path)

    offset = mot_graph.SPATIAL_SHIFT_TIMEFRAMES
    for i, lidar_path in enumerate(lidar_pcl_path_list):
        geometry_list.extend(add_nuscenes_pointcloud(lidar_path, GREY, offset * i))

    return geometry_list

def visualize_input_graph_new(mot_graph:NuscenesMotGraph, 
                    spatial_edge_color:np.ndarray = RED,
                    temporal_edge_color:np.ndarray = BLUE):
    geometry_list = []

    
    nodes_3d_coord = mot_graph.graph_obj.x[:,:3]
    edge_indices= mot_graph.graph_obj.edge_index
    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # temporal Edges
    temporal_edges_mask = mot_graph.graph_obj.temporal_edges_mask
    temporal_edge_indices_undirected = edge_indices[:,temporal_edges_mask[:,0] ]

    line_set_sequence = add_line_set(
                        nodes= nodes_3d_coord,
                        edge_indices= temporal_edge_indices_undirected, 
                        color= temporal_edge_color)
    geometry_list += line_set_sequence
    #----------------------------------------
    # Spatial Edges
    spatial_edges_mask = ~temporal_edges_mask
    spatial_edge_indices_undirected = edge_indices[:,spatial_edges_mask[:,0]]
    line_set_sequence = add_line_set(
                        nodes= nodes_3d_coord,
                        edge_indices= spatial_edge_indices_undirected, 
                        color= spatial_edge_color)
    geometry_list += line_set_sequence
    #----------------------------------------
    return geometry_list

def visualize_input_graph(mot_graph:NuscenesMotGraph):
    geometry_list = []

    nodes_3d_coord = mot_graph.graph_obj.x[:,:3]
    edge_indices= mot_graph.graph_obj.edge_index
    edge_features = mot_graph.graph_obj.edge_attr

    # Color Points/Nodes
    point_sequence = add_pointcloud(nodes_3d_coord,
                                    color= None)
    geometry_list += point_sequence
    
    assert mot_graph.label_type is not None

    if mot_graph.label_type == "binary":
        edge_type_numbers = edge_features.argmax(dim = 1)
        input_lineset = add_line_set_labeled(nodes = nodes_3d_coord,
                            edge_indices= edge_indices, 
                            edge_labels= edge_type_numbers
                            )
    geometry_list += input_lineset
    # Draw Graph/Edges with Lineset
    # Spatial Edges Red Edges
    

    return geometry_list

def visualize_eval_graph_new(mot_graph:NuscenesMotGraph):
    geometry_list = []
    edge_indices = mot_graph.graph_obj.temporal_directed_edge_indices
    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Color Points/Nodes
    point_sequence = add_pointcloud(mot_graph.graph_obj.x[:,:3],
                                    color= None)
    geometry_list += point_sequence

    #----------------------------------------
    # Active and inactive Edges
    active_edges :torch.Tensor = mot_graph.graph_obj.temporal_directed_edge_preds
    # active_edges :torch.Tensor = mot_graph.graph_obj.active_edges
    # only_active_edges_indices =  temporal_directed_edge_indices[:,active_edges]
    line_set_sequence = add_line_set_labeled(nodes = mot_graph.graph_obj.x[:,:3],
                                    edge_indices = edge_indices,
                                    edge_labels = active_edges,
                                    true_color = BLUE * 0.5,
                                    false_color = LIGHTGREY )
    geometry_list += line_set_sequence

    return geometry_list

def visualize_eval_graph(mot_graph:NuscenesMotGraph):
    geometry_list = []
    edge_indices = mot_graph.graph_obj.edge_index
    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Color Points/Nodes
    point_sequence = add_pointcloud(mot_graph.graph_obj.x[:,:3],
                                    color= None)
    geometry_list += point_sequence

    # Basic graph

    # line_set_sequence = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
    #                                 edge_indices= edge_indices,
    #                                 color = LIGHTGREY*0.1
    #                                 )
    # geometry_list += line_set_sequence

    #----------------------------------------
    # Active and inactive Edges

    active_edges:torch.Tensor = mot_graph.graph_obj.active_edges
    only_active_edges_indices =  edge_indices[:,active_edges]
    # line_set_sequence = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
    #                                 edge_indices= only_active_edges_indices,
    #                                 color = BLUE
    #                                 )
    line_set_sequence = add_line_set_labeled(nodes= mot_graph.graph_obj.x[:,:3],
                                    edge_indices= edge_indices,
                                    edge_labels= active_edges,
                                    true_color= BLUE * 0.5,
                                    false_color=LIGHTGREY )
    geometry_list += line_set_sequence

    #----------------------------------------
    # Get correct predictions
    # active_edges:torch.Tensor = mot_graph.graph_obj.active_edges
    # only_active_edges_indices =  edge_indices[:,active_edges]
    # edge_labels_for_active_edges = mot_graph.graph_obj.edge_labels[active_edges]
    # correct_predictions_edge_indices = only_active_edges_indices[:,edge_labels_for_active_edges >=1]

    # line_set_sequence = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
    #                                 edge_indices= correct_predictions_edge_indices,
    #                                 color = YELLOW
    #                                 )
    # geometry_list += line_set_sequence
    
    #----------------------------------------
    # GT 
    # line_set_sequence = add_line_set_labeled(nodes= mot_graph.graph_obj.x[:,:3],
    #                                 edge_indices= edge_indices,
    #                                 edge_labels= mot_graph.graph_obj.edge_labels,
    #                                 true_color= GREEN * 0.5,
    #                                 false_color=LIGHTGREY )

    # geometry_list += line_set_sequence
    #----------------------------------------

    return geometry_list

def visualize_output_graph_new(mot_graph:NuscenesMotGraph):
    geometry_list = []

    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Color Points/Nodes
    point_sequence = add_pointcloud(mot_graph.graph_obj.x[:,:3],
                                    color= None)
    geometry_list += point_sequence
    #----------------------------------------
    # Get all correct edge predictions aka True positive
    # prediction=1 and label =1
    temporal_directed_edge_indices :torch.Tensor = mot_graph.graph_obj.temporal_directed_edge_indices
    temporal_directed_positive_active_edges_mask :torch.BoolTensor = mot_graph.graph_obj.temporal_directed_edge_preds > 0
    temporal_directed_edge_labels :torch.Tensor = mot_graph.graph_obj.temporal_directed_edge_labels
    # Get set of positive active edges
    edge_indices_for_positive_active_edges = temporal_directed_edge_indices[:,temporal_directed_positive_active_edges_mask]
    # GEt set of corresponding labels
    labels_corresponding_to_positive_active_edges = temporal_directed_edge_labels[temporal_directed_positive_active_edges_mask]
    # Get set of correctly assigned active edges
    true_positive_predictions_edge_indices =edge_indices_for_positive_active_edges[:, labels_corresponding_to_positive_active_edges > 0] 

    line_set_sequence = add_line_set(
                    nodes= mot_graph.graph_obj.x[:,:3],
                    edge_indices= true_positive_predictions_edge_indices, 
                    color= GREEN)
    geometry_list += line_set_sequence
    
    #----------------------------------------
    # Get all incorrect positive edge predictions aka False Positives
    # prediction=1 but label =0 
    false_positive_predictions_edge_indices = edge_indices_for_positive_active_edges[:, labels_corresponding_to_positive_active_edges <= 0] 
    line_set_sequence = add_line_set(
                    nodes= mot_graph.graph_obj.x[:,:3],
                    edge_indices= false_positive_predictions_edge_indices, 
                    color= RED)
    geometry_list += line_set_sequence
    #----------------------------------------
    # Get the False Negatives aka prediction= 0 but label=1
    temporal_directed_negative_active_edges_mask :torch.BoolTensor = mot_graph.graph_obj.temporal_directed_edge_preds <= 0
     # Get set of negative active edges
    edge_indices_for_negative_active_edges = temporal_directed_edge_indices[:,temporal_directed_negative_active_edges_mask]
    # GEt set of corresponding labels
    labels_corresponding_to_negative_active_edges = temporal_directed_edge_labels[temporal_directed_negative_active_edges_mask]
    # Get set of edge_indices for False Negative prediction 
    false_negative_predictions_edge_indices = edge_indices_for_negative_active_edges[:, labels_corresponding_to_negative_active_edges > 0]

    line_set_sequence = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
                    edge_indices= false_negative_predictions_edge_indices, 
                    color= BLUE)

    geometry_list += line_set_sequence
    #----------------------------------------
    # Get the True Negatives aka prediction= 0 but label=0
    # Get set of edge_indices for True Negative prediction 
    true_negative_predictions_edge_indices = edge_indices_for_negative_active_edges[ :, labels_corresponding_to_negative_active_edges <= 0]
    line_set_sequence = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
                    edge_indices= true_negative_predictions_edge_indices,
                    color= GREY)

    geometry_list += line_set_sequence

    assert len(mot_graph.graph_obj.temporal_directed_edge_indices.T) == \
        len(true_positive_predictions_edge_indices.T) + \
        len(false_positive_predictions_edge_indices.T) + \
        len(false_negative_predictions_edge_indices.T) + \
        len(true_negative_predictions_edge_indices.T)


    #----------------------------------------
    # Spatial Edges
    temporal_edges_mask = mot_graph.graph_obj.temporal_edges_mask
    spatial_edges_mask = ~temporal_edges_mask
    spatial_edge_indices_undirected = mot_graph.graph_obj.edge_index[:,spatial_edges_mask[:,0] ]

    line_set_sequence = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
                    edge_indices= spatial_edge_indices_undirected , color= GREY)
    geometry_list += line_set_sequence
    #----------------------------------------

    return geometry_list

def visualize_output_graph(mot_graph:NuscenesMotGraph):
    geometry_list = []

    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Color Points/Nodes
    point_sequence = add_pointcloud(mot_graph.graph_obj.x[:,:3],
                                    color= None)
    geometry_list += point_sequence
    #----------------------------------------
    # Active and inactive Edges
    # edge_preds = mot_graph.graph_obj.edge_preds
    # active_edges = mot_graph.graph_obj.active_edges
    line_set_sequence = add_line_set_labeled(nodes= mot_graph.graph_obj.x[:,:3],
                                    edge_indices= mot_graph.graph_obj.edge_index,
                                    edge_labels= mot_graph.graph_obj.edge_labels)

    geometry_list += line_set_sequence
    #----------------------------------------

    return geometry_list

def visualize_basic_graph(mot_graph:NuscenesMotGraph):

    geometry_list = []

    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Temporal Edges 
    temporal_edges_mask = mot_graph.graph_obj.temporal_edges_mask
    temporal_edges = mot_graph.graph_obj.edge_index.T[temporal_edges_mask].reshape(-1,2)
    temporal_edges = temporal_edges.T
    
    line_set_sequence_temporal_connections = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
                                    edge_indices= temporal_edges,
                                    color = np.asarray([ 0, 0, 1]))
                                    

    geometry_list += line_set_sequence_temporal_connections
    
    #----------------------------------------
    # Spatial Edges

    spatial_edges_mask = ~ mot_graph.graph_obj.temporal_edges_mask
    spatial_edges = mot_graph.graph_obj.edge_index.T[spatial_edges_mask].reshape(-1,2)
    spatial_edges = spatial_edges.T

    line_set_sequence_spatial_connections = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
                                    edge_indices= spatial_edges,
                                    color=np.asarray([ 0, 1, 0]))
                                    
                                    
    geometry_list += line_set_sequence_spatial_connections

    #----------------------------------------
    # Color Points/Nodes
    point_sequence = add_pointcloud(mot_graph.graph_obj.x[:,:3],
                                    color= None)
    geometry_list += point_sequence
    #----------------------------------------
    
    
    #----------------------------------------

    return geometry_list

def visualize_graph_obj_without_GT(graph_obj:Graph):

    geometry_list = []
    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Temporal Edges 
    temporal_edges_mask = graph_obj.temporal_edges_mask
    temporal_edges = graph_obj.edge_index.T[temporal_edges_mask].reshape(-1,2)
    temporal_edges = temporal_edges.T
    
    line_set_sequence_temporal_connections = add_line_set(nodes= graph_obj.x[:,:3],
                                    edge_indices= temporal_edges,
                                    color = np.asarray([ 0, 0, 1]))
                                    

    geometry_list += line_set_sequence_temporal_connections
    
    #----------------------------------------
    # Spatial Edges

    spatial_edges_mask = ~ graph_obj.temporal_edges_mask
    spatial_edges = graph_obj.edge_index.T[spatial_edges_mask].reshape(-1,2)
    spatial_edges = spatial_edges.T

    line_set_sequence_spatial_connections = add_line_set(nodes= graph_obj.x[:,:3],
                                    edge_indices= spatial_edges,
                                    color=np.asarray([ 0, 1, 0]))
                                    
                                    
    geometry_list += line_set_sequence_spatial_connections

    return geometry_list

def visualize_graph_obj_without_GT_selected_edges(graph_obj:Graph, selected_edge_indices:torch.Tensor):

    geometry_list = []
    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Selected Edges 
    line_set_sequence_temporal_connections = add_line_set(nodes= graph_obj.x[:,:3],
                                    edge_indices= selected_edge_indices,
                                    color = RED)
    geometry_list += line_set_sequence_temporal_connections
    #----------------------------------------
    # Spatial Edges
    all_edge_indices:List[int] = graph_obj.edge_index.T.tolist()
    selected_edges = selected_edge_indices.T.tolist()
    selected_edges_intuples = []
    for i in range(len(selected_edges)):
        indices_tuple = tuple(selected_edges[i])
        selected_edges_intuples.append(indices_tuple)

    unselected_edges =[]
    unselected_edges_mask = torch.zeros(len(all_edge_indices))
    for i in range(len(all_edge_indices)):
        if tuple(all_edge_indices[i]) not in selected_edges_intuples:
            # indices_in_list :List[int] = all_edge_indices[i]
            # unselected_edges.append(indices_in_list)
            unselected_edges_mask[i] = 1
        else:
            unselected_edges_mask[i] = 0
    
    unselected_edges_mask :torch.Tensor = unselected_edges_mask > 0 
    unselected_edges = graph_obj.edge_index[:,unselected_edges_mask]
    # unselected_edges = unselected_edges.T
    line_set_sequence_spatial_connections = add_line_set(nodes= graph_obj.x[:,:3],
                                    edge_indices= unselected_edges,
                                    color=GREY)    
                                    
    geometry_list += line_set_sequence_spatial_connections

    return geometry_list

def visualize_geometry_list(geometry_list:list):
    o3d.visualization.draw_geometries(geometry_list)

def main(mode:str = "single", filterBoxes_categoryQuery="vehicle.car"):
    dataset = NuscenesDataset('v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes")
   
    MotGraphList =[]
    if mode == "single":
        MotGraph_object = dataset.get_single_nuscenes_mot_graph_debugging(
                                    specific_device=None,
                                    label_type="binary",
                                    filterBoxes_categoryQuery = filterBoxes_categoryQuery)
        MotGraphList.append(MotGraph_object)
    elif mode == "mini_train":
        pass
    elif mode == "mini_val":
        dataset_params = {}
        mot_graph_dataset = NuscenesMOTGraphDataset(dataset_params= dataset_params,
                                                    mode= mode, 
                                                    nuscenes_handle=dataset.nuscenes_handle)
    else:
        MotGraphList = dataset.get_nuscenes_mot_graph_list_debugging(True,
                                    specific_device=None,
                                    label_type="binary",
                                    filterBoxes_categoryQuery = filterBoxes_categoryQuery)

    for i, mot_graph in enumerate(MotGraphList):
        geometry_list_input = visualize_input_graph(mot_graph)
        geometry_list_output = visualize_output_graph(mot_graph)
        # geometry_list_basic = visualize_basic_graph(mot_graph)

        if len(geometry_list_input) != 0:
            print("Input-Graph: starting {}-th visualization!".format(i))
            o3d.visualization.draw_geometries(geometry_list_input)
            print("Input-Graph:stoped {}-th visualization!".format(i))

        if len(geometry_list_output) != 0:   
            print("Output-Graph: starting {}-th visualization!".format(i))
            o3d.visualization.draw_geometries(geometry_list_output)
            print("Output-Graph:stoped {}-th visualization!".format(i))

        # if len(geometry_list_basic) != 0: 
        #     print("starting {}-th visualization!".format(i))
        #     o3d.visualization.draw_geometries(geometry_list_basic)
        #     print("stoped {}-th visualization!".format(i))
