'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import json
import os
import os.path as osp
import re
from collections import OrderedDict
from copy import deepcopy
from tracemalloc import start
from turtle import st
from typing import Any, Dict, List, Tuple
from cv2 import threshold

import numpy as np
import pandas as pd
from sklearn.utils import assert_all_finite
from tensorboard import summary
import torch
from datasets.NuscenesDataset import NuscenesDataset
from datasets.mot_graph import Graph
from datasets.nuscenes.classes import name_from_id
from datasets.nuscenes.reporting import (add_results_to_submit,
                                         build_results_dict)
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.nuscenes import Box, NuScenes
from pyquaternion import Quaternion
from pytorch_lightning import Callback
from torch.nn import functional
from torch_scatter import scatter_add
from zmq import device

from utils.misc import load_pickle, save_pickle
from utils.nuscenes_helper_functions import (
    get_gt_sample_annotation_pose, transform_detections_lidar2world_frame)

###########################################################################
# MOT Metrics
###########################################################################
UNTRACKED_ID = -1 # Describes Nodes that have not been considered for tracking yet 


def compute_nuscenes_3D_mot_metrics(
                config_path:str,
                eval_set:str, 
                result_path:str,
                out_mot_files_path,
                nuscenes_version,
                nuscenes_dataroot,
                verbose:bool =True,
                render_classes = None,
                render_curves :bool = True):
    """
    Returns Individual and overall 3D MOTmetrics for all sequeces belonging to the given evaluation set
    Args:
    config_path: str, relative path to evaluation config file 
    eval_set: str, defines evaluation set
    result_path: str, absolute path to the json-file with the tracked detections for the evaluation set
    out_mot_files_path: str: absolute path to directory, for the generated outputs 
    nuscenes_version:  Nuscenes-spefic version, same as the version used for nuscenes_handle
    nuscenes_dataroot: absolute path to Nuscenes dataset, same as when initializing the nuscenes_handle\
    verbose: ...
    render_classes: ....
    render_curves: bool, if true: generates plots and exports them as pdf for computed MOT metrics  
    Returns:
        - metrics_summary: Dict[str,Any]
    """
    assert eval_set in NuscenesDataset.ALL_SPLITS, "Given Eval set is not a valid split for nuscenes evaluation! \n Given{}".format(eval_set, )

    cfg_ =None
    with open(config_path, 'r') as _f:
        cfg_ = TrackingConfig.deserialize(json.load(_f))

    os.makedirs(out_mot_files_path, exist_ok=True) # Make sure dir exists

    nusc_eval = TrackingEval(config=cfg_, 
                            result_path=result_path, 
                            eval_set=eval_set, 
                            output_dir=out_mot_files_path,
                            nusc_version= nuscenes_version, 
                            nusc_dataroot= nuscenes_dataroot, 
                            verbose=verbose,
                            render_classes=render_classes)

    metrics_summary = nusc_eval.main(render_curves=render_curves)

    if verbose:
        print(metrics_summary)
    
    return metrics_summary

def get_node_indices_from_timeframe_i(timeframe_numbers:torch.Tensor, timeframe:int, as_tuple:bool= False)-> torch.Tensor:
    '''
    Returns node indices of a given timeframe.
    Args:
    timeframe_numbers: torch.Tensor shape(num_nodes,1)
    timeframe: int, from 0 to max_number_of_frames_per_graph
    as_tuple: Bool, if True, indices are returned as tuple, see doc of torch.nonzero() for details.
    '''
    nodes_from_frame_i_mask = timeframe_numbers == timeframe
    node_idx_from_frame_i = torch.nonzero(nodes_from_frame_i_mask, as_tuple=as_tuple)
    return node_idx_from_frame_i

def filter_out_spatial_edges(incoming_edge_idx:torch.Tensor, 
                                graph_obj:Graph):
    """
    incoming_edge_idx : torch.tensor, shape(num_incoming_edges), describes the indices of the list of edges. 
                    These have different source_nodes and the same target_node
    """
    incoming_edge_idx_without_spatial_connections = []

    temporal_edges_mask:torch.Tensor = graph_obj.temporal_edges_mask
    spatial_edges_mask = ~temporal_edges_mask
    spatial_edge_idx:torch.Tensor = graph_obj.edge_index[:,spatial_edges_mask[:,0]] #
    
    incoming_edges:torch.Tensor = graph_obj.edge_index[:,incoming_edge_idx]
    target_node_id:torch.Tensor = incoming_edges[1,0]
    assert (target_node_id == incoming_edges[1]).all()

    spatial_rows, spatial_columns = spatial_edge_idx[0], spatial_edge_idx[1]
    spatial_target_node_id_mask:torch.Tensor = spatial_columns == target_node_id 
    spatial_source_node_ids:torch.Tensor = spatial_rows[spatial_target_node_id_mask] # These source_nodes must leave
    unnecessary_spatial_edges:torch.Tensor = spatial_edge_idx[:,spatial_target_node_id_mask]
    # current_nodes_from_frame_i_mask = graph_obj.timeframe_number == current_timeframe

    # Get only the temporal edges
    # elementwise Comparison of 2 vectors 
    incoming_rows = incoming_edges[0].unsqueeze(dim = 1) # give extra dimension to make it broadcastable
    mask_2d_comparison = (spatial_source_node_ids == incoming_rows) # shape = [num_incoming_rows, num_spatial_source_nodes]
    mask4_spatial_incoming_rows = (mask_2d_comparison).sum(dim=1) >= 1 # dim = 1 to identify incoming_rows that to spatial edges
    mask4_temporal_incoming_rows = ~mask4_spatial_incoming_rows # identify incoming_rows that to temporal edges
    incoming_edges_without_spatial_connections = incoming_edges[:, mask4_temporal_incoming_rows]

    # Retrieve the corresponding edge_indes from graph_obj.edge_index
    incoming_edge_idx_without_spatial_connections = incoming_edge_idx[mask4_temporal_incoming_rows]
    test_bool_tensor: torch.BoolTensor = graph_obj.edge_index[:,incoming_edge_idx_without_spatial_connections]\
                            == incoming_edges_without_spatial_connections
    assert test_bool_tensor.all()
    # for incoming_edge_index in incoming_edge_idx:
    #     incoming_edge_indices = graph_obj.edge_index[:,incoming_edge_index]
    #     if incoming_edge_indices[0] not in spatial_source_node_ids:
    #         incoming_edge_idx_without_spatial_connections.append(incoming_edge_index)
    
    # if incoming_edge_idx_without_spatial_connections:
    #     incoming_edge_idx_without_spatial_connections = torch.stack(incoming_edge_idx_without_spatial_connections)
    # else:
    #     incoming_edge_idx_without_spatial_connections = incoming_edge_idx
    # incoming_edges_without_spatial_connections = graph_obj.edge_index[:,incoming_edge_idx_without_spatial_connections] #

    source_node_ids = incoming_edges_without_spatial_connections[0]
    timeframe_number = graph_obj.timeframe_number
    source_node_times = timeframe_number[source_node_ids]
    target_node_time = timeframe_number[target_node_id]
    
    if not len(incoming_edge_idx_without_spatial_connections) > 0:
        print("Found a node without any temporal connections! ")
        assert incoming_edge_idx_without_spatial_connections.shape[0] == 0
    else:
        if (target_node_time == source_node_times).all() and len(source_node_times) > 0:
            print("not possible")
        assert (target_node_time != source_node_times).all()
    
    return incoming_edge_idx_without_spatial_connections

def filter_for_past_edges(incoming_edge_idx:torch.Tensor, 
                            current_timeframe:int, frames_per_graph:int,
                            graph_obj:Graph):
    '''
    Returns edges connecting to past and present edges

    incoming_edge_idx : torch.tensor, shape(num_incoming_edges), describes the indices of the list of edges. 
                    These have different source_nodes and the same target_node
    current_timeframe: int
    '''
    past_edges = None
    # Determine Future timeframes
    # max_possible_timeframe = (mot_graph.max_frame_dist-1)
    max_possible_timeframe = (frames_per_graph-1)
    future_timeframes = range(max_possible_timeframe, current_timeframe , -1)
    # Last frame has no connections to past frames
    # Just continoue with given edges
    if max_possible_timeframe == current_timeframe:
        return incoming_edge_idx
    
    # Determine all future nodes indices
    future_node_idx = []
    for timeframe in future_timeframes: 
        # future_nodes_from_frame_i_mask = mot_graph.graph_dataframe["timeframes_all"] == timeframe
        future_nodes_from_frame_i_mask = graph_obj.timeframe_number == timeframe
        future_node_idx_from_frame_i = torch.nonzero(future_nodes_from_frame_i_mask, as_tuple=True)[0]
        future_node_idx.append(future_node_idx_from_frame_i)
    future_node_idx = torch.cat(future_node_idx) # concatenate

    #Check wich edges connect to future nodes
    incoming_edge_indices = graph_obj.edge_index[:,incoming_edge_idx] # shape: 2,num_incoming_edge_idx
    rows:torch.Tensor = incoming_edge_indices[0] 
    columns:torch.Tensor = incoming_edge_indices[1]
    test_tensor = columns.eq(future_node_idx.unsqueeze(dim=1)) # unqueeze to make it broadcastable
    test_tensor = test_tensor.sum(dim= 0)

    # test_if_self = (graph_obj.edge_index[0] == graph_obj.edge_index[1].unsqueeze(1))
    # print(test_if_self.sum(dim = 0).all() == False)
    
    assert test_tensor.all() == False, \
        'Future node as target node! Invalid!' + \
            ' \Target node must be part of current timeframe/' + \
                ' must be the same for all incoming edges'
    edges_connecting_to_future_nodes_mask = rows.eq(future_node_idx.unsqueeze(dim=1)).sum(dim= 0) # integer Tensor I think
    edges_connecting_to_future_nodes_mask = edges_connecting_to_future_nodes_mask >= 1 # Make into torch.ByteTensor
    edges_connecting_to_past_nodes_mask = ~edges_connecting_to_future_nodes_mask
    past_edges = incoming_edge_idx[edges_connecting_to_past_nodes_mask]
    return past_edges

def assign_definitive_connections(mot_graph:NuscenesMotGraph, tracking_threshold:float):
    '''
    backtracks the neighboring node with the highest confidence to connect with a give node.
    Idea: iterate in reverse. start backtracking from last frame and continue until reaching the first frame
    Edges for backtracking are differentiated between incoming(flowing in) edges and outgoing (flowing out) edges.
    Therefore, we follow the COO convention and the flow-convention (edges are described: source_node to target_node)
    rows = source_nodes, columns = target_nodes

    Saves variables in graph_object (Graph):
    mot_graph.graph_obj.active_edges: torch.ByteTensor, shape(num_edges), True if connecting a target_node with corresponding active neighbor
    mot_graph.graph_obj.tracking_confidence: torch.LongTensor, shape(num_edges), contains edge_predictions of corresponding active edges
    mot_graph.graph_obj.active_neighbors: torch.IntTensor, shape(num_edges) , contains the node_index of the corresponding active neighbor  

    Returns: void 
    '''
    def activate_edge_state(active_edge_id:torch.Tensor,
                tracking_confidence:torch.Tensor,
                active_neighbor_node_index:torch.Tensor,
                active_connections:torch.Tensor, 
                tracking_confidence_list:torch.Tensor,
                active_neighbors:torch.Tensor ):
        # Describes if an edge is active or inactive. 
        # If not active the element is NaN
        active_connections[active_edge_id] = 1 
        # safe the confidence (prediction value) that lead to this tracking.
        # If not active the element is NaN
        tracking_confidence_list[active_edge_id] = tracking_confidence 
        # Describes the source node that is 
        # actively connected by this active edge . If not active the element is NaN
        active_neighbors[active_edge_id] = active_neighbor_node_index

    def deactivate_edges(lossing_idx_global :torch.Tensor, 
                active_connections:torch.Tensor, 
                tracking_confidence_list:torch.Tensor,
                active_neighbors:torch.Tensor ):

        active_connections[lossing_idx_global] = float('nan') 
        tracking_confidence_list[lossing_idx_global] = float('nan')
        active_neighbors[lossing_idx_global] = float('nan')


    def selfcompare_vector(vector_list_scalars:torch.Tensor):
        assert vector_list_scalars.size() == torch.Size([len(vector_list_scalars)]), "tensor is not one dimensional"
        comparison_matrix = vector_list_scalars.eq(vector_list_scalars.unsqueeze(dim=1))
        return comparison_matrix.fill_diagonal_(False)

    def assing_single_active_edge_per_target_node():
        """
        Returns:
        active_connections:
        Describes if an edge is active or inactive.If not active the element is NaN
        
        tracking_confidence_list:
        safes the confidence (prediction value) that lead to this tracking.
        If not active the element is NaN
        
        active_neighbors:
        Describes the source node that is 
        actively connected by this active edge . If not active the element is NaN
        """
        edge_indices = mot_graph.graph_obj.edge_index
        edge_preds = mot_graph.graph_obj.edge_preds 

        active_connections = torch.zeros_like(edge_preds) * float('nan')
        tracking_confidence_list = torch.zeros_like(edge_preds) *float('nan')
        active_neighbors = torch.zeros_like(edge_preds) *float('nan')

        timeframes = range(1,mot_graph.max_frame_dist) # start at 1 to skip first frame

        # Iterate in reverse to start from the last frame
        for i in reversed(timeframes): 
            # filter nodes from frame i
            nodes_from_frame_i_mask = mot_graph.graph_dataframe["timeframes_all"] == i
            # nodes_from_frame_i_mask = ~nodes_from_frame_i_mask
            node_idx_from_frame_i = torch.nonzero(nodes_from_frame_i_mask, as_tuple=True)[0]
            # filter their incoming edges
            for node_id in node_idx_from_frame_i:
                #########################################
                incoming_edge_idx = []
                rows, columns = edge_indices[0],edge_indices[1] # COO-format
                edge_mask:torch.Tensor = torch.eq(columns, node_id)
                incoming_edge_idx = edge_mask.nonzero().squeeze()
                incoming_edge_idx= filter_out_spatial_edges(incoming_edge_idx, mot_graph.graph_obj)
                if incoming_edge_idx.shape[0] == 0:
                    # Found a node without temporal edges
                    # cannot assign any active connections to it 
                    # it will be considered as a node without any active connections and therefore will be assigned a new trackId
                    # continue with next node
                    continue
                incoming_edge_idx = filter_for_past_edges(incoming_edge_idx, i, mot_graph.max_frame_dist, mot_graph.graph_obj)
                if incoming_edge_idx.shape[0] == 0:
                    # Found a node without temporal edges from the past
                    # cannot assign any active connections to it 
                    # it will be considered as a node without any active connections and therefore will be assigned a new trackId
                    # continue with next node
                    continue
                #######################################
                # Threshold the incoming predictions. 
                # Otherwise predictions without probability above a certain threshold will be selected
                incoming_edge_predictions = edge_preds[incoming_edge_idx]
                valid_edges_mask:torch.Tensor = (incoming_edge_predictions > tracking_threshold)
                if valid_edges_mask.any()==False:
                    # Given all incoming edges have a prediction probability (tracking certainty) lower than the given threshold
                    # Therefore, this node can not be connected to any of previous nodes
                    # No unique active edge can be assigned
                    # continue with next node
                    continue
                valid_incoming_edge_predictions = torch.zeros_like(incoming_edge_predictions)
                valid_incoming_edge_predictions[valid_edges_mask] = incoming_edge_predictions[valid_edges_mask]
                ########################################
                # Get edge predictions and provide the active neighbor and its tracking confindence
                
                probabilities = functional.softmax(valid_incoming_edge_predictions, dim=0 )
                # Get the edge_index with the highest log_likelihood
                log_probabilities = functional.log_softmax(valid_incoming_edge_predictions, dim=0)
                local_index = torch.argmax(log_probabilities, dim = 0)
                assert local_index == torch.argmax(functional.softmax(valid_incoming_edge_predictions, dim=0 ), dim = 0)
                
                # look for the global edge index
                global_index = incoming_edge_idx[local_index]
                active_edge_id = global_index

                # assert len(active_edge_id) == 1,"more than one active connection is ambigous!"
                # tracking_confidence is equal to the highest edge_prediction of the valid incoming edges
                tracking_confidence = valid_incoming_edge_predictions[local_index] # tracking confidence for evaluation AMOTA-metric
                
                # set active edges
                active_edge = edge_indices[:,active_edge_id]
                assert rows[active_edge_id] == active_edge[0]

                # # Describes if an edge is active or inactive. 
                # # If not active the element is NaN
                # active_connections[active_edge_id] = 1 
                # # safe the confidence (prediction value) that lead to this tracking.
                # # If not active the element is NaN
                # tracking_confidence_list[active_edge_id] = tracking_confidence 
                # # Describes the source node that is 
                # # actively connected by this active edge . If not active the element is NaN
                # active_neighbors[active_edge_id] = rows[active_edge_id] 
                source_node_index = rows[active_edge_id] 
                activate_edge_state(active_edge_id, 
                        tracking_confidence=tracking_confidence, 
                        active_neighbor_node_index= source_node_index,
                        active_connections= active_connections,
                        tracking_confidence_list = tracking_confidence_list,
                        active_neighbors=active_neighbors,)

        return active_connections, tracking_confidence_list, active_neighbors
        
    def decide_active_edge_between_ambiguous_edges(
                all_global_ambigous_edge_idx:torch.Tensor, 
                active_connections:torch.Tensor, 
                tracking_confidence_list:torch.Tensor,
                active_neighbors:torch.Tensor ):
        """
        Computes a winner active edge and sets all remaining ambiguous edges as inactive
        Takes a subset of active edges that are connected to the same source node but 
        """
        # # Get global node id
        # current_global_edge_id = isnotNan_mask_idx[local_row_index]
        # current_global_source_node_id = active_neighbors[current_global_edge_id] 
        # assert (current_global_source_node_id == active_neighbors[isnotNan_mask_idx[ambigous_node_idx]]).all
        # local_active_neighbor = local_active_neighbor_list[local_row_index]
        # assert current_global_source_node_id == local_active_neighbor
        # # Get all active edges
        # global_ambigous_edge_idx = isnotNan_mask_idx[ambigous_node_idx]
        # # Combine current and other ambigous edge_idx
        # all_global_ambigous_edge_idx = torch.cat([current_global_edge_id.unsqueeze(dim=1),global_ambigous_edge_idx], dim=0)
        all_global_ambigous_edge_idx = all_global_ambigous_edge_idx.squeeze_()
        ambigous_active_edges = active_connections[all_global_ambigous_edge_idx]
        ambigous_tracking_confidence = tracking_confidence_list[all_global_ambigous_edge_idx]
    
        # Perform softmax + argmax
        log_probs = functional.log_softmax(ambigous_tracking_confidence, dim = 0)
        winning_subsample_id = torch.argmax(log_probs, dim = 0)
        winning_id_global = all_global_ambigous_edge_idx[winning_subsample_id]
        
        # Set remaining active edge-candidates to inactive -> False
        lossing_idx_global = all_global_ambigous_edge_idx[all_global_ambigous_edge_idx != winning_id_global]
        # active_connections[lossing_idx_global] = float('nan') 
        # tracking_confidence_list[lossing_idx_global] = float('nan')
        deactivate_edges(lossing_idx_global, active_connections, tracking_confidence_list, active_neighbors)

    ##########################################################################
    # Main thread start here
    edge_indices = mot_graph.graph_obj.edge_index
    edge_preds = mot_graph.graph_obj.edge_preds 
    ##########################################################################
    # Assign for every node an active edge except the first frames nodes
    # Start from the latest nodes and propagate reverse in time 
    active_connections, tracking_confidence_list, active_neighbors = \
            assing_single_active_edge_per_target_node()
    ##########################################################################    
    # Determine if any active source_node has more than 1 active edges assigned
    # to them from nodes from the same timeframe
    # Check for duplicates in active_neighbors

    isNan_mask = active_neighbors !=active_neighbors
    isnotNan_mask = ~isNan_mask
    local_active_neighbor_list = active_neighbors[isnotNan_mask]

    comparison_matrix = selfcompare_vector(local_active_neighbor_list)
    # comparison_matrix = local_active_neighbor_list.eq(local_active_neighbor_list.unsqueeze(dim=1))
    # comparison_matrix.fill_diagonal_(False)
    # assert (comparison_matrix == selfcompare_vector(local_active_neighbor_list)).all()

    num_ambigous_neighbors_per_source_node = comparison_matrix.sum(dim=0)
    # Tell us if there are any nodes connected to the same source node
    # However, only indicates potential candidates for ambigous active edges 
    # because they are valid if the connected target nodes are in different timeframes
    is_ambigous_neighbor_candidates = num_ambigous_neighbors_per_source_node > 0

    if(is_ambigous_neighbor_candidates.any()):
        comparison_matrix = comparison_matrix.triu() # only consider each connection once
        # Important!: These are indices belonging to the comparison matrix
        # The indices are derived from the local_indices from the "local_active_neighbor_list"- variable
        ambigous_source_node_local_idx = comparison_matrix.nonzero() # List with all the ambigous_edge_candidate's indices from the comparison matrix
        isnotNan_mask_idx = isnotNan_mask.nonzero() # Mask to map local indices to global indices

        # get corresponding source node and its active_edge_idx
        # Iterate row by row through comparison_matrix : 
        # comparison_matrix.shape[0] = len(local_active_neighbor_list)
        # filter duplicates(active edge candidates) 
        for local_row_index in range(len(local_active_neighbor_list)):
            row_idx = ambigous_source_node_local_idx[:,0]
            column_idx = ambigous_source_node_local_idx[:,1]
            # Check if the current active_node has ambigous connections
            if local_row_index in row_idx:
                # Get index of ambigous edge candidates
                local_mask = row_idx == local_row_index
                ambigous_candidates_node_idx = column_idx[local_mask]
                ##########################################################################
                # Check if they are from the same time frame
                current_global_edge_id = isnotNan_mask_idx[local_row_index] # current edge index from edges_indices-list
                global_ambigous_candidates_edge_idx = \
                    isnotNan_mask_idx[ambigous_candidates_node_idx] # contains the global indices to access the edges thata also have the same active neigbor_node
                
                # Include the current active edges to the candidate list
                all_global_ambigous_edge_candidates_idx = torch.cat([current_global_edge_id.unsqueeze(dim=1),global_ambigous_candidates_edge_idx], dim=0)
                all_global_ambigous_edge_candidates_idx = all_global_ambigous_edge_candidates_idx.squeeze()
                # Get the edges from candidates
                ambigous_candidate_edges = \
                    edge_indices[:,all_global_ambigous_edge_candidates_idx] # contains the ambigous candidates edges with source[0] and target[1] node index
                ambigous_candidate_edges = ambigous_candidate_edges.squeeze() 
                if ambigous_candidate_edges.shape[0] != edge_indices.shape[0] \
                    and ambigous_candidate_edges.shape[1] != len(all_global_ambigous_edge_candidates_idx):
                    ambigous_candidate_edges = ambigous_candidate_edges.T

                # must retain basic shape [2,num_edges]
                assert ambigous_candidate_edges.shape[0] == edge_indices.shape[0] \
                        and ambigous_candidate_edges.shape[1] == len(all_global_ambigous_edge_candidates_idx),\
                            "The ambigous candidate edge Tensor does not come in the common edge.inidizes shape [2,num_edges]"
                # Get corresponding Target nodes
                ambiguous_candidates_target_node_indices = ambigous_candidate_edges[1]
                # Get corresponding timeframes
                timeframe_per_node:torch.Tensor  = mot_graph.graph_obj.timeframe_number
                ambiguous_candidates_target_node_timeframe = timeframe_per_node[ambiguous_candidates_target_node_indices]
                current_active_edge_timeframe = ambiguous_candidates_target_node_timeframe[0]

                # Comparison matrix between timeframes 
                # timeframe_comparison_matrix :torch.Tensor = \
                #     ambiguous_candidates_target_node_timeframe.eq(\
                #             ambiguous_candidates_target_node_timeframe(dim=1))
                # timeframe_comparison_matrix.fill_diagonal_(False)
                timeframe_comparison_matrix :torch.Tensor = selfcompare_vector(ambiguous_candidates_target_node_timeframe.squeeze())

                # Check if the candidates in the same timeframe
                # If True, continue as usual
                if (ambiguous_candidates_target_node_timeframe == current_active_edge_timeframe ).all():
                    ambigous_node_idx = all_global_ambigous_edge_candidates_idx
                    decide_active_edge_between_ambiguous_edges(ambigous_node_idx, 
                            active_connections, 
                            tracking_confidence_list, 
                            active_neighbors)
                
                # If a subset of nodes is within the same timeframe
                # Then divide ambigous_candidates into timeframe-bins
                elif((timeframe_comparison_matrix.sum(dim=0) > 0).any()):
                    # Filter only between nodes from the same timeframe
                    considered_timeframes = torch.unique(ambiguous_candidates_target_node_timeframe)
                    for timeframe_i in considered_timeframes:
                        time_mask :torch.BoolTensor= timeframe_i == ambiguous_candidates_target_node_timeframe.squeeze()
                        ambigous_node_idx_time_i =  all_global_ambigous_edge_candidates_idx[time_mask]
                        # Do nothing if there is only one edge candidate for a certain timeframe
                        # Otherwise filter ambigous edges for timeframe_i
                        if time_mask.sum() > 1:
                            assert len(ambigous_node_idx_time_i) > 1 
                            decide_active_edge_between_ambiguous_edges(ambigous_node_idx_time_i,
                                active_connections, 
                                tracking_confidence_list, 
                                active_neighbors)
                # if all timeframes are different to eachother then skip this process
                # and continue with the next row
                else:
                    continue
                ##################################################################################
                # # 
                # # Get global node id
                # current_global_edge_id = isnotNan_mask_idx[local_row_index]
                # current_global_source_node_id = active_neighbors[current_global_edge_id] 
                # assert (current_global_source_node_id == active_neighbors[isnotNan_mask_idx[ambigous_node_idx]]).all
                # local_active_neighbor = local_active_neighbor_list[local_row_index]
                # assert current_global_source_node_id == local_active_neighbor
                # # Get all active edges
                # global_ambigous_edge_idx = isnotNan_mask_idx[ambigous_node_idx]
                # # Combine current and other ambigous edge_idx
                # all_global_ambigous_edge_idx = torch.cat([current_global_edge_id.unsqueeze(dim=1),global_ambigous_edge_idx], dim=0)
                # all_global_ambigous_edge_idx = all_global_ambigous_edge_idx.squeeze_()
                # ambigous_active_edges = active_connections[all_global_ambigous_edge_idx]
                # ambigous_tracking_confidence = tracking_confidence_list[all_global_ambigous_edge_idx]
            
                # # Perform softmax + argmax
                # log_probs = functional.log_softmax(ambigous_tracking_confidence, dim = 0)
                # winning_subsample_id = torch.argmax(log_probs, dim = 0)
                # winning_id_global = all_global_ambigous_edge_idx[winning_subsample_id]
                
                # # Set remaining active edge-candidates to inactive -> False
                # lossing_idx_global = all_global_ambigous_edge_idx[all_global_ambigous_edge_idx != winning_id_global]
                # active_connections[lossing_idx_global] = float('nan') 
                # tracking_confidence_list[lossing_idx_global] = float('nan')
    #########################################################################################
    # Assign the final output to the graph object
    mot_graph.graph_obj.active_edges = active_connections >=1 # Make into torch.ByteTensor
    mot_graph.graph_obj.tracking_confidence = tracking_confidence_list # torch.LongTensor
    mot_graph.graph_obj.active_neighbors = active_neighbors # torch.IntTensor

#TrackID = InstanceID
def assign_track_ids(graph_object:Graph, frames_per_graph:int, nuscenes_handle:NuScenes):
    '''
    assign track Ids if active_edges have been determined
    
    Returns:
    tracking_IDs : torch tensor, shape(num_nodes)
    tracking_ID_dict : Dict{} contains mapping from custom tracking Ids to the same number again.
                            len(Dict.keys) = num_tracking_IDs
                            later on it should be used to match with the official_tracking Ids (Instance Ids) 
    '''
    def init_new_tracks(selected_node_idx : torch.Tensor, 
                        tracking_IDs : torch.IntTensor, 
                        tracking_ID_dict: Dict[int, int]):
        '''
        give new track Ids to selected Nodes
        '''
        # Determine helping variables
        device = graph_object.device()
        common_tracking_dtype = tracking_IDs.dtype
        # Generate new tracking Ids depending on number of untracked nodes
        num_selected_nodes = selected_node_idx.shape[0]
        last_id = max([key_tracking_id for key_tracking_id in tracking_ID_dict])
        assert last_id == torch.max(tracking_IDs)
        new_start = last_id + 1
        new_tracking_IDs:torch.IntTensor = torch.arange(start= new_start, end= new_start + num_selected_nodes,
                                step=1, dtype= common_tracking_dtype, device=device)
        # assign new tracking ids to untracked nodes
        tracking_IDs[selected_node_idx] = new_tracking_IDs
        # Update dictionary
        update_tracking_dict(new_tracking_IDs.tolist(), tracking_ID_dict)

    def update_tracking_dict(new_tracking_ids :List[int],
                                tracking_ID_dict: Dict[int, int]):
        tracking_ID_dict.update({track_id : track_id for track_id in new_tracking_ids})

    common_tracking_dtype = torch.int
    device = graph_object.device()
    tracking_IDs:torch.Tensor = torch.ones(graph_object.x.shape[0], dtype=common_tracking_dtype).to(device)
    tracking_IDs =  tracking_IDs * UNTRACKED_ID

    # tracking_confidence_by_node_id:torch.Tensor = torch.zeros(graph_object.x.shape[0],dtype=torch.float32).to(device)
    tracking_confidence_by_node_id:torch.Tensor = torch.ones(graph_object.x.shape[0],dtype=torch.float32).to(device)
    tracking_ID_dict: Dict[int, int]= {}

    edge_indices = graph_object.edge_index

    # Go forward frame by frame, start at first frame and stop at last frame 
    for timeframe in range(frames_per_graph):
        node_idx_from_frame_i = get_node_indices_from_timeframe_i(graph_object.timeframe_number, timeframe=timeframe, as_tuple=True)
        node_idx_from_frame_i = node_idx_from_frame_i[0] # second colum is just zeros

        # If first frame Assing new tracking Ids
        if timeframe == 0:
            num_initial_boxes = node_idx_from_frame_i.shape[0]
            initial_tracking_IDs:torch.IntTensor = \
                            torch.arange(start=0, end= num_initial_boxes,
                            step=1, 
                            dtype= common_tracking_dtype,
                            device=device)
            tracking_IDs[node_idx_from_frame_i] = initial_tracking_IDs

            update_tracking_dict(initial_tracking_IDs.tolist(), tracking_ID_dict)
        # For all remaining frames 
        else:
            # check for active edges
            # Get active edge for edge selected node
            # Get all incoming edges 
            untracked_node_idx = []
            for target_node_id in node_idx_from_frame_i:
                
                rows, columns = edge_indices[0],edge_indices[1] # COO-format
                edge_mask:torch.Tensor = torch.eq(columns, target_node_id)
                incoming_edge_idx_past_and_future = edge_mask.nonzero().squeeze()
                temporal_incoming_edge_idx = filter_out_spatial_edges(incoming_edge_idx_past_and_future, graph_object)
                if temporal_incoming_edge_idx.shape[0] == 0:
                    # Found a node without temporal edges
                    # it will be considered as a node without any active connections and therefore will be assigned a new trackId
                    # continue with next node
                    print("Found node without temporal edges!")
                    
                    assert graph_object.active_edges[temporal_incoming_edge_idx].any() == False
                incoming_edge_idx_past = filter_for_past_edges(temporal_incoming_edge_idx, 
                                                                timeframe, frames_per_graph, graph_object)

                current_active_edges = graph_object.active_edges[incoming_edge_idx_past]

                #### DEBUGGING ###########
                timeframes:torch.Tensor = graph_object.timeframe_number
                ###################

                # check if active edge available
                if current_active_edges.any():
                    assert current_active_edges.sum(dim=0) == 1
                    # active_edge_id_local = torch.argmax( torch.tensor(current_active_edges,dtype=torch.float32) )
                    # active_edge_id = incoming_edge_idx_past_and_future[current_active_edges]
                    active_edge_id = incoming_edge_idx_past[current_active_edges]
                    # active_edge_id = incoming_edge_idx_past_and_future[active_edge_id_local]
                    source_node_id =  rows[active_edge_id]

                    #### DEBUGGING #######
                    source_node_time = timeframes[source_node_id]
                    target_node_time = timeframes[target_node_id]
                    assert source_node_time != target_node_time
                    ###################

                    assert graph_object.active_neighbors[active_edge_id] == source_node_id
                    # check if source node has an assigned track id
                    if(tracking_IDs[source_node_id] == UNTRACKED_ID):
                        print(tracking_IDs[source_node_id])
                    assert tracking_IDs[source_node_id] != UNTRACKED_ID, \
                                    "Source node was not given a track ID !!!\n" \
                                    + " Tracking cannot be performed, either invalid active edge or invalid source_node_id!\n"\
                                    + "tracking_IDs[source_node_id]: {} \n".format(tracking_IDs[source_node_id])\
                                    +"Tracking_IDS: {} \n".format(tracking_IDs)
                    # adopt track id
                    tracking_IDs[target_node_id] = tracking_IDs[source_node_id]
                    tracking_confidence_by_node_id[target_node_id] = graph_object.tracking_confidence[active_edge_id]
                else: # add target_node_id to list of untracked nodes
                    untracked_node_idx.append(target_node_id)

            # Assign new track Ids to untracked nodes
            # check if list contains elements
            if untracked_node_idx:
                untracked_node_idx = torch.stack(untracked_node_idx, dim=0)
                init_new_tracks( untracked_node_idx, tracking_IDs, tracking_ID_dict)
                # tracking confidence of newly tracked objects is implicitly 0.0

        ######################################################################################################
        # Check that all nodes from timeframe i have been assigned a trackID
        if (tracking_IDs[node_idx_from_frame_i] == UNTRACKED_ID).any():
            print(tracking_IDs[node_idx_from_frame_i])
        assert (tracking_IDs[node_idx_from_frame_i] == UNTRACKED_ID).any() == False, \
                    "Some node has not been assigned a tracking ID at timeframe {}\n".format(timeframe) \
                    + "tracking_IDs[node_idx_from_frame_i] :{}\n".format(tracking_IDs[node_idx_from_frame_i])

    # check if all nodes have been assigned a tracking id
    assert torch.isnan(tracking_IDs).all() == False, "Not all nodes have been assigned a tracking Id"

    return tracking_IDs, tracking_ID_dict, tracking_confidence_by_node_id

def get_considered_sample_tokens(available_sample_tokens:List[str], 
                                    starting_sample_token= None, 
                                    ending_sample_token =None):
    """
    """
    considered_sample_tokens:List[str] = []
    startAppending = False
    if starting_sample_token is None:
        starting_sample_token = available_sample_tokens[0]

    if ending_sample_token is None:
        ending_sample_token = available_sample_tokens[-1]


    for sample_token in available_sample_tokens:
        if sample_token == starting_sample_token:
            startAppending = True
        elif sample_token == ending_sample_token:
            startAppending = False
            considered_sample_tokens.append(sample_token)
        
        if startAppending:
            considered_sample_tokens.append(sample_token)

    return considered_sample_tokens

def add_tracked_boxes_to_submission(submission: Dict[str, Dict[str, Any]],
                                        mot_graph:NuscenesMotGraph,
                                        local2global_tracking_id_dict :Dict[int,int] = None,
                                        starting_sample_token:str= None,
                                        ending_sample_token:str= None,
                                        use_gt = False) -> Dict[str, Dict[str, Any]]:
    """
    Builds list of sample_results-dictionaries for a specific time-frame.
    Idea Mirror sample_annotation-table

    Args:
    starting_sample_token: [Included] Describes the first sample_frame from which the tracking is started. 
                Tracked objects appearing before that token are ignored and not submitted to the submission dict
    ending_sample_token: [Included] Describes the last sample_frame that is still considered for the tracking.
                Tracked objects appearing after that token are ignored and not submitted to the submission dict


    sample_results-dictionaries structure :
    sample_result {
        "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
        "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
        "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
        "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                        Note that the tracking_name cannot change throughout a track.
        "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                        We average over frame level scores to compute the track level score.
                                        The score is used to determine positive and negative tracks via thresholding.
    }
    """
    
    
    considered_sample_tokens = get_considered_sample_tokens(
                                        mot_graph.graph_dataframe["available_sample_tokens"],
                                        starting_sample_token,
                                        ending_sample_token)

    trackingBoxes_dict :Dict[str,List[Dict[str, Any]]] = {
                sample_tokens: [] for sample_tokens in considered_sample_tokens
                }
    first_node_id = 0
    last_node_id = len(mot_graph.graph_dataframe["boxes_list_all"]) - 1

    

    if starting_sample_token is not None:
        start_timeframe:int = mot_graph.graph_dataframe['available_sample_tokens'].index(starting_sample_token)
        t_frame_number: torch.Tensor = mot_graph.graph_obj.timeframe_number 
        start_node_idx:torch.Tensor = (t_frame_number == start_timeframe).squeeze().nonzero(as_tuple=True)[0]
        start_node_idx:List[int] = start_node_idx.tolist()
        first_node_id:int = start_node_idx[0]

    if ending_sample_token is not None:
        end_timeframe:int = mot_graph.graph_dataframe['available_sample_tokens'].index(ending_sample_token)
        t_frame_number: torch.Tensor = mot_graph.graph_obj.timeframe_number 
        end_node_idx:torch.Tensor = (t_frame_number == end_timeframe).squeeze().nonzero(as_tuple=True)[0]
        end_node_idx:List[int] = end_node_idx.tolist()
        last_node_id:int = end_node_idx[-1] 

    selected_node_idx = range(first_node_id, last_node_id + 1)

    for node_id in selected_node_idx:
        box:Box = mot_graph.graph_dataframe["boxes_list_all"][node_id]
        
        # transformed_box = box.copy()
        # transformed_box.rotate()

        # sample_token: str
        # translation: Tuple[float, float, float] = (0, 0, 0),
        # size: Tuple[float, float, float] = (0, 0, 0),
        # rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        # velocity: Tuple[float, float] = (0, 0),
        current_sample_token:str = mot_graph.graph_dataframe["sample_tokens"][node_id]

        size : List[float]  = box.wlh.tolist()
        translation: List[float] = None
        rotation: List[float] = None
        velocity : List[float]= [0.0, 0.0] # fixed velocity
        if use_gt:
            translation, orientation = get_gt_sample_annotation_pose(mot_graph.nuscenes_handle,box.token)
        else:
            translation: List[float] = box.center.tolist() # LIDAR FRAME
            orientation: Quaternion = box.orientation # LIDAR FRAME
            translation, orientation = transform_detections_lidar2world_frame(mot_graph.nuscenes_handle, 
                            translation, orientation, current_sample_token,
                            sample_annotation_token = box.token)

        rotation: List[float] = [orientation.w, orientation.x, orientation.y, orientation.z]
        
        

        # tracking_id: str = '',  # Instance id of this object.
        # tracking_name: str = '',  # The class name used in the tracking challenge.
        # tracking_score: float = -1.0): 
        # TRACKING_ID
        tracking_id_local:torch.Tensor = mot_graph.graph_obj.tracking_IDs[node_id]
        tracking_id :int = tracking_id_local.squeeze().tolist()
        if local2global_tracking_id_dict:
            tracking_id_global:int = local2global_tracking_id_dict[tracking_id]
            tracking_id :int = tracking_id_global
        tracking_id : str = str(tracking_id)

        # TRACKING NAME = CLASS --> transform(CLASS_ID)
        class_id:torch.Tensor = mot_graph.graph_dataframe["class_ids"][node_id].squeeze()
        class_id :int = class_id.tolist()
        tracking_name : str = name_from_id(class_id= class_id) # name_from_id(instance.class_id)
        # TRACKING SCORE = Confidence for detection
        tracking_score:torch.Tensor = mot_graph.graph_obj.tracking_confidence_by_node_id[node_id]
        tracking_score : float = tracking_score.squeeze().tolist() # confidence 

        tracking_box_dict: Dict[str, Any] = build_results_dict(current_sample_token, translation,
                                size, 
                                rotation, velocity,
                                tracking_id, tracking_name,
                                tracking_score)


        trackingBoxes_dict[current_sample_token].append(tracking_box_dict)
    
    assert len(trackingBoxes_dict) <= mot_graph.max_frame_dist, "tracking boxes are assigned to more sample_tokens then there are frames per graph"

    for current_sample_token in trackingBoxes_dict:
        trackingBoxes:List[Dict[str, Any]] = trackingBoxes_dict[current_sample_token]
        add_results_to_submit(submission, frame_token=current_sample_token, predicted_instance_dicts= trackingBoxes)
        
    return submission


def prepare_for_submission(submission: Dict[str, Dict[str, Any]]):
    '''
     submission {
        "meta": {
            "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
            "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
            "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
            "use_map":      <bool>  -- Whether this submission uses map data as an input.
            "use_external": <bool>  -- Whether this submission uses external data as an input.
        },
        "results": {
            sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
        }
        }
    '''
    submission = {
        "meta": {
            "use_camera":  False,
            "use_lidar":   True,
            "use_radar":   False,
            "use_map":     False,
            "use_external": False
            },
        "results": {
           }
        }
    return submission
    

class MOTMetricsLogger(Callback):
    """
    Callback to compute MOT Validation metrics during training
    IF unfinished, complete it with compute_nuscenes_3D_mot_metrics()
    """
    def __init__(self, compute_oracle_results):
        super(MOTMetricsLogger).__init__()
        self.compute_oracle_results = compute_oracle_results

    # def _compute_mot_metrics(self, epoch_num, pl_module, oracle_results = False):
    #     constr_satisf_rate = pl_module.track_all_seqs(dataset=self.dataset,
    #                                                   output_files_dir=self.output_files_dir,
    #                                                   use_gt=oracle_results)

    #     # Compute MOT Metrics
    #     mot_metrics_summary = compute_mot_metrics(gt_path=osp.join(DATA_PATH, 'MOT_eval_gt'),
    #                                               out_mot_files_path=self.output_files_dir,
    #                                               seqs=self.dataset.seq_names)
    #     mot_metrics_summary['constr_sr'] = constr_satisf_rate
    #     mot_metrics_summary['epoch_num'] = epoch_num + 1

    #     return mot_metrics_summary

    # def on_train_start(self, trainer, pl_module):
    #     self.available_data = len(trainer.val_dataloaders) > 0 and len(trainer.val_dataloaders[0]) > 0
    #     if self.available_data:
    #         self.dataset = trainer.val_dataloaders[0].dataset
    #         # Determine the path in which MOT results will be stored
    #         if trainer.logger is not None:
    #             save_dir = osp.join(trainer.logger.save_dir, trainer.logger.name, trainer.logger.version )

    #         else:
    #             save_dir = trainer.default_save_path

    #         self.output_files_dir = osp.join(save_dir, 'mot_files')
    #         self.output_metrics_dir = osp.join(save_dir, 'mot_metrics')
    #         os.makedirs(self.output_metrics_dir, exist_ok=True)

    #     # Compute oracle results if needed
    #     if self.available_data and self.compute_oracle_results:
    #         mot_metrics_summary = self._compute_mot_metrics(trainer.current_epoch, pl_module, oracle_results=True)
    #         print(mot_metrics_summary)
    #         oracle_path = osp.join(self.output_metrics_dir, 'oracle.npy')
    #         save_pickle(mot_metrics_summary.to_dict(), oracle_path)
    #         trainer.oracle_metrics = mot_metrics_summary

    # def on_epoch_end(self, trainer, pl_module):
    #     # Compute MOT metrics on validation data, save them and log them
    #     if self.available_data:
    #         mot_metrics_summary = self._compute_mot_metrics(trainer.current_epoch, pl_module, oracle_results=False)
    #         metrics_path = osp.join(self.output_metrics_dir, f'epoch_{trainer.current_epoch + 1:03}.npy')
    #         save_pickle(mot_metrics_summary.to_dict(), metrics_path)

    #         if self.compute_oracle_results:
    #             for metric in pl_module.hparams['eval_params']['mot_metrics_to_norm']:
    #                 mot_metrics_summary['norm_' + metric] = mot_metrics_summary[metric] / trainer.oracle_metrics[metric]

    #         if pl_module.logger is not None and hasattr(pl_module.logger, 'experiment'):
    #             metric_names = pl_module.hparams['eval_params']['mot_metrics_to_log']
    #             if pl_module.hparams['eval_params']['log_per_seq_metrics']:
    #                 metrics_log ={f'{metric}/val/{seq}': met_dict[seq] for metric, met_dict in mot_metrics_summary.items() for seq in
    #                               list(self.dataset.seq_names) + ['OVERALL'] if metric in metric_names}

    #             else:
    #                 metrics_log ={f'{metric}/val': met_dict['OVERALL'] for metric, met_dict in mot_metrics_summary.items()
    #                               if metric in metric_names}
    #                 pl_module.logger.log_metrics(metrics_log, step = trainer.global_step)

