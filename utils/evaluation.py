'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
from tracemalloc import start
from turtle import st
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

import os
import os.path as osp

from copy import deepcopy
from collections import OrderedDict

import torch
from torch_scatter import scatter_add
from pytorch_lightning import Callback

import re

from zmq import device
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from datasets.mot_graph import Graph

from utils.misc import load_pickle, save_pickle

from torch.nn import functional

from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import Box
from nuscenes.eval.tracking.data_classes import TrackingConfig,TrackingBox
from nuscenes.eval.tracking.evaluate import TrackingEval
from datasets.nuscenes.reporting import add_results_to_submit,build_results_dict
from datasets.nuscenes.classes import name_from_id
from pyquaternion import Quaternion

###########################################################################
# MOT Metrics
###########################################################################

def compute_nuscenes_3D_mot_metrics(gt_path, out_mot_files_path, seqs, print_results = True):
    """

    Returns:
        Individual and overall MOTmetrics for all sequeces
    """
    # TrackingEval()
    summary = None 
    return summary

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

def assign_definitive_connections(mot_graph:NuscenesMotGraph):
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

    def filter_for_past_edges(incoming_edge_idx, current_timeframe:int):
        '''
        incoming_edge_idx : torch.tensor, shape(num_incoming_edges) # 
        current_timeframe: int
        '''
        past_edges = None
        # Determine Future timeframes
        max_possible_timeframe = (mot_graph.max_frame_dist-1)
        future_timeframes = range(max_possible_timeframe, current_timeframe , -1)
        # Last frame has no connections to past frames
        # Just continoue with given edges
        if max_possible_timeframe == current_timeframe:
            return incoming_edge_idx
        
        # Determine all future nodes indices
        future_node_idx = []
        for timeframe in future_timeframes: 
            future_nodes_from_frame_i_mask = mot_graph.graph_dataframe["timeframes_all"] == timeframe
            future_node_idx_from_frame_i = torch.nonzero(future_nodes_from_frame_i_mask, as_tuple=True)[0]
            future_node_idx.append(future_node_idx_from_frame_i)
        future_node_idx = torch.cat(future_node_idx) # concatenate

        #Check wich edges connect to future nodes
        incoming_edge_indices = mot_graph.graph_obj.edge_index[:,incoming_edge_idx] # shape: 2,num_incoming_edge_idx
        rows:torch.Tensor = incoming_edge_indices[0] 
        columns:torch.Tensor = incoming_edge_indices[1]
        test_tensor = columns.eq(future_node_idx.unsqueeze(dim=1)) # unqueeze to make it broadcastable
        test_tensor = test_tensor.sum(dim= 0)
        # test_if_self = (mot_graph.graph_obj.edge_index[0] == mot_graph.graph_obj.edge_index[1].unsqueeze(1))
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

    edge_indices = mot_graph.graph_obj.edge_index
    edge_preds = mot_graph.graph_obj.edge_preds 
    mot_graph.graph_obj.x

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
            incoming_edge_idx = []
            rows, columns = edge_indices[0],edge_indices[1] # COO-format
            edge_mask:torch.Tensor = torch.eq(columns, node_id)
            incoming_edge_idx = edge_mask.nonzero().squeeze()
            incoming_edge_idx = filter_for_past_edges(incoming_edge_idx, i)
            # Get edge predictions and provide the active neighbor and its tracking confindence
            incoming_edge_predictions = edge_preds[incoming_edge_idx]
            probabilities = functional.softmax(incoming_edge_predictions, dim=0 )
            log_probabilities = functional.log_softmax(incoming_edge_predictions, dim=0)
            local_index = torch.argmax(log_probabilities, dim = 0)
            assert local_index == torch.argmax(functional.softmax(incoming_edge_predictions, dim=0 ), dim = 0)
            
            # look for the global edge index
            global_index = incoming_edge_idx[local_index]
            active_edge_id = global_index

            # assert len(active_edge_id) == 1,"more than one active connection is ambigous!"

            tracking_confidence = probabilities[local_index] # tracking confidence for evaluation AMOTA-metric
            
            # set active edges
            active_edge = edge_indices[:,active_edge_id]
            assert rows[active_edge_id] == active_edge[0]
            active_connections[active_edge_id] = 1
            tracking_confidence_list[active_edge_id] = tracking_confidence
            active_neighbors[active_edge_id] = rows[active_edge_id]

    # Determine if any active source_node has more than 1 active edges assigned to them
    # check for duplicates in active_neighbors
    isNan_mask = active_neighbors !=active_neighbors
    isnotNan_mask = ~isNan_mask
    local_active_neighbor_list = active_neighbors[isnotNan_mask]
    comparison_matrix = local_active_neighbor_list.eq(local_active_neighbor_list.unsqueeze(dim=1))
    comparison_matrix.fill_diagonal_(False)
    num_ambigous_neighbors_per_source_node = comparison_matrix.sum(dim=0)
    is_ambigous_neighbors = num_ambigous_neighbors_per_source_node > 0

    # if(False):
    if(is_ambigous_neighbors.any()):
        comparison_matrix = comparison_matrix.triu()
        ambigous_source_node_local_idx = comparison_matrix.nonzero()
        isnotNan_mask_idx = isnotNan_mask.nonzero()
        ambigous_source_node_global_idx = isnotNan_mask_idx[ambigous_source_node_local_idx]
        # get corresponding source node and its active_edge_idx

        # filter duplicates(active edge candidates)
        for local_row_index in range(len(local_active_neighbor_list)):
            row_idx = ambigous_source_node_local_idx[:,0]
            column_idx = ambigous_source_node_local_idx[:,1]
            # Check if the current active_node has ambigous connections
            if local_row_index in row_idx:
                # Get index of ambigous 
                local_mask = row_idx == local_row_index
                ambigous_node_idx = column_idx[local_mask]
                # Get global node id
                current_global_edge_id = isnotNan_mask_idx[local_row_index]
                current_global_source_node_id = active_neighbors[current_global_edge_id] 
                assert (current_global_source_node_id == active_neighbors[isnotNan_mask_idx[ambigous_node_idx]]).all
                local_active_neighbor = local_active_neighbor_list[local_row_index]
                assert current_global_source_node_id == local_active_neighbor
                # Get all active edges
                global_ambigous_edge_idx = isnotNan_mask_idx[ambigous_node_idx]
                # Combine current and other ambigous edge_idx
                all_global_ambigous_edge_idx = torch.cat([current_global_edge_id.unsqueeze(dim=1),global_ambigous_edge_idx], dim=0)
                all_global_ambigous_edge_idx = all_global_ambigous_edge_idx.squeeze_()
                ambigous_active_edges = active_connections[all_global_ambigous_edge_idx]
                ambigous_tracking_confidence = tracking_confidence_list[all_global_ambigous_edge_idx]
            
                # Perform softmax + argmax
                log_probs = functional.log_softmax(ambigous_tracking_confidence, dim = 0)
                winning_subsample_id = torch.argmax(log_probs, dim = 0)
                winning_id_global = all_global_ambigous_edge_idx[winning_subsample_id]
                
                # Set remaining active edge-candidates to inactive -> False
                lossing_idx_global = all_global_ambigous_edge_idx[all_global_ambigous_edge_idx != winning_id_global]
                active_connections[lossing_idx_global] = float('nan') 
                tracking_confidence_list[lossing_idx_global] = float('nan')

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
                        tracking_IDs : torch.Tensor, 
                        tracking_ID_dict: Dict[torch.Tensor, torch.Tensor]):
        '''
        give new track Ids to selected Nodes
        '''
        # Determine helping variables
        device = graph_object.device()
        common_tracking_dtype = tracking_IDs.dtype
        # Generate new tracking Ids depending on number of untracked nodes
        num_selected_nodes = selected_node_idx.shape[0]
        last_id = max([key_tracking_id for key_tracking_id in tracking_ID_dict])
        nonNan_mask = ~(tracking_IDs!=tracking_IDs)
        assert last_id == torch.max(tracking_IDs[nonNan_mask])
        new_start = last_id + 1
        new_tracking_IDs = torch.arange(start= new_start, end= new_start + num_selected_nodes,
                                step=1, dtype= common_tracking_dtype, device=device)
        # assign new tracking ids to untracked nodes
        tracking_IDs[selected_node_idx] = new_tracking_IDs
        # Update dictionary
        update_tracking_dict(new_tracking_IDs, tracking_ID_dict)

    def update_tracking_dict(new_tracking_ids :torch.Tensor,
                                tracking_ID_dict: Dict[torch.Tensor, torch.Tensor]):
        tracking_ID_dict.update({track_id : track_id for track_id in new_tracking_ids})

    device = graph_object.device()

    tracking_IDs:torch.Tensor = torch.zeros(graph_object.x.shape[0]).to(device)
    tracking_IDs =  tracking_IDs * float('nan')
    common_tracking_dtype = tracking_IDs.dtype

    tracking_confidence_by_node_id:torch.Tensor = torch.zeros(graph_object.x.shape[0]).to(device)

    tracking_ID_dict: Dict[torch.Tensor, torch.Tensor]= {}

    edge_indices = graph_object.edge_index

    # Go forward frame by frame, start at first frame and stop at last frame 
    for timeframe in range(frames_per_graph):
        node_idx_from_frame_i = get_node_indices_from_timeframe_i(graph_object.timeframe_number, timeframe=timeframe, as_tuple=True)
        node_idx_from_frame_i = node_idx_from_frame_i[0] # second colum is just zeros

        # If first frame Assing new tracking Ids
        if timeframe == 0:
            num_initial_boxes = node_idx_from_frame_i.shape[0]
            initial_tracking_IDs = torch.arange(start=0, end= num_initial_boxes, step=1, dtype= common_tracking_dtype, device=device)
            tracking_IDs[node_idx_from_frame_i] = initial_tracking_IDs

            update_tracking_dict(initial_tracking_IDs, tracking_ID_dict)

        else:
            # check for active edges
            # Get active edge for edge selected node
            # Get all incoming edges 
            untracked_node_idx = []
            for target_node_id in node_idx_from_frame_i:
                
                rows, columns = edge_indices[0],edge_indices[1] # COO-format
                edge_mask:torch.Tensor = torch.eq(columns, target_node_id)
                incoming_edge_idx_past_and_future = edge_mask.nonzero().squeeze()
                current_active_edges = graph_object.active_edges[incoming_edge_idx_past_and_future]
                # check if active edge available
                if current_active_edges.any():
                    assert current_active_edges.sum(dim=0) == 1
                    # active_edge_id_local = torch.argmax( torch.tensor(current_active_edges,dtype=torch.float32) )
                    active_edge_id = incoming_edge_idx_past_and_future[current_active_edges]
                    # active_edge_id = incoming_edge_idx_past_and_future[active_edge_id_local]
                    source_node_id =  rows[active_edge_id]
                    
                    assert graph_object.active_neighbors[active_edge_id] == source_node_id
                    # check if source node has an assigned track id
                    assert tracking_IDs[source_node_id] != float('nan'), "Source node was not given a track ID !!! tracking cannot be performed, either invalid active edge"
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


    # check if all nodes have been assigned a tracking id
    assert torch.isnan(tracking_IDs).all() == False, "Not all nodes have been assigned a tracking Id"

    return tracking_IDs, tracking_ID_dict, tracking_confidence_by_node_id
    
def add_tracked_boxes_to_submission(submission: Dict[str, Dict[str, Any]],
                                        mot_graph:NuscenesMotGraph,
                                        local2global_tracking_id_dict :Dict[torch.Tensor,torch.Tensor] = None,
                                        starting_sample_token:str= None,
                                        ending_sample_token:str= None) -> Dict[str, Dict[str, Any]]:
    """
    Builds list of sample_results-dictionaries for a specific time-frame.
    Idea Mirror sample_annotation-table

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
    trackingBoxes_dict :Dict[str,List[Dict[str, Any]]] = {
                sample_tokens: [] for sample_tokens in mot_graph.graph_dataframe["available_sample_tokens"]
                }
    first_node_id = 0
    last_node_id = len(mot_graph.graph_dataframe["boxes_list_all"])

    

    if starting_sample_token is not None:
        start_timeframe:int = mot_graph.graph_dataframe['available_sample_tokens'].index(starting_sample_token)
        t_frame_number: torch.Tensor = mot_graph.graph_obj.timeframe_number 
        start_node_idx = (t_frame_number == start_timeframe).nonzero().squeeze()
        first_node_id = start_node_idx[0]

    if ending_sample_token is not None:
        end_timeframe:int = mot_graph.graph_dataframe['available_sample_tokens'].index(ending_sample_token)
        t_frame_number: torch.Tensor = mot_graph.graph_obj.timeframe_number 
        end_node_idx = (t_frame_number == end_timeframe).nonzero().squeeze()
        last_node_id = end_node_idx[-1]

    selected_node_idx = range(first_node_id, last_node_id + 1)

    for node_id in selected_node_idx:
        box:Box = mot_graph.graph_dataframe["boxes_list_all"][node_id]
        # sample_token: str
        # translation: Tuple[float, float, float] = (0, 0, 0),
        # size: Tuple[float, float, float] = (0, 0, 0),
        # rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        # velocity: Tuple[float, float] = (0, 0),
        current_sample_token:str = mot_graph.graph_dataframe["sample_tokens"][node_id]
        translation: List[float] = box.center.tolist()
        size : List[float]  = box.wlh.tolist()
        orientation: Quaternion = box.orientation
        rotation: List[float] = [orientation.w, orientation.x, orientation.y, orientation.z]
        velocity : List[float]= [0.0, 0.0] # fixed velocity

        # tracking_id: str = '',  # Instance id of this object.
        # tracking_name: str = '',  # The class name used in the tracking challenge.
        # tracking_score: float = -1.0): 
        
        tracking_id_local:torch.Tensor = mot_graph.graph_obj.tracking_IDs[node_id]
        tracking_id_float :float = tracking_id_local.squeeze().tolist()
        if local2global_tracking_id_dict:
            tracking_id_global = local2global_tracking_id_dict[tracking_id_local]
            tracking_id_float :float = tracking_id_global.squeeze().tolist()

        tracking_id_int :int = int(tracking_id_float)
        tracking_id : str = str(tracking_id_int) 
        class_id:torch.Tensor = mot_graph.graph_dataframe["class_ids"][node_id].squeeze()
        class_id :int = class_id.tolist()
        tracking_name : str = name_from_id(class_id= class_id) # name_from_id(instance.class_id)
        tracking_score:torch.Tensor = mot_graph.graph_obj.tracking_confidence_by_node_id[node_id]
        tracking_score : float = tracking_score.squeeze().tolist() # confidence 

        tracking_box_dict: Dict[str, Any] = build_results_dict(current_sample_token, translation,
                                size, 
                                rotation, velocity,
                                tracking_id, tracking_name,
                                tracking_score)


        trackingBoxes_dict[current_sample_token].append(tracking_box_dict)
    
    assert len(trackingBoxes_dict) == mot_graph.max_frame_dist, "tracking boxes are assigned to more sample_tokens then there are frames per graph"

    for current_sample_token in trackingBoxes_dict:
        trackingBoxes = trackingBoxes_dict[current_sample_token]
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
    

# def compute_mot_metrics(gt_path, out_mot_files_path, seqs, print_results = True):
#     """
#     The following code is adapted from
#     https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/apps/eval_motchallenge.py
#     It computes all MOT metrics from a set of output tracking files in MOTChallenge format
#     Args:
#         gt_path: path where MOT ground truth files are stored. Each gt file must be stored as
#         <SEQ NAME>/gt/gt.txt
#         out_mot_files_path: path where output files are stored. Each file must be named <SEQ NAME>.txt
#         seqs: Names of sequences to be evaluated

#     Returns:
#         Individual and overall MOTmetrics for all sequeces
#     """
#     def _compare_dataframes(gts, ts):
#         """Builds accumulator for each sequence."""
#         accs = []
#         names = []
#         for k, tsacc in ts.items():
#             if k in gts:
#                 accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
#                 names.append(k)

#         return accs, names

#     mm.lap.default_solver = 'lapsolver'
#     gtfiles = [os.path.join(gt_path, i, 'gt/gt.txt') for i in seqs]
#     tsfiles = [os.path.join(out_mot_files_path, '%s.txt' % i) for i in seqs]

#     gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
#     ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D')) for f in tsfiles])

#     mh = mm.metrics.create()
#     accs, names = _compare_dataframes(gt, ts)

#     # We will need additional metrics to compute IDF1, etc. from different splits inf CrossValidationEvaluator
#     summary = mh.compute_many(accs, names=names,
#                               metrics=mm.metrics.motchallenge_metrics + ['num_objects',
#                                                                          'idtp', 'idfn', 'idfp', 'num_predictions'],
#                               generate_overall=True)
#     if print_results:
#         print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

#     return summary

# class MOTMetricsLogger(Callback):
#     """
#     Callback to compute MOT Validation metrics during training
#     """
#     def __init__(self, compute_oracle_results):
#         super(MOTMetricsLogger).__init__()
#         self.compute_oracle_results = compute_oracle_results

#     def _compute_mot_metrics(self, epoch_num, pl_module, oracle_results = False):
#         constr_satisf_rate = pl_module.track_all_seqs(dataset=self.dataset,
#                                                       output_files_dir=self.output_files_dir,
#                                                       use_gt=oracle_results)

#         # Compute MOT Metrics
#         mot_metrics_summary = compute_mot_metrics(gt_path=osp.join(DATA_PATH, 'MOT_eval_gt'),
#                                                   out_mot_files_path=self.output_files_dir,
#                                                   seqs=self.dataset.seq_names)
#         mot_metrics_summary['constr_sr'] = constr_satisf_rate
#         mot_metrics_summary['epoch_num'] = epoch_num + 1

#         return mot_metrics_summary

#     def on_train_start(self, trainer, pl_module):
#         self.available_data = len(trainer.val_dataloaders) > 0 and len(trainer.val_dataloaders[0]) > 0
#         if self.available_data:
#             self.dataset = trainer.val_dataloaders[0].dataset
#             # Determine the path in which MOT results will be stored
#             if trainer.logger is not None:
#                 save_dir = osp.join(trainer.logger.save_dir, trainer.logger.name, trainer.logger.version )

#             else:
#                 save_dir = trainer.default_save_path

#             self.output_files_dir = osp.join(save_dir, 'mot_files')
#             self.output_metrics_dir = osp.join(save_dir, 'mot_metrics')
#             os.makedirs(self.output_metrics_dir, exist_ok=True)

#         # Compute oracle results if needed
#         if self.available_data and self.compute_oracle_results:
#             mot_metrics_summary = self._compute_mot_metrics(trainer.current_epoch, pl_module, oracle_results=True)
#             print(mot_metrics_summary)
#             oracle_path = osp.join(self.output_metrics_dir, 'oracle.npy')
#             save_pickle(mot_metrics_summary.to_dict(), oracle_path)
#             trainer.oracle_metrics = mot_metrics_summary

#     def on_epoch_end(self, trainer, pl_module):
#         # Compute MOT metrics on validation data, save them and log them
#         if self.available_data:
#             mot_metrics_summary = self._compute_mot_metrics(trainer.current_epoch, pl_module, oracle_results=False)
#             metrics_path = osp.join(self.output_metrics_dir, f'epoch_{trainer.current_epoch + 1:03}.npy')
#             save_pickle(mot_metrics_summary.to_dict(), metrics_path)

#             if self.compute_oracle_results:
#                 for metric in pl_module.hparams['eval_params']['mot_metrics_to_norm']:
#                     mot_metrics_summary['norm_' + metric] = mot_metrics_summary[metric] / trainer.oracle_metrics[metric]

#             if pl_module.logger is not None and hasattr(pl_module.logger, 'experiment'):
#                 metric_names = pl_module.hparams['eval_params']['mot_metrics_to_log']
#                 if pl_module.hparams['eval_params']['log_per_seq_metrics']:
#                     metrics_log ={f'{metric}/val/{seq}': met_dict[seq] for metric, met_dict in mot_metrics_summary.items() for seq in
#                                   list(self.dataset.seq_names) + ['OVERALL'] if metric in metric_names}

#                 else:
#                     metrics_log ={f'{metric}/val': met_dict['OVERALL'] for metric, met_dict in mot_metrics_summary.items()
#                                   if metric in metric_names}
#                     pl_module.logger.log_metrics(metrics_log, step = trainer.global_step)
