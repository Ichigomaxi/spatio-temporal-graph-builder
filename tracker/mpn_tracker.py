'''
Taken and adapted from https://github.com/aleksandrkim61/EagerMOT
Check out the corresponding Paper https://arxiv.org/abs/2104.14682
This is serves as inspiration for our own code
'''
from typing import  Any, Dict, Iterable, List
from matplotlib.style import available
import numpy as np

import torch

from nuscenes.nuscenes import Box
from datasets.mot_graph import Graph
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from utils import graph

from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from model.mpn import MOTMPNet

from utils.evaluation import add_tracked_boxes_to_submission, assign_definitive_connections, assign_track_ids
from datasets.nuscenes.reporting import add_results_to_submit, build_results_dict
from utils.nuscenes_helper_functions import skip_sample_token, get_all_samples_from_scene
class MPNTracker:
    """
    Class used to track video sequences.

    See 'track'  method for an overview.
    """

    def __init__(self, dataset, graph_model, use_gt, eval_params=None,
                 dataset_params=None, logger=None):

        self.dataset = dataset
        self.use_gt = use_gt
        self.logger = logger

        self.eval_params = eval_params
        self.dataset_params = dataset_params

        self.graph_model = graph_model

        if self.graph_model is not None:
            self.graph_model.eval()


class NuscenesMPNTracker(MPNTracker):
    """
    Tracks the Nuscenes Sequences from validation and test set. 
    Does this based on the edge predictions of the consecutive graphs given the GNN-model
    Main method is self.track
    """
    def __init__(self, dataset:NuscenesMOTGraphDataset ,
                    graph_model:MOTMPNet, 
                    use_gt:bool, 
                    eval_params=None, dataset_params=None, logger=None):
        super().__init__(dataset, graph_model, use_gt, eval_params, dataset_params, logger)
        ###

        print("Initialized Nuscenes Tracker")

    def _predict_edges(self, graph_obj:Graph):
        """

        """
        # Predict active edges
        if self.use_gt:  # For debugging purposes and obtaining oracle results
            edge_preds = graph_obj.edge_labels

        else:
            with torch.no_grad():
                edge_preds = torch.sigmoid(self.graph_model(graph_obj)['classified_edges'][-1].view(-1))

        return edge_preds

    def _perform_tracking_for_mot_graph(self,mot_graph:NuscenesMotGraph,):

        dataset:NuscenesMOTGraphDataset = self.dataset
        # Compute active connections
        assign_definitive_connections(mot_graph)
        
        # Assign Tracks
        tracking_IDs, tracking_ID_dict, tracking_confidence_by_node_id = assign_track_ids(mot_graph.graph_obj, 
                    frames_per_graph = mot_graph.max_frame_dist, 
                    nuscenes_handle = dataset.nuscenes_handle)
        mot_graph.graph_obj.tracking_IDs = tracking_IDs
        mot_graph.graph_obj.tracking_confidence_by_node_id = tracking_confidence_by_node_id

        return tracking_ID_dict

    def _load_and_infere_mot_graph(self,scene_token:str, sample_token:str):
        dataset:NuscenesMOTGraphDataset = self.dataset
        mot_graph:NuscenesMotGraph = dataset.get_from_frame_and_seq(
                                                scene_token ,
                                                sample_token ,
                                                return_full_object = True)
        # Compute edge predictions
        edge_preds = self._predict_edges(mot_graph.graph_obj)
        mot_graph.graph_obj.edge_preds = edge_preds
        return mot_graph

    def _init_new_global_tracks(self, global_tracking_dict:Dict[int, str],
                                    new_tracking_id_dict:Dict[int, int],
                                    selected_local_tracking_ids:List[int],
                                    sample_token:str):
        '''
        give new track Ids to selected Nodes
        '''
        # Determine helping variables
        common_tracking_dtype = torch.int

        # Generate new tracking Ids depending on number of untracked nodes
        last_id = 0
        # if global_tracking_dict is not empty update it
        if global_tracking_dict:
            last_id = max([key_tracking_id for key_tracking_id in global_tracking_dict])

            new_start = last_id + 1
            num_selected_nodes = len(selected_local_tracking_ids)
            t_new_tracking_IDs:torch.IntTensor = torch.arange(start= new_start, 
                                    end= new_start + num_selected_nodes,
                                    step=1, dtype= common_tracking_dtype)
            new_tracking_IDs:List[int] = t_new_tracking_IDs.squeeze().tolist()
        # if global_tracking_dict empty just fill it with given tracking ids
        else:
            new_tracking_IDs:List[int] = selected_local_tracking_ids
        # Update global dictionary
        self._update_global_tracking_dict(new_tracking_IDs, global_tracking_dict, sample_token)
        # Update new_tracking_dict
        for i, local_tracking_id in enumerate(selected_local_tracking_ids):
            new_tracking_id_dict[local_tracking_id] = new_tracking_IDs[i]

    def _update_global_tracking_dict(self,
                        new_tracking_ids:List[int], 
                        tracking_ID_dict:Dict[int, str], 
                        sample_token:str):
        tracking_ID_dict.update({track_id : sample_token for track_id in new_tracking_ids})


    def _assign_global_tracking_ids(self, previous_mot_graph:NuscenesMotGraph, new_mot_graph:NuscenesMotGraph,
                            previous_tracking_id_dict:Dict[int,int] ,
                            new_tracking_id_dict:Dict[int, int] ,
                            global_tracking_dict:Dict[int, str],
                            concatenation_sample_token: str):
        '''
        '''
        
        # make sure both mot_graphs contain the same time_frame aka sample_token
        assert concatenation_sample_token in previous_mot_graph.graph_dataframe['available_sample_tokens']\
                and concatenation_sample_token in new_mot_graph.graph_dataframe['available_sample_tokens']

        # Get corresponding old_node_idx to new_node_idx from concatenation frame 
        old_available_tokens:List[str] = previous_mot_graph.graph_dataframe['available_sample_tokens']
        old_time_frame = old_available_tokens.index(concatenation_sample_token)
        t_old_frame_number: torch.Tensor = previous_mot_graph.graph_obj.timeframe_number 
        old_node_idx:torch.Tensor = (t_old_frame_number == old_time_frame).nonzero(as_tuple=True)[0]
        old_node_idx:List[int] = old_node_idx.tolist()

        new_available_tokens:List[str] = new_mot_graph.graph_dataframe['available_sample_tokens']
        new_time_frame = new_available_tokens.index(concatenation_sample_token)
        t_new_frame_number: torch.Tensor = new_mot_graph.graph_obj.timeframe_number 
        new_node_idx:torch.Tensor = (t_new_frame_number == new_time_frame).nonzero(as_tuple=True)[0]
        new_node_idx:List[int] = new_node_idx.tolist()
        
        assert len(old_node_idx) == len(new_node_idx), "Detection Mismatch! Number of detections for previous and current graph are not identical"
        box_i:Box = previous_mot_graph.graph_dataframe["boxes_list_all"][old_node_idx[0]]
        box_j:Box = new_mot_graph.graph_dataframe["boxes_list_all"][new_node_idx[0]]
        assert box_i == box_j
        
        # Assign global tracking_ids
        t_old_local_tracking_ids:torch.IntTensor = previous_mot_graph.graph_obj.tracking_IDs
        old_local_tracking_ids:List[int] = t_old_local_tracking_ids.tolist()
        t_new_local_tracking_ids:torch.IntTensor = new_mot_graph.graph_obj.tracking_IDs
        new_local_tracking_ids:List[int] = t_new_local_tracking_ids.tolist()

        changed_local_tracking_ids:List[int] = []
        for i in range(len(old_node_idx)):
            old_node_id = old_node_idx[i]
            new_node_id = new_node_idx[i]
            # Get old local tracking Id 
            old_local_tracking_id:int = old_local_tracking_ids[old_node_id]
            # Get current local tracking Id
            new_local_tracking_id:int = new_local_tracking_ids[new_node_id]
            # Get global tracking Id and update new tracking dict with it
            global_tracking_id:int = previous_tracking_id_dict[old_local_tracking_id]
            new_tracking_id_dict[new_local_tracking_id] = global_tracking_id
            
            changed_local_tracking_ids.append(new_local_tracking_id)

        # Update global_tracking_dict. 
        # Add the new global tracking ids for new tracks initialized in new_mot_graph 
        unchanged_local_tracking_ids :List[int] = [new_local_tracking_id 
                                for new_local_tracking_id in new_tracking_id_dict
                                    if  new_local_tracking_id not in changed_local_tracking_ids]

        all_tracking_ids = [track_id for track_id in global_tracking_dict]
        largest_track_id = max(all_tracking_ids)
        self._init_new_global_tracks(global_tracking_dict, new_tracking_id_dict,
                                        unchanged_local_tracking_ids, new_mot_graph.start_frame)

    def _add_tracked_boxes_to_submission(self, 
                                        submission:Dict[str, Dict[str, Any]],
                                        mot_graph:NuscenesMotGraph,
                                        local2global_tracking_id_dict:Dict[torch.Tensor,torch.Tensor],
                                        starting_sample_token:str = None,
                                        ending_sample_token:str = None):
        '''
        '''
        assert starting_sample_token not in submission["results"], submission["results"][starting_sample_token]
        assert ending_sample_token not in submission["results"], submission["results"][ending_sample_token]

        add_tracked_boxes_to_submission(submission,
                                        mot_graph,
                                        local2global_tracking_id_dict,
                                        starting_sample_token,
                                        ending_sample_token)

    def track(self, scene_table:List[dict], submission: Dict[str, Dict[str, Any]]):
        """
        Main method. Given a sequence name, it tracks all detections and produces an output DataFrame, where each
        detection is assigned an ID.

        It starts loading a the graph corresponding to an entire video sequence and detections, then uses an MPN to
        sequentially evaluate batches of frames (i.e. subgraphs) and finally rounds predictions and applies
        postprocessing.

        """
        from time import time
        scene_token = scene_table['token']
        seq_name = scene_table['name']

        print(f"Processing Seq {seq_name}")

        dataset:NuscenesMOTGraphDataset = self.dataset
        frames_per_graph = dataset.dataset_params['max_frame_dist']
        filtered_list_scene_sample_tuple = dataset.get_filtered_samples_from_one_scene(scene_token)
        all_available_samples = get_all_samples_from_scene(scene_token, dataset.nuscenes_handle )

        current_sample_token = scene_table['first_sample_token']
        previous_mot_graph:NuscenesMotGraph = None
        previous_tracking_dict = {}
        global_tracking_dict: Dict[int:str] = {}
        while (current_sample_token != scene_table['last_sample_token']):
            t = time()
            ##############################################################################
            # check if sample_token is indexed by dataset. 
            # If not then add empty list to the summmary and skip to next sample_token
            if ( (scene_token,current_sample_token) not in dataset.seq_frame_ixs):

                ##############################################################################
                # check if current sample_token is within the last #frames_per_graph frames
                if (current_sample_token in [all_available_samples[-frames_per_graph:]]):
                    # if true we can concatenate it with the last available frame
                    _, last_sample_token = filtered_list_scene_sample_tuple[-1]
                    last_mot_graph = self._load_and_infere_mot_graph(scene_token , last_sample_token)
                    last_tracking_ID_dict = self._perform_tracking_for_mot_graph( last_mot_graph )
                    # Concatenate
                    #TODO
                    self._assign_global_tracking_ids(previous_mot_graph,last_mot_graph,
                                            previous_tracking_dict,
                                            last_tracking_ID_dict, global_tracking_dict,
                                            current_sample_token)
                    # Add Tracked boxes except the ones from current sample token
                    start_concat_sample_token = skip_sample_token(current_sample_token,0, dataset.nuscenes_handle)
                    self._add_tracked_boxes_to_submission(submission,
                                                            last_mot_graph,
                                                            last_tracking_ID_dict,
                                                            start_concat_sample_token)
                    ##########################
                    # Assign last sample_token. Should be the same as last frame of scene
                    next_sample_token = last_mot_graph.graph_dataframe['available_sample_tokens'][-1]
                    ###########################
                    # Assign previous variables
                    previous_mot_graph = last_mot_graph
                    previous_tracking_dict = last_tracking_ID_dict

                else:
                    add_results_to_submit(submission ,frame_token= current_sample_token, predicted_instance_dicts=[])
                    ##########################
                    # Go to next sample_token
                    next_sample_token = skip_sample_token(current_sample_token,0, dataset.nuscenes_handle)
                    ###########################
                    # Delete previous variables
                    previous_mot_graph = None
                    previous_tracking_dict = None
            else:
                ##############################################################################
                # Perform tracking on mot-graph for the next #frames_per_graph frames
                # Returns Tracking Ids and tracking_id_dict
                
                # Load the graph corresponding to the entire subsequence
                current_mot_graph = self._load_and_infere_mot_graph(scene_token ,current_sample_token)

                current_tracking_ID_dict = self._perform_tracking_for_mot_graph(current_mot_graph)

                ##############################################################################
                # Concatenate with previous Update new tracking dict
                if ( (previous_mot_graph is not None) 
                    and current_sample_token in previous_mot_graph.graph_dataframe['available_sample_tokens']):

                    # Token where previous and current graphs are supposed to be concatenated
                    previous_last_sample_token = previous_mot_graph.graph_dataframe['available_sample_tokens'][-1]
                    assert current_sample_token == previous_last_sample_token, \
                                        "Timeframe Mismatch! Last sample token from previous graph \
                                        and first sample token from current graph are not identical!" 
                    # Concatenate
                    # Update Dictionaries
                    #TODO
                    concatenation_token = current_sample_token
                    self._assign_global_tracking_ids(previous_mot_graph, current_mot_graph, 
                                                    previous_tracking_dict, current_tracking_ID_dict,
                                                    global_tracking_dict,
                                                    concatenation_token)

                    # Add tracked boxes to summary
                    # Add Tracked boxes except the ones from current sample token
                    start_concat_sample_token = skip_sample_token(current_sample_token,0, dataset.nuscenes_handle)
                    self._add_tracked_boxes_to_submission(submission, current_mot_graph,
                                                            current_tracking_ID_dict,
                                                            starting_sample_token = start_concat_sample_token)
                    
                # If there is not any previous mot_graph or the previous went out of scope for the current graph-"window"
                # open new tracks
                else:
                    #TODO
                    selected_local_tracking_ids = [ local_tracking_id for local_tracking_id in current_tracking_ID_dict]
                    self._init_new_global_tracks(global_tracking_dict, current_tracking_ID_dict, 
                                                    selected_local_tracking_ids, current_sample_token)

                    # Add tracked boxes to summary
                    self._add_tracked_boxes_to_submission(submission, current_mot_graph,
                                                            current_tracking_ID_dict)
                
                ##############################################################################
                # Save new_tracking_dict as old_tracking_dict and new_mot_graph as old_mot_graph
                previous_mot_graph = current_mot_graph
                previous_tracking_dict = current_tracking_ID_dict

                ##############################################################################
                # Go to next sample_token 
                next_sample_token = skip_sample_token(current_sample_token,
                                                    frames_per_graph - 2, 
                                                    dataset.nuscenes_handle)

            # Assign new current_sample_token
            current_sample_token = next_sample_token
            # print Information
            print("Max Mem allocated:", torch.cuda.max_memory_allocated(torch.device(self.dataset.device))/1e9)
            print(f"Done with Sequence chunk, it took {time()-t}")