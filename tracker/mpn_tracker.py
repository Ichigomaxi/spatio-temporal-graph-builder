'''
Taken and adapted from https://github.com/aleksandrkim61/EagerMOT
Check out the corresponding Paper https://arxiv.org/abs/2104.14682
This is serves as inspiration for our own code
'''
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import torch
from datasets.mot_graph import Graph
from datasets.nuscenes.reporting import (
    add_results_to_submit, build_results_dict,
    check_submission_for_missing_samples,
    insert_empty_lists_for_selected_frame_tokens,
    is_sample_token_in_submission_contained)
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from matplotlib.style import available
from model.mpn import MOTMPNet
from nuscenes.nuscenes import Box
from utils import graph
from utils.evaluation import (add_tracked_boxes_to_submission,
                              assign_definitive_connections, assign_definitive_connections_new, assign_track_ids,assign_track_ids_new)
from utils.nuscenes_helper_functions import (get_all_samples_from_scene,
                                             skip_sample_token)
from visualization.visualize_graph import (visualize_eval_graph,
                                           visualize_geometry_list,
                                           visualize_output_graph)


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
                    tracking_threshold:float,
                    eval_params=None, dataset_params=None, logger=None, ):
        super().__init__(dataset, graph_model, use_gt, eval_params, dataset_params, logger)
        ###
        self.tracking_threshold = tracking_threshold
        print("Initialized Nuscenes Tracker")

        if "use_gt_detections" in self.dataset_params:
            assert not((use_gt == True) and (self.dataset_params["use_gt_detections"]==False)),\
                "Incompatible configurations. Impossible to use ground truth edge labels while loading external detections"

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

    def _perform_tracking_for_mot_graph(self,mot_graph:NuscenesMotGraph):

        dataset:NuscenesMOTGraphDataset = self.dataset
        # Compute active connections
        # assign_definitive_connections(mot_graph, self.tracking_threshold)
        assign_definitive_connections_new(mot_graph, self.tracking_threshold)
        
        # Assign Tracks
        # tracking_IDs, tracking_ID_dict, tracking_confidence_by_node_id = assign_track_ids(mot_graph.graph_obj, 
        #             frames_per_graph = mot_graph.max_frame_dist, 
        #             nuscenes_handle = dataset.nuscenes_handle)
        tracking_IDs, tracking_ID_dict, tracking_confidence_by_node_id = assign_track_ids_new(mot_graph)
        mot_graph.graph_obj.tracking_IDs = tracking_IDs
        mot_graph.graph_obj.tracking_confidence_by_node_id = tracking_confidence_by_node_id
        # if self.use_gt:
        #     mot_graph.graph_obj.tracking_confidence_by_node_id = torch.ones_like(tracking_confidence_by_node_id)

        return tracking_ID_dict

    def _load_and_infere_mot_graph(self,scene_token:str, sample_token:str):
        dataset:NuscenesMOTGraphDataset = self.dataset
        
        inference_mode = not self.dataset_params["use_gt_detections"]

        mot_graph:NuscenesMotGraph = dataset.get_from_frame_and_seq(
                                                scene_token ,
                                                sample_token ,
                                                return_full_object = True,
                                                inference_mode=inference_mode)
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
        if not isinstance(new_tracking_IDs, list):
            print("TYPE:",type(new_tracking_IDs))
            new_tracking_IDs = [new_tracking_IDs]
        assert isinstance(new_tracking_IDs, list), "new_tracking_IDs is not a list!!\n It is {}".format(new_tracking_IDs)
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
                                        ending_sample_token:str = None,
                                        use_gt = False):
        '''
        '''
        assert starting_sample_token not in submission["results"], submission["results"][starting_sample_token]
        assert ending_sample_token not in submission["results"], submission["results"][ending_sample_token]

        add_tracked_boxes_to_submission(submission,
                                        mot_graph,
                                        local2global_tracking_id_dict,
                                        starting_sample_token,
                                        ending_sample_token,
                                        use_gt)

    def is_faulty_graph(self,current_mot_graph :NuscenesMotGraph):
        #Check that graph does not contain any dummy boxes
        if current_mot_graph.graph_obj.contains_dummies:
            print("Found Faulty graph with dummy object in tracker!")
            print("Add to unbuildable list !")
            return True
        # Check that graph edges are undirected
        if current_mot_graph.graph_obj.is_directed():
            print("Found Faulty graph with directed edges in tracker!")
            print("Add to unbuildable list !")
            return True
        # Check that temporal edges are able to be directed
        edge_indices = current_mot_graph.graph_obj.edge_index
        temporal_edges_mask = current_mot_graph.graph_obj.temporal_edges_mask
        temporal_edge_index = current_mot_graph.graph_obj.edge_index[:,temporal_edges_mask[:,0]]
        edge_indices = temporal_edge_index
        # Make edges undirected
        sorted_edges, _ = torch.sort(edge_indices, dim=0)
        undirected_edges, orig_indices = torch.unique(sorted_edges, return_inverse=True, dim=1)
        if not (sorted_edges.shape[1] == 2 * undirected_edges.shape[1]):
            print("Found Faulty graph with temporal edges that cannot be transformed to directed graph in tracker!")
            print("Some edges were not duplicated")
            return True

        return False

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
        frames_per_graph:int = dataset.dataset_params['max_frame_dist']
        filtered_list_scene_sample_tuple = dataset.get_filtered_samples_from_one_scene(scene_token)
        all_available_samples = get_all_samples_from_scene(scene_token, dataset.nuscenes_handle )

        current_sample_token = scene_table['first_sample_token']
        previous_mot_graph:NuscenesMotGraph = None
        previous_tracking_dict = {}
        global_tracking_dict: Dict[int:str] = {}
        potentially_missed_sample_token:Set[str] = set([])
        ################
        # Handle Situation were the current scene is not able to be reconstructed by the Dataset object
        # Therefore the dataset does not yield any graph object for this scene
        # Set the current_sample_token to the last sample_token to skip the while loop
        if not filtered_list_scene_sample_tuple:
            current_sample_token = scene_table['last_sample_token']
            potentially_missed_sample_token = set(all_available_samples)
        #####
        unbuildable_sample_tokens = []
        while (current_sample_token != scene_table['last_sample_token']):
            t = time()
            ##############################################################################
            # check if sample_token is indexed by dataset. 
            # If not then add empty list to the summmary and skip to next sample_token
            if ( (scene_token,current_sample_token) not in dataset.seq_frame_ixs 
                    or current_sample_token in unbuildable_sample_tokens):
                
                # Get list of all sample_tokens from last possible graph for this scene
                _, last_filtered_sample_token = filtered_list_scene_sample_tuple[-1]
                last_filtered_sample_tokens = []
                sample_token = last_filtered_sample_token
                for i in range(frames_per_graph):
                    last_filtered_sample_tokens.append(sample_token)
                    sample_token = skip_sample_token(sample_token,0, dataset.nuscenes_handle)

                ##############################################################################
                # Complement tracking by considering the very last token of the last buildable graph
                # do this by allowing overlap of 2 timeframes instead of one in 3 frames_per_graph circumstance
                # check if current sample_token is within the last #frames_per_graph frames
                # and if it still is within the considered frames of the last graph of the scene
                if (current_sample_token in all_available_samples[-frames_per_graph:]
                        and previous_mot_graph is not None
                        and current_sample_token == last_filtered_sample_tokens[1] 
                        and current_sample_token != last_filtered_sample_tokens[-1]
                        and current_sample_token not in unbuildable_sample_tokens):
                    # if true we can concatenate it with the last available frame
                    
                    last_mot_graph = self._load_and_infere_mot_graph(scene_token , last_filtered_sample_token)

                    if self.is_faulty_graph(last_mot_graph):
                        unbuildable_sample_tokens.append(current_sample_token)
                        continue

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
                                                            start_concat_sample_token,
                                                            use_gt=self.use_gt)

                    # assert is_sample_token_in_submission_contained(submission,current_sample_token),\
                    #         "last considered frame of dataset has not been considered, scene:{}, sample:{}".format(seq_name,current_sample_token)
                    ##########################
                    # Assign last sample_token. Should be the same as last frame of scene
                    next_sample_token = last_mot_graph.graph_dataframe['available_sample_tokens'][-1]
                    ###########################
                    # Assign previous variables
                    previous_mot_graph = last_mot_graph
                    previous_tracking_dict = last_tracking_ID_dict
                # If the sample is not part of a buildable graph skip to the next token
                else:
                    # log potentially missed sample_tokens
                    sample_token = current_sample_token
                    print("lost tracks")
                    for i in range(frames_per_graph):
                        potentially_missed_sample_token.update({sample_token})
                        sample_token = skip_sample_token(sample_token,0, dataset.nuscenes_handle)
                    # add_results_to_submit(submission ,frame_token= current_sample_token, predicted_instance_dicts=[])
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

                if self.is_faulty_graph(current_mot_graph):
                    unbuildable_sample_tokens.append(current_sample_token)
                    continue

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
                                                            starting_sample_token = start_concat_sample_token,
                                                            use_gt=self.use_gt)
                    
                # If there is not any previous mot_graph or the previous went out of scope for the current graph-"window"
                # open new tracks
                else:
                    #TODO
                    selected_local_tracking_ids = [ local_tracking_id for local_tracking_id in current_tracking_ID_dict]
                    self._init_new_global_tracks(global_tracking_dict, current_tracking_ID_dict, 
                                                    selected_local_tracking_ids, current_sample_token)

                    # Add tracked boxes to summary
                    self._add_tracked_boxes_to_submission(submission, current_mot_graph,
                                                            current_tracking_ID_dict,
                                                            use_gt=self.use_gt)
                
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
            if  "visualize_graph" in self.eval_params \
                and self.eval_params["visualize_graph"] \
                and previous_mot_graph is not None:
                geometry_list = visualize_eval_graph(previous_mot_graph)
                visualize_geometry_list(geometry_list)
                geometry_list = visualize_output_graph(previous_mot_graph)
                visualize_geometry_list(geometry_list)
        ##########################################
        # Complement Submission with missing sample_tokens 
        # add empty lists to submission
        
        missing_samples:Set[str] = check_submission_for_missing_samples(submission, potentially_missed_sample_token)
        if missing_samples:
            insert_empty_lists_for_selected_frame_tokens(submission, missing_samples)

        missing_samples:Set[str] = check_submission_for_missing_samples(submission, set(all_available_samples))
        if missing_samples:
            insert_empty_lists_for_selected_frame_tokens(submission, missing_samples)

        if hasattr(dataset.unused_scene_sample_tokens_dict,"after_tracking"):
            unknown_tokens:List[str] = dataset.unused_scene_sample_tokens_dict["after_tracking"]
            unknown_tokens.extend(unbuildable_sample_tokens)
            dataset.unused_scene_sample_tokens_dict["after_tracking"] = unknown_tokens
        else:
            dataset.unused_scene_sample_tokens_dict["unbuildable_after_tracking"] = unbuildable_sample_tokens
            dataset.unused_scene_sample_tokens_dict["potentially_missed_after_tracking"] = list(potentially_missed_sample_token)
        