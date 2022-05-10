'''
Taken and adapted from https://github.com/aleksandrkim61/EagerMOT
Check out the corresponding Paper https://arxiv.org/abs/2104.14682
This is serves as inspiration for our own code
'''

import numpy as np

import torch

from datasets.mot_graph import Graph

from utils.graph import get_knn_mask, to_undirected_graph, to_lightweight_graph

from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from model.mpn import MOTMPNet

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
        print("Initialized Tracker")

    def track(self, seq_name, output_path=None):
        """
        Main method. Given a sequence name, it tracks all detections and produces an output DataFrame, where each
        detection is assigned an ID.

        It starts loading a the graph corresponding to an entire video sequence and detections, then uses an MPN to
        sequentially evaluate batches of frames (i.e. subgraphs) and finally rounds predictions and applies
        postprocessing.

        """
        from time import time


        #print(f"Processing Seq {seq_name}")
        frames_per_graph = self._estimate_frames_per_graph(seq_name)
        max_frame_dist = self._estimate_max_frame_dist(seq_name, frames_per_graph)
        frame_cutpoints = self._determine_seq_cutpoints(seq_name)
        subseq_dfs = []
        constr_sr = 0
        for start_frame, end_frame in zip(frame_cutpoints[:-1], frame_cutpoints[1:]):
            t = time()

            # Load the graph corresponding to the entire subsequence
            subseq_graph = self._load_full_seq_graph_object(seq_name, start_frame if start_frame == frame_cutpoints[0] else start_frame - max_frame_dist,
                                                            end_frame, max_frame_dist)

            # Feed graph through MPN in batches
            subseq_graph = self._evaluate_graph_in_batches(subseq_graph, frames_per_graph)

            # Round predictions and assign IDs to trajectories
            subseq_graph = self._project_graph_model_output(subseq_graph)
            constr_sr += subseq_graph.constr_satisf_rate
            subseq_df = self._assign_ped_ids(subseq_graph)
            subseq_dfs.append(subseq_df)

            del subseq_graph
            #print("Max Mem allocated:", torch.cuda.max_memory_allocated(torch.device('cuda:0'))/1e9)
            #print(f"Done with Sequence chunk, it took {time()-t}")


        constr_sr /= (len(frame_cutpoints) - 1)
        seq_df = self._merge_subseq_dfs(subseq_dfs)
        # Postprocess trajectories
        if self.eval_params['add_tracktor_detects']:

            seq_df = self._add_tracktor_detects(seq_df, seq_name)


        # postprocess = Postprocessor(seq_df.copy(),
        #                             seq_info_dict=self.dataset.seq_info_dicts[seq_name],
        #                             eval_params=self.eval_params)

        seq_df = postprocess.postprocess_trajectories()


        if output_path is not None:
            self._save_results_to_file(seq_df, output_path)

        return seq_df, constr_sr