'''
Taken and adapted from https://github.com/aleksandrkim61/EagerMOT
Check out the corresponding Paper https://arxiv.org/abs/2104.14682
This is serves as inspiration for our own code
'''
from typing import  List
import numpy as np

import torch

from datasets.mot_graph import Graph
from utils import graph

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

    def _predict_edges(self, subgraph):
        """

        """
        # Prune graph edges
        knn_mask = get_knn_mask(pwise_dist=subgraph.reid_emb_dists, edge_ixs=subgraph.edge_index,
                                num_nodes=subgraph.num_nodes, top_k_nns=self.dataset_params['top_k_nns'],
                                use_cuda=True, reciprocal_k_nns=self.dataset_params['reciprocal_k_nns'],
                                symmetric_edges=True)
        subgraph.edge_index = subgraph.edge_index.T[knn_mask].T
        subgraph.edge_attr = subgraph.edge_attr[knn_mask]
        if hasattr(subgraph, 'edge_labels'):
            subgraph.edge_labels = subgraph.edge_labels[knn_mask]

        # Predict active edges
        if self.use_gt:  # For debugging purposes and obtaining oracle results
            pruned_edge_preds = subgraph.edge_labels

        else:
            with torch.no_grad():
                pruned_edge_preds = torch.sigmoid(self.graph_model(subgraph)['classified_edges'][-1].view(-1))

        edge_preds = torch.zeros(knn_mask.shape[0]).to(pruned_edge_preds.device)
        edge_preds[knn_mask] = pruned_edge_preds

        if self.eval_params['set_pruned_edges_to_inactive']:
            return edge_preds, torch.ones_like(knn_mask)

        else:
            return edge_preds, knn_mask  # In this case, pruning an edge counts as not predicting a value for it at all
            # However, if it is pruned for every batch, then it counts as inactive.

    def track(self, scene_table:List[dict], output_path=None):
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

        frames_per_graph = self.dataset.dataset_params['max_frame_dist']
        max_frame_dist = self.dataset.dataset_params['max_frame_dist']
        list_scene_sample_tuple = self.dataset.get_samples_from_one_scene(scene_token)

        edge_predictions = {}
        graphs = {}
        for scene_sample_tuple in list_scene_sample_tuple:
            t = time()
            # # Load the graph corresponding to the entire subsequence
            sample_token_current = scene_sample_tuple[1]
            mot_graph = self.dataset.get_from_frame_and_seq(
                                            scene_token ,
                                            sample_token_current ,
                                            return_full_object = True)
            # subseq_graph = self._load_full_seq_graph_object(seq_name, start_frame if start_frame == frame_cutpoints[0] else start_frame - max_frame_dist,
            #                                                 end_frame, max_frame_dist)

            # # Feed graph through MPN in batches
            # Predict active edges
            edge_preds = None
            if self.use_gt:  # For debugging purposes and obtaining oracle results
                edge_preds = mot_graph.graph_obj.edge_labels

            else:
                with torch.no_grad():
                    edge_preds = torch.sigmoid(self.graph_model(mot_graph.graph_obj)['classified_edges'][-1].view(-1))
                    print(edge_preds)

            edge_predictions[sample_token_current] = edge_preds
            graphs[sample_token_current] = mot_graph 
            

            # subseq_graph = self._evaluate_graph_in_batches(subseq_graph, frames_per_graph)

            # # Round predictions and assign IDs to trajectories
            # subseq_graph = self._project_graph_model_output(subseq_graph)


            # del subseq_graph
            #print("Max Mem allocated:", torch.cuda.max_memory_allocated(torch.device('cuda:0'))/1e9)
            #print(f"Done with Sequence chunk, it took {time()-t}")

        # if output_path is not None:
        #     self._save_results_to_file(seq_df, output_path)

        return None