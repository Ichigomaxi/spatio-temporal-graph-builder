'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import os
import os.path as osp
from pickletools import read_uint1
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
from tensorboard import summary
import torch
from cv2 import log
from datasets.nuscenes.reporting import save_to_json_file
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from datasets.NuscenesDataset import NuscenesDataset
from model.mpn import MOTMPNet
from torch import optim as optim_module
from torch.nn import functional as F
from torch.optim import lr_scheduler as lr_sched_module
from torch_geometric.data import DataLoader
from tracker.mpn_tracker import NuscenesMPNTracker
from utils.evaluation import (assign_definitive_connections, assign_track_ids,
                              add_tracked_boxes_to_submission, prepare_for_submission)
from utils.misc import save_pickle
from utils.path_cfg import OUTPUT_PATH
from visualization.visualize_graph import (visualize_eval_graph,
                                           visualize_geometry_list,visualize_output_graph)


class MOTNeuralSolver(pl.LightningModule):
    """
    Pytorch Lightning wrapper around the MPN defined in model/mpn.py.
    (see https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html)

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """
    def __init__(self, hparams:dict):
        super().__init__()

        
        # deprecated way to assign hparams instead use self.save_hyperparameters(hparams)
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        
        self.model = self.load_model()
        self.skipped_batches = 0
        self.skipped_batches_val = 0
        self.skipped_batches_test = 0
    
    def forward(self, x):
        self.model(x)

    def load_model(self):

        model =  MOTMPNet(self.hparams['graph_model_params'])

        return model

    # def _get_data(self, mode, return_data_loader = True):
    #     assert mode in NuscenesDataset.ALL_SPLITS

    #     dataset = NuscenesMOTGraphDataset(dataset_params=self.hparams['dataset_params'],
    #                               mode=mode,
    #                               splits= None,
    #                               logger= None)

    #     if return_data_loader and len(dataset) > 0:
    #         train_dataloader = DataLoader(dataset,
    #                                       batch_size = self.hparams['train_params']['batch_size'],
    #                                       shuffle = True if mode == 'train' else False,
    #                                       num_workers=self.hparams['train_params']['num_workers'])
    #         return train_dataloader
        
    #     elif return_data_loader and len(dataset) == 0:
    #         return []
        
    #     else:
    #         return dataset

    # def train_dataloader(self):
    #     return self._get_data(mode = 'train')

    # def val_dataloader(self):
    #     return self._get_data('val')

    # def test_dataset(self, return_data_loader=False):
    #     return self._get_data('test', return_data_loader = return_data_loader)

    def configure_optimizers(self):
        optim_class = getattr(optim_module, self.hparams['train_params']['optimizer']['type'])
        optimizer = optim_class(self.model.parameters(), **self.hparams['train_params']['optimizer']['args'])

        if self.hparams['train_params']['lr_scheduler']['type'] is not None:
            lr_sched_class = getattr(lr_sched_module, self.hparams['train_params']['lr_scheduler']['type'])
            lr_scheduler = lr_sched_class(optimizer, **self.hparams['train_params']['lr_scheduler']['args'])

            return [optimizer], [lr_scheduler]

        else:
            return optimizer

    def _compute_loss(self, outputs, batch):

        if self.hparams['dataset_params']['label_type'] == "binary":
            # Filter out graphs that contains dummies
            # Therefore return a zero to loss (that contains requires_grad)
            dummy_coeff = None
            if any(batch.contains_dummies):
                # print("batch.contains_dummies:\n" ,batch.contains_dummies)
                dummy_coeff = 0
                return torch.zeros(1,dtype=torch.float32, requires_grad=True)
            else:
                dummy_coeff = 1

            # Define Balancing weight
            
            positive_vals = batch.edge_labels.sum()

            if positive_vals and \
                (self.hparams['train_params']['loss_params']['weighted_loss'] == True):
                pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals

            else: # If there are no positives labels, avoid dividing by zero
                pos_weight = torch.zeros(1).to(self.device)

            # Compute Weighted BCE:
            loss = 0
            num_steps = len(outputs['classified_edges'])
            for step in range(num_steps):
                
                predicted_edges_step_i = outputs['classified_edges'][step].view(-1)
                edge_labels = batch.edge_labels.view(-1)
                # Filter for Faulty predictions        
                loss_i = dummy_coeff * F.binary_cross_entropy_with_logits(predicted_edges_step_i,
                                                                edge_labels,
                                                                pos_weight= pos_weight)
                loss += loss_i
            return loss
        elif self.hparams['dataset_params']['label_type'] == "multiclass":
            return torch.zeros(1,dtype=torch.float32, requires_grad=True)
            
        else: 
            return torch.zeros(1,dtype=torch.float32, requires_grad=True)

    def _train_val_step(self, batch, batch_idx, train_val):
        device = (next(self.model.parameters())).device
        batch.to(device)

        # compute output logits given input batch
        outputs = self.model(batch)
        # compute Loss
        loss = self._compute_loss(outputs, batch)

        # Skip batches by returning None and increase counter
        if any(batch.contains_dummies):
            if train_val == 'train':
                self.skipped_batches += 1
            else:
                self.skipped_batches_val += 1\
            # TODO remove this return?
            return None

        logs = {**{'loss': loss}}
        log = {key + f'/{train_val}': val for key, val in logs.items()}

        if train_val == 'train':
            return {'loss': loss, 'log': log}

        else:
            return log

    def training_step(self, batch, batch_idx):

        log_dict = self._train_val_step(batch, batch_idx, 'train')
        self.log('skipped_batches_train',self.skipped_batches)
        if log_dict is not None:
            self.log_dict(log_dict)

        return log_dict

    def validation_step(self, batch, batch_idx):
        log_dict = self._train_val_step(batch, batch_idx, 'val')
        self.log('skipped_batches_val',self.skipped_batches_val)
        if log_dict is not None:
            self.log_dict(log_dict)
        
        return log_dict

    def training_epoch_end(self, all_training_step_outputs) -> None:
        # Reset count
        self.skipped_batches = 0
        self.skipped_batches_val = 0

    def validation_epoch_end(self, outputs):
        metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
        metrics = {metric_name: torch.as_tensor(metric) for metric_name, metric in metrics.items()}
        log_dict = {'val_loss_epochend': metrics['loss/val'], 'log_epochend': metrics}
        self.log_dict(log_dict)
        return log_dict


    def inference_step(self,batch):
        # compute output logits given input batch
        logits = self.model(batch)['classified_edges'][-1].view(-1)
        # compute the probability of the edges belonging to class 1 == "active"/"Connecting" edge
        edge_preds = torch.sigmoid(logits)
        # probabilities = torch.sigmoid(logits_output)
        # Skip batches by returning None and increase counter
        # self.log('edge_preds', edge_preds)

        return edge_preds
    
    # def predict_step(self, batch, batch_idx):
    #     """
    #     Overwrite for inference
    #     """
    #     pass

    # def test_step(self, batch, batch_idx):
    #     """
    #     Overwrite for testing
    #     """
    #     batch.to(self.device)
    #     if any(batch.contains_dummies):
    #         self.skipped_batches_test += 1
    #         return None
    #     probabilities = self.inference_step(batch) 
    #     active_connections = assign_definitive_connections(batch)
        
    #     return {'probabilities': probabilities,
    #                 'active_connections':active_connections}

    # def test_epoch_end(self, outputs):
    #     self.skipped_batches_test = 0

    def track_single_graphs(self, output_files_dir,
                        dataset:NuscenesMOTGraphDataset, 
                        use_gt:bool = False, 
                        verbose:bool = False,
                        tracking_threshold:float = 0.5):
        """
        Used for Inference step and evaluation
        """
        seq_sample_list = dataset.seq_frame_ixs
        
        inferred_mot_graphs = []
        summaries = []
        num_mot_graph_in_dataset = len(seq_sample_list)
        # limit number for debugging purposes
        if self.hparams['eval_params']['debbuging_mode']:
            upper_bound = 5
            num_mot_graph_in_dataset = upper_bound
        
        for i in range(num_mot_graph_in_dataset):
            # scene_token, sample_token = seq_sample_list[i]
            # mot_graph = dataset.get_from_frame_and_seq(scene_token,sample_token,
            #         return_full_object=True,
            #         inference_mode=True)
            seq_name, start_frame = dataset.seq_frame_ixs[i]
            scene = dataset.nuscenes_handle.get("scene", seq_name)
            print("scene_token: ", scene["token"], "scene_name", scene["name"] )
            print("sample_token: ",start_frame)
            mot_graph = dataset.get_from_frame_and_seq(
                                        seq_name = seq_name,
                                        start_frame = start_frame,
                                        return_full_object=True,
                                        inference_mode=True)
             # Predict active edges
            if use_gt:  # For debugging purposes and obtaining oracle results
                edge_preds = mot_graph.graph_obj.edge_labels
            else:
                with torch.no_grad():
                    edge_preds = self.inference_step(mot_graph.graph_obj)
            mot_graph.graph_obj.edge_preds = edge_preds

            # Compute active connections
            assign_definitive_connections(mot_graph, tracking_threshold)
            
            # # Assign Tracks
            tracking_IDs, tracking_ID_dict, tracking_confidence_by_node_id = assign_track_ids(mot_graph.graph_obj, 
                        frames_per_graph = mot_graph.max_frame_dist, 
                        nuscenes_handle = dataset.nuscenes_handle)
            mot_graph.graph_obj.tracking_IDs = tracking_IDs
            mot_graph.graph_obj.tracking_confidence_by_node_id = tracking_confidence_by_node_id
            if use_gt:  # For debugging purposes and obtaining oracle results
                mot_graph.graph_obj.tracking_confidence_by_node_id = torch.ones_like(tracking_confidence_by_node_id)
            
            # # Prepare submission dict
            summary = {}
            summary = prepare_for_submission(summary)
            # # Build TrackingBox List for evaluation
            summary = add_tracked_boxes_to_submission(summary, mot_graph = mot_graph,use_gt=use_gt)
            
            if(self.hparams['eval_params']['visualize_graph']):
                geometry_list = visualize_eval_graph(mot_graph)
                visualize_geometry_list(geometry_list)
                geometry_list = visualize_output_graph(mot_graph)
                visualize_geometry_list(geometry_list)
            
            inferred_mot_graphs.append(mot_graph)
            summaries.append(summary)

        if (self.hparams['eval_params']['save_single_graph_submission']):
            for i, summary in enumerate(summaries):
                inferred_detections_path = osp.join(output_files_dir,str(i))
                os.makedirs(inferred_detections_path, exist_ok=True) # Make sure dir exists
                save_to_json_file(summary, folder_name= inferred_detections_path , 
                        version= self.hparams['test_dataset_mode'] )

        # save objects for visualization
        if(self.hparams['eval_params']['save_graphs']):
            os.makedirs(output_files_dir, exist_ok=True) # Make sure dir exists
            # Save in pickle format
            pickle_file_path = osp.join(output_files_dir,"inferred_mot_graphs.pkl")
            with open(pickle_file_path, 'wb') as f:
                torch.save(inferred_mot_graphs,f, pickle_protocol = 4)
        



    def track_all_seqs(self, output_files_dir,
                        dataset:NuscenesMOTGraphDataset, 
                        use_gt:bool = False, 
                        verbose:bool = False,
                        tracking_threshold: float = 0.5):
        """
        Used for Inference step and evaluation
        """

        # Initiallize some kind of Tracker object
        tracker = NuscenesMPNTracker(
                            dataset = dataset,
                            graph_model = self.model,
                            use_gt= use_gt,
                            eval_params = self.hparams['eval_params'],
                            dataset_params = self.hparams['dataset_params'],
                            tracking_threshold = tracking_threshold)

        # Prepare submission dict
        summary = {}
        summary = prepare_for_submission(summary)

        # Track detection sequence by sequence

        split = self.hparams["test_dataset_mode"]
        # sequence_names = dataset.nuscenes_dataset.splits_to_scene_names[split]
        scene_tables = dataset.seqs_to_retrieve
        for scene_table in scene_tables:
            seq_name = scene_table['name']
            if verbose:
                print("Tracking sequence ", seq_name)

            # computing tracking for a certain scene/sequence 
            # Return list of tracked bounding boxes as Trackbox-class
            tracker.track(scene_table, summary)
            
            # save_pickle()
            
            if verbose:
                print("Done! \n")

        if (self.hparams['eval_params']['save_submission']):
            os.makedirs(output_files_dir, exist_ok=True) # Make sure dir exists
            results_file = save_to_json_file(summary, folder_name= output_files_dir , 
                    version= self.hparams['test_dataset_mode'] )
        
        return summary, results_file
