'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import os
import os.path as osp
from typing import Dict
from cv2 import log

import pandas as pd

from torch_geometric.data import DataLoader

import torch

from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.nn import functional as F

import pytorch_lightning as pl
from datasets.NuscenesDataset import NuscenesDataset

from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from datasets.nuscenes_mot_graph import NuscenesMotGraph

from model.mpn import MOTMPNet
from utils.path_cfg import OUTPUT_PATH

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

            if positive_vals:
                pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals

            else: # If there are no positives labels, avoid dividing by zero
                pos_weight = torch.zeros(1)

            # Compute Weighted BCE:
            loss = 0
            num_steps = len(outputs['classified_edges'])
            for step in range(num_steps):
                loss_i = dummy_coeff * F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1),
                                                                batch.edge_labels.view(-1),
                                                                pos_weight= pos_weight)
                loss += loss_i
            return loss
        elif self.hparams['dataset_params']['label_type'] == "binary":
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
                self.skipped_batches_val += 1
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
