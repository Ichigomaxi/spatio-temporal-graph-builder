'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import os
import os.path as osp
from typing import Dict

import pandas as pd

from torch_geometric.data import DataLoader

import torch

from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.nn import functional as F

import pytorch_lightning as pl

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
    
    def forward(self, x):
        self.model(x)

    def load_model(self):

        model =  MOTMPNet(self.hparams['graph_model_params'])

        return model

    # def _get_data(self, mode, return_data_loader = True):
    #     assert mode in ('train', 'val', 'test')

    #     dataset = NuscenesMOTGraphDataset(dataset_params=self.hparams['dataset_params'],
    #                               mode=mode,
    #                               cnn_model=self.cnn_model,
    #                               splits= self.hparams['data_splits'][mode],
    #                               logger=None)

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
        # Define Balancing weight
        positive_vals = batch.edge_labels.sum()

        if positive_vals:
            pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals

        else: # If there are no positives labels, avoid dividing by zero
            pos_weight = 0

        # Compute Weighted BCE:
        loss = 0
        num_steps = len(outputs['classified_edges'])
        for step in range(num_steps):
            loss += F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1),
                                                            batch.edge_labels.view(-1),
                                                            pos_weight= pos_weight)
        return loss

    def _train_val_step(self, batch, batch_idx, train_val):
        device = (next(self.model.parameters())).device
        batch.to(device)

        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)
        logs = {**{'loss': loss}}
        log = {key + f'/{train_val}': val for key, val in logs.items()}

        if train_val == 'train':
            return {'loss': loss, 'log': log}

        else:
            return log

    def training_step(self, batch, batch_idx):
        return self._train_val_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        # self.log("val_loss", loss)
        return self._train_val_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs):
        metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
        metrics = {metric_name: torch.as_tensor(metric) for metric_name, metric in metrics.items()}
        return {'val_loss': metrics['loss/val'], 'log': metrics}
