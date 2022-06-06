'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import sacred
from sacred import Experiment
from zmq import device

# from mot_neural_solver.utils.evaluation import MOTMetricsLogger
from utils.misc import make_deterministic, get_run_str_and_save_dir, ModelCheckpoint

from utils.path_cfg import OUTPUT_PATH
import os.path as osp

from pl_module.pl_module import MOTNeuralSolver

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import tqdm
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

###############
#For DATALOADER
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
##################

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()

ex.add_config('configs/testing_nuscenes_dataloader_w_centerpoint_det_ws.yaml')

# Config for naming record files
ex.add_config({'run_id': 'test_run_gpu1',
               'add_date': True,
               'cross_val_split': None})

@ex.config
def cfg( eval_params, dataset_params, graph_model_params, data_splits):

    # Make sure that the edges encoder MLP input dim. matches the number of edge features used.
    # graph_model_params['encoder_feats_dict']['edge_in_dim'] = len(dataset_params['edge_feats_to_use'])

    # If we're training on all the available training data, disable validation
    # if data_splits['train'] =='all_train' or data_splits['val'] is None:
    #     data_splits['val'] = []
    pass


@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)
    make_deterministic(12345)
    hparams_dict = dict(_config)

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], _config['cross_val_split'], _config['add_date'])
    
    #########################################
    # Load Data 
    train_dataset = NuscenesMOTGraphDataset(_config['dataset_params'], mode =_config["test_dataset_mode"],device=_config['gpu_settings']['torch_device'])
    train_loader = DataLoader(train_dataset,batch_size = 1)
    
    ###################################
    for i, batch in tqdm(enumerate(train_loader),total=len(train_dataset)):
        # if(i< 10):
        #     print("Train-Batch:",i)
        #     print(batch)
        # else:
        #     # break
        #     print("Train-Batch:",i)
        #     print(batch)
        if (batch.contains_dummies).any():
            print("batch number :",i )


