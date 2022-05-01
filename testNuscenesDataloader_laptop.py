'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import sacred
from sacred import Experiment

# from mot_neural_solver.utils.evaluation import MOTMetricsLogger
from utils.misc import make_deterministic, get_run_str_and_save_dir, ModelCheckpoint

from utils.path_cfg import OUTPUT_PATH
import os.path as osp

from pl_module.pl_module import MOTNeuralSolver

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

###############
#For DATALOADER
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm 
##################

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()

ex.add_config('configs/testing_nuscenes_dataloader_laptop.yaml')
# ex.add_config('configs/debug_config_file.yaml')

# # Config for naming record files
# ex.add_config({'run_id': 'train_w_default_config',
#                'add_date': True,
#                'cross_val_split': None})

@ex.config
def cfg( eval_params, dataset_params, graph_model_params, data_splits):

    # Make sure that the edges encoder MLP input dim. matches the number of edge features used.
    # graph_model_params['encoder_feats_dict']['edge_in_dim'] = len(dataset_params['edge_feats_to_use'])

    # If we're training on all the available training data, disable validation
    if data_splits['train'] =='all_train' or data_splits['val'] is None:
        data_splits['val'] = []


@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)
    make_deterministic(12345)
    hparams_dict = dict(_config)
    # pytorch lightning model
    model = MOTNeuralSolver(hparams = hparams_dict)

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], _config['cross_val_split'], _config['add_date'])

    if _config['train_params']['tensorboard']:
        logger = TensorBoardLogger(OUTPUT_PATH, name='experiments', version=run_str)

    else:
        logger = None

    ckpt_callback = ModelCheckpoint(save_epoch_start = _config['train_params']['save_epoch_start'],
                                    save_every_epoch = _config['train_params']['save_every_epoch'])
    
    #########################################
    # Load Data 
    # nusc = NuScenes(version='v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes", verbose=True)
    
    train_dataset = NuscenesMOTGraphDataset(_config['dataset_params'], mode ="mini_train")
    train_loader = DataLoader(train_dataset,batch_size = 2, num_workers=_config['train_params']['num_workers'])

    eval_dataset = NuscenesMOTGraphDataset(_config['dataset_params'], mode ="mini_val")
    eval_loader = DataLoader(eval_dataset,batch_size = 2, num_workers=_config['train_params']['num_workers'])
    
    ###################################
    # Uncomment for testing if loader works
    # for i, batch in enumerate(tqdm(train_loader)):
    #     if i < 4:
    #         print("Train batch", i, ":")
    #         print(batch)
    #     else:
    #         break

    # for i, batch in enumerate(tqdm(eval_loader)):
    #     if i < 4:
    #         print("Eval batch", i, ":")
    #         print(batch)
    #     else:
    #         break
    
    trainer = Trainer(callbacks=[ckpt_callback],
                    max_epochs=_config['train_params']['num_epochs'],
                    logger =logger,
                    )

    trainer.fit(model,train_loader,eval_loader)