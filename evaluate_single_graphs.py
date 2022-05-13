'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import sacred
from sacred import Experiment

from utils.misc import make_deterministic, get_run_str_and_save_dir

import os.path as osp

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from pl_module.pl_module import MOTNeuralSolver
from utils.evaluation import compute_nuscenes_3D_mot_metrics

# for NuScenes eval
# from nuscenes.eval.common.config import config_factory
# from nuscenes.eval.tracking.evaluate import TrackingEval

from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset

# Pytorch Lightning Callback For Trainig 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils.misc import ModelCheckpoint as ModelCheckpointCustom

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config('configs/evaluate_mini_tracking_single_graphs_cfg.yaml')
# ex.add_config({'run_id': 'evaluation',
#                'add_date': True})

@ex.automain
def main(_config, _run):

    ###############
    #sacred.commands.print_config(_run) # No need to print config, as it's overwritten by the one from the ckpt.
    make_deterministic(12345)

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], None, _config['add_date'])
    out_files_dir = osp.join(save_dir, 'mot_files')
    ###############
    # Load model from checkpoint and update config entries that may vary from the ones used in training
    checkpoint_path = _config['ckpt_path'] \
                        if osp.exists(_config['ckpt_path'])  \
                        else osp.join(_config['output_path'], _config['ckpt_path'])
    model = MOTNeuralSolver.load_from_checkpoint(checkpoint_path = checkpoint_path)
    model.to(_config['gpu_settings']['torch_device'])
    model.hparams.update({
                        'output_path' : _config['output_path'],
                        'ckpt_path' : _config['ckpt_path'],
                        'eval_params': _config['eval_params'],
                        'data_splits': _config['data_splits'],
                        'test_dataset_mode': _config['test_dataset_mode'],
                        'dataset_params': _config['dataset_params'],
                        'gpu_settings' : _config['gpu_settings']
                        })

    # #######################################################
    # # Determine Logger
    # if _config['train_params']['tensorboard']:
    #     logger = TensorBoardLogger(_config['output_path'], name='experiments', version=run_str)
    # else:
    #     logger = None
    # #############################################
    # # Load test or validation dataset
    test_dataset = NuscenesMOTGraphDataset(_config['dataset_params'],
                                            mode = _config['test_dataset_mode'], 
                                            device=_config['gpu_settings']['torch_device'])
    # test_loader = DataLoader(test_dataset,
    #                             batch_size = _config['eval_params']['batch_size'],
    #                             shuffle= False,
    #                             num_workers=_config['eval_params']['num_workers'])
    # #############################################
    # # Define Trainers
    # accelerator = _config['gpu_settings']['device_type']
    # devices = _config['gpu_settings']['device_id']
    # trainer = Trainer(gpus=devices,
    #                 logger =logger,
    #                 default_root_dir= osp.join(_config['output_path'], 'evalutation_single_graphs', run_str)
    #                 )

    ###############
    # Get output MOT results files 
    output_files_dir = osp.join(_config['output_path'], 'evalutation_single_graphs', run_str)

    model.track_single_graphs(output_files_dir,
                        test_dataset, 
                        use_gt = False, 
                        verbose = True)
    # trainer.test(model=model, 
    #                 dataloaders=test_loader)

