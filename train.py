'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import sacred
from sacred import Experiment

# from mot_neural_solver.utils.evaluation import MOTMetricsLogger
from utils.misc import make_deterministic, get_run_str_and_save_dir
from utils.misc import ModelCheckpoint as ModelCheckpointCustom

from utils.path_cfg import OUTPUT_PATH
import os.path as osp

from pl_module.pl_module import MOTNeuralSolver

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

###############
#For DATALOADER
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from torch_geometric.loader import DataLoader
##################

# Pytorch Lightning Callback For Trainig 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()

ex.add_config('configs/tracking_cfg.yaml')

# Config for naming record files
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
    # load weights from previous checkpoint if desired
    if _config['load_checkpoint']:
        model = MOTNeuralSolver.load_from_checkpoint(checkpoint_path=_config['ckpt_path'] \
                                                    if osp.exists(_config['ckpt_path'])  \
                                                    else osp.join(_config['output_path'], _config['ckpt_path']))

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], _config['cross_val_split'], _config['add_date'])

    # Define Logger
    if _config['train_params']['tensorboard']:
        logger = TensorBoardLogger(_config['output_path'], name='experiments', version=run_str)

    else:
        logger = None

    
    #########################################
    # Load Data 
    # nusc = NuScenes(version='v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes", verbose=True)
    # nusc = NuScenes(version='v1.0-trainval', dataroot='/media/HDD2/Datasets/mini_nusc', verbose=True)
    
    train_dataset = NuscenesMOTGraphDataset(_config['dataset_params'],
                                            mode =_config["train_dataset_mode"], 
                                            device=_config['gpu_settings']['torch_device'])

    train_loader = DataLoader(train_dataset,
                            batch_size = _config['train_params']['batch_size'],
                            shuffle= True,
                            num_workers=_config['train_params']['num_workers'])

    eval_dataset = NuscenesMOTGraphDataset(_config['dataset_params'],
                                    mode =_config["eval_dataset_mode"],
                                    nuscenes_handle = train_dataset.get_nuscenes_handle(),
                                    device =_config['gpu_settings']['torch_device'])

    eval_loader = DataLoader(eval_dataset,
                                batch_size = _config['train_params']['batch_size'],
                                shuffle= False,
                                num_workers=_config['train_params']['num_workers'])
    ###################################

    # Validation percentage check is deprecated in favour of limit_val_batches 
    # val_percent_check = _config['eval_params']['val_percent_check']
    # limit_val_batches = _config['eval_params']['val_percent_check']

    #nb_sanity_val_steps is deprecated

    # check_val_every_n_epoch=_config['eval_params']['check_val_every_n_epoch'] is deprecated
    # default_save_path=osp.join(OUTPUT_PATH, 'experiments', run_str) is deprecated
    
    # Set up Callbacks for training ################################
    callbacks = []
    # Essemtial Callbacks:
    # Learning rate monitor to log lr
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor_callback)

    # save model weights 
    # Save up to k best models 
    checkpoint_callback = ModelCheckpoint(dirpath=osp.join(_config['output_path'], 'experiments', run_str,"model_checkpoints"),
                                        save_top_k=_config['train_params']['num_save_top_k'],
                                        save_last = True,
                                        monitor="loss/val",
                                        mode = "min",
                                        every_n_epochs = 1 if _config['train_params']['save_every_epoch'] else 0,
                                        verbose = True)

    callbacks.append(checkpoint_callback)

    # Optional Callbacks:
    # Enable Early stopping
    if(_config['train_params']['include_early_stopping']):
        early_stop_callback = EarlyStopping(monitor="loss/val", min_delta=0.00, patience=5, verbose=False)
        callbacks.append(early_stop_callback)
    # save model weights 
    # Custom ModelCheckpoint Callback from Neural Solver-Paper
    if(_config["train_params"]["include_custom_checkpointing"]):
        ckpt_callback_custom = ModelCheckpointCustom(save_epoch_start = _config['train_params']['save_epoch_start'],
                                    save_every_epoch = _config['train_params']['save_every_epoch'])
        callbacks.append(ckpt_callback_custom)
    #############################################

    accelerator = _config['gpu_settings']['device_type']
    devices = _config['gpu_settings']['device_id']

    trainer = Trainer(gpus=devices,
                    callbacks=callbacks,
                    max_epochs=_config['train_params']['num_epochs'],
                    logger =logger,
                    default_root_dir= osp.join(_config['output_path'], 'experiments', run_str)
                    )

    
    trainer.fit(model,train_loader,eval_loader)

    # `skipped_batches_train`, `loss`, `log`, `skipped_batches_val`, `loss/val`, `val_loss_epochend`, `log_epochend`