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

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config('configs/tracking_cfg.yaml')
# Config for naming record files
ex.add_config({'run_id': 'train_w_default_config',
               'add_date': True,
               'cross_val_split': None})

@ex.config
def cfg( eval_params, dataset_params, graph_model_params, data_splits):

    # Make sure that the edges encoder MLP input dim. matches the number of edge features used.
    graph_model_params['encoder_feats_dict']['edge_in_dim'] = len(dataset_params['edge_feats_to_use'])

    # If we're training on all the available training data, disable validation
    if data_splits['train'] =='all_train' or data_splits['val'] is None:
        data_splits['val'] = []


@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)
    make_deterministic(12345)

    # pytorch lightning model
    model = MOTNeuralSolver(hparams = dict(_config))

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], _config['cross_val_split'], _config['add_date'])

    if _config['train_params']['tensorboard']:
        logger = TensorBoardLogger(OUTPUT_PATH, name='experiments', version=run_str)

    else:
        logger = None

    ckpt_callback = ModelCheckpoint(save_epoch_start = _config['train_params']['save_epoch_start'],
                                    save_every_epoch = _config['train_params']['save_every_epoch'])

    # Validation percentage check is deprecated in favour of limit_val_batches 
    # val_percent_check = _config['eval_params']['val_percent_check']
    # limit_val_batches = _config['eval_params']['val_percent_check']

    #nb_sanity_val_steps is deprecated
    
    trainer = Trainer(gpus=1,
                      callbacks=[ckpt_callback],
                      weights_summary = None,
                      checkpoint_callback=False,
                      max_epochs=_config['train_params']['num_epochs'],
                      check_val_every_n_epoch=_config['eval_params']['check_val_every_n_epoch'],
                      logger =logger,
                      default_save_path=osp.join(OUTPUT_PATH, 'experiments', run_str))
    trainer.fit(model)