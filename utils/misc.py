'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import os
import os.path as osp
import pickle
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml

from utils.path_cfg import OUTPUT_PATH


def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pickle(path):
    with open(path, 'rb') as file:
        ob = pickle.load(file)
    return ob

def save_pickle(ob, path):
    with open(path, 'wb') as file:
        pickle.dump(ob, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_run_str(run_id, cross_val_split, add_date):
    if cross_val_split is None:
        run_str = run_id
    else:
        run_str = run_id + f"_split_{cross_val_split}"

    if add_date:
        date = '{date:%m-%d_%H:%M}'.format(date=datetime.now())
        run_str = date + '_' + run_str

    return run_str

def get_run_str_and_save_dir(run_id, cross_val_split, add_date):
    run_str = get_run_str(run_id, cross_val_split,
                          add_date=add_date)
    unique_id_assert = f"Run ID string {run_str} already exists, try setting add_date_to_run_str=True"
    save_dir = osp.join(OUTPUT_PATH, 'experiments', run_str)

    assert not osp.exists(save_dir), unique_id_assert

    return run_str, save_dir

def save_model_hparams_and_sacred_config(out_files_dir:str, hparams:Dict[str,Any],_config:Dict[str,Any]):
    save_model_hparams(out_files_dir, hparams)
    save_sacred_config(out_files_dir, _config)

def save_sacred_config(out_files_dir:str,_config:Dict[str,Any]):
    os.makedirs(out_files_dir, exist_ok=True) # Make sure dir exists
    file_path = f"{out_files_dir}/config.yaml"
    print(f"Saving sacred config to file_path: {file_path}")
    with open(file_path, 'w') as file:
        documents = yaml.dump(_config, file)

def save_model_hparams(out_files_dir:str, hparams:Dict[str,Any]):
    os.makedirs(out_files_dir, exist_ok=True) # Make sure dir exists
    file_path = f"{out_files_dir}/hparams.yaml"
    print(f"Saving hparams to file_path: {file_path}")
    save_hparams_to_yaml(config_yaml=file_path, hparams=hparams)

class ModelCheckpoint(Callback):
    """
    Taken from https://github.com/dvl-tum/mot_neural_solver
    Check out the corresponding Paper https://arxiv.org/abs/1912.07515
    This is serves as inspiration for our own code

    Callback to allow saving models on every epoch, even if there's no validation loop
    """
    def __init__(self, save_epoch_start = 0, save_every_epoch=False):
        super(ModelCheckpoint, self).__init__()
        self.save_every_epoch = save_every_epoch
        self.save_epoch_start = save_epoch_start

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 >= self.save_epoch_start and self.save_every_epoch:
            filepath = osp.join(trainer.default_root_dir,'checkpoints', f"epoch_{trainer.current_epoch+1:03}.ckpt")
            os.makedirs(osp.dirname(filepath), exist_ok = True)
            trainer.save_checkpoint(filepath)
            print(f"Saving model at {filepath}")
