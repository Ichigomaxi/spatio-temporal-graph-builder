
import pickle

import sacred
from sacred import Experiment

from utils.misc import make_deterministic, get_run_str, save_sacred_config

from utils.path_cfg import OUTPUT_PATH
import os.path as osp
import os

###############
#For DATALOADER
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset


from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()

ex.add_config('configs/proprocess_dataset_nuscenes.yaml')

# Config for naming record files
# ex.add_config({'run_id': 'train_w_default_config',
#                'add_date': True,
#                'cross_val_split': None})

@ex.config
def cfg(data_splits):
    print('config sacred function')

@ex.automain
def main(_config, _run):

    sacred.commands.print_config(_run)
    make_deterministic(12345)
    hparams_dict = dict(_config)
    
    run_str = get_run_str(_config['run_id'], _config['cross_val_split'], _config['add_date'])
    output_path = osp.join( _config['output_path'], "dataset", run_str )
    save_sacred_config(output_path,_config)
    #########################################
    # Load Data 
   
    train_dataset = NuscenesMOTGraphDataset(_config['dataset_params'],
                                            mode =_config["train_dataset_mode"], 
                                            device=_config['gpu_settings']['torch_device'])

    eval_dataset = NuscenesMOTGraphDataset(_config['dataset_params'],
                                    mode =_config["eval_dataset_mode"],
                                    nuscenes_handle = train_dataset.get_nuscenes_handle(),
                                    device =_config['gpu_settings']['torch_device'])

    filename_train = "sequence_sample_list_" + _config["train_dataset_mode"] + ".pkl"
    filename_val = "sequence_sample_list_" + _config["eval_dataset_mode"] + ".pkl"

    filepath_train = osp.join( output_path,filename_train)
    filepath_val = osp.join( output_path,filename_val)
    
    os.makedirs(os.path.dirname(filepath_train), exist_ok=True)
    os.makedirs(os.path.dirname(filepath_val), exist_ok=True)

    # Dump into file (save)
    with open(filepath_train, 'wb') as f:
        sequence_sample_list = train_dataset.seq_frame_ixs
        pickle.dump(sequence_sample_list, f,protocol=4)
    
    with open(filepath_val, 'wb') as f:
        sequence_sample_list = eval_dataset.seq_frame_ixs
        pickle.dump(sequence_sample_list, f,protocol=4)
    # Possible to load with the following code 
    # Uncomment only for demonstration purposes
     
    sequence_sample_list = []
    with open(filepath_train, 'rb') as f:
        sequence_sample_list = pickle.load(f)
        print (sequence_sample_list)
    ###################################