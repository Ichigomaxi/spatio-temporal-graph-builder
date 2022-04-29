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
from torch_geometric.loader import DataLoader
##################

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()

ex.add_config('configs/tracking_cfg.yaml')
# ex.add_config('configs/debug_config_file.yaml')

# Config for naming record files
ex.add_config({'run_id': 'train_w_default_config',
               'add_date': True,
               'cross_val_split': None})

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
    nusc = NuScenes(version='v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes", verbose=True)
    # nusc = NuScenes(version='v1.0-trainval', dataroot='/media/HDD2/Datasets/mini_nusc', verbose=True)
    split = create_splits_scenes()
    print(split.keys())
    split_scene_list = []
    for scene_name in split['mini_train']:
        for scene in nusc.scene:
            if scene['name']==scene_name:
                split_scene_list.append(scene)

    sample_dict = {}
    i = 0 
    for scene in split_scene_list:
        last_sample_token =""
        sample_token = scene['first_sample_token']
        while(last_sample_token == ""):
            
            sample = nusc.get('sample', sample_token)
            sample_dict[i] = (scene['token'],sample["token"])
            i += 1
            sample_token = sample["next"]
            if(sample["token"]== scene['last_sample_token']):
                last_sample_token = scene['last_sample_token']

    #Create List of Graph objects
    #______________________________________________________________#
    # Decide if only first scene should be computed
    only_first_scene = True
    scene_token0, sample_token0= sample_dict[0]
    device = "cuda:1"
    device = model.device
    #_______________________________________________________________#

    MotGraphList= []
    for sample_key in sample_dict:
        scene_token_current, sample_token_current= sample_dict[sample_key]
        if(only_first_scene):
            if(scene_token0 == scene_token_current):
                object = NuscenesMotGraph(nuscenes_handle = nusc,
                            start_frame=sample_token_current,
                            max_frame_dist = 3, 
                            filterBoxes_categoryQuery='vehicle.car',
                            device= device)
                is_possible2construct = object.is_possible2construct
                if is_possible2construct:
                    object.construct_graph_object()
                    # object.assign_edge_labels_one_hot()
                    # object.assign_edge_labels(label_type='multiclass')
                    object.assign_edge_labels(label_type='binary')
                    MotGraphList.append(object)
        else:
            object = NuscenesMotGraph(nuscenes_handle = nusc,
                            start_frame=sample_token_current,
                            max_frame_dist = 3,  
                            filterBoxes_categoryQuery='vehicle.car',device= device)
            is_possible2construct = object.is_possible2construct
            if is_possible2construct:
                object.construct_graph_object()
                object.assign_edge_labels(label_type='binary')
                # object.assign_edge_labels(label_type='multiclass')
                MotGraphList.append(object)


    graph_list = []
    for graph in MotGraphList:
        graph_list.append(graph.graph_obj)
    train_loader = DataLoader(graph_list,
                            batch_size=_config['train_params']['batch_size'],
                            num_workers=_config['train_params']['num_workers'],
                            shuffle =False)
    ###################################

    # Validation percentage check is deprecated in favour of limit_val_batches 
    # val_percent_check = _config['eval_params']['val_percent_check']
    # limit_val_batches = _config['eval_params']['val_percent_check']

    #nb_sanity_val_steps is deprecated

    # check_val_every_n_epoch=_config['eval_params']['check_val_every_n_epoch'] is deprecated
    # default_save_path=osp.join(OUTPUT_PATH, 'experiments', run_str) is deprecated
    accelerator = _config['gpu_settings']['device_type']
    devices = _config['gpu_settings']['device_id']

    trainer = Trainer(gpus=devices,
                    callbacks=[ckpt_callback],
                    weights_summary = None,
                    checkpoint_callback=False,
                    max_epochs=_config['train_params']['num_epochs'],
                    logger =logger,
                    )

    
    trainer.fit(model,train_loader)