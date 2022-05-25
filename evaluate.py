'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import json
import os
import os.path as osp

import sacred
# for NuScenes eval
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval
from sacred import SETTINGS, Experiment

from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from pl_module.pl_module import MOTNeuralSolver
from utils.evaluation import compute_nuscenes_3D_mot_metrics
from utils.misc import get_run_str_and_save_dir, make_deterministic, save_model_hparams_and_sacred_config

SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config('configs/evaluate_mini_tracking_cfg.yaml')
# ex.add_config({'run_id': 'evaluation',
#                'add_date': True})

@ex.automain
def main(_config, _run):

    ###############
    #sacred.commands.print_config(_run) # No need to print config, as it's overwritten by the one from the ckpt.
    make_deterministic(12345)

    run_str, save_dir = get_run_str_and_save_dir(_config['run_id'], None, _config['add_date'])
    out_files_dir = osp.join(save_dir, 'mot_metrics')
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
    ################
    # Save hparams and config.yaml
    save_model_hparams_and_sacred_config(out_files_dir, model.hparams, _config)
    
    ###############
    # Get output MOT results files 

    # Load test or validation dataset
    # test_dataset = model.test_dataset()

    test_dataset = NuscenesMOTGraphDataset(_config['dataset_params'],
                                            mode = _config['test_dataset_mode'], 
                                            device=_config['gpu_settings']['torch_device'])

    submission_summary, result_path_ = model.track_all_seqs(dataset = test_dataset,
                                              output_files_dir = out_files_dir,
                                              use_gt = _config['eval_params']['use_gt'],
                                               verbose=True)

    if _config['test_dataset_mode'] != "test":

        config_path = 'configs/nuscenes_eval/tracking_nips_2019.json'
        eval_set = _config['test_dataset_mode']

        os.makedirs(out_files_dir, exist_ok=True) # Make sure dir exists
        output_dir_ =  osp.join(out_files_dir, 'eval_results')
        version_:str = test_dataset.nuscenes_dataset.version
        dataroot_:str = test_dataset.nuscenes_dataset.dataroot

        compute_nuscenes_3D_mot_metrics(
                                config_path= config_path,
                                eval_set=eval_set,
                                result_path= result_path_,
                                out_mot_files_path= output_dir_,
                                nuscenes_version=version_, 
                                nuscenes_dataroot=dataroot_, 
                                verbose = True, 
                                render_classes=None, 
                                render_curves=True)

        # cfg_ =None
        # with open(config_path, 'r') as _f:
        #     cfg_ = TrackingConfig.deserialize(json.load(_f))
        # eval_set_ = _config['test_dataset_mode']

        # os.makedirs(out_files_dir, exist_ok=True) # Make sure dir exists
        # output_dir_ =  osp.join(out_files_dir, 'eval_results')

        # version_:str = test_dataset.nuscenes_dataset.version
        # dataroot_:str = test_dataset.nuscenes_dataset.dataroot

        # verbose_ = True

        # render_classes_ = None

        # nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
        #                      nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
        #                      render_classes=render_classes_)

        # render_curves_:bool = True #computes  PR and TP

        # metrics_summary = nusc_eval.main(render_curves=render_curves_)

        # print(metrics_summary)

