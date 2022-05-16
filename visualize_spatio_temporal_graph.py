import argparse
from random import sample
from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from datasets.NuscenesDataset import NuscenesDataset
# from datasets.NuscenesDataset import NuscenesDataset
from visualization.visualize_graph import main, visualize_input_graph, visualize_output_graph, visualize_geometry_list
import sacred
from sacred import Experiment

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config(r'configs\visualize_mini_val.yaml')
# ex.add_config({'run_id': 'evaluation',
#                'add_date': True})

@ex.automain
def main(_config, _run):
    dataset = NuscenesDataset('v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes")
    dataset_params = _config['dataset_params']

    mot_graph_dataset = NuscenesMOTGraphDataset(dataset_params, 
                                mode= _config['dataset_mode'],
                                splits=_config['data_splits'],
                                nuscenes_handle= dataset.nuscenes_handle)

    for i in range(len(mot_graph_dataset)):
        scene_token, sample_token = mot_graph_dataset.seq_frame_ixs[i]
        mot_graph = mot_graph_dataset.get_from_frame_and_seq(scene_token, sample_token,True,False)
        geometry_list = visualize_input_graph(mot_graph)
        visualize_geometry_list(geometry_list)
        geometry_list = visualize_output_graph(mot_graph)
        visualize_geometry_list(geometry_list)