
import sacred
from sacred import SETTINGS, Experiment

from datasets.nuscenes_mot_graph_dataset import NuscenesMOTGraphDataset
from datasets.NuscenesDataset import NuscenesDataset
# from datasets.NuscenesDataset import NuscenesDataset
from visualization.visualize_graph import (main, visualize_basic_graph,
                                           visualize_geometry_list,
                                           visualize_input_graph,
                                           visualize_output_graph, build_geometries_tracking_boxes, RED, BLUE,GREEN,GREY,BLACK,YELLOW, build_geometries_input_graph_w_pointcloud_w_Bboxes)
from utils.inputs import load_external_tracklets
from utils.nuscenes_helper_functions import find_correspondences_between_tracking_boxes, select_all_boxes_from_k_sample,find_correspondences_between_GT_annotation_boxes
import torch
import numpy as np

SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()
ex.add_config(r'configs\\visualize_mini_tracking_submissions_cfg_laptop.yaml')
# ex.add_config({'run_id': 'evaluation',
#                'add_date': True})

@ex.automain
def main(_config, _run):
    dataset = NuscenesDataset('v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes")
    dataset_params = _config['dataset_params']

    mot_graph_dataset = NuscenesMOTGraphDataset(dataset_params, 
                                mode= _config['test_dataset_mode'],
                                splits=_config['data_splits'],
                                nuscenes_handle= dataset.nuscenes_handle)
    seqs_to_retrieve = mot_graph_dataset.seqs_to_retrieve
    # Load our submission
    path_to_detection_file_ours = r"C:\Users\maxil\Documents\projects\master_thesis\nuscenes_tracking_results\06-08__09-23_evaluation\val_tracking.json"
    ours_tracking_dict = load_external_tracklets(path_to_detection_file_ours,seqs_to_retrieve ,dataset.nuscenes_handle)

    # # Load their submission
    path_to_detection_file = r"C:\Users\maxil\Documents\projects\master_thesis\nuscenes_tracking_results\cbmot_work_dir_4\tracking_result.json"
    theirs_tracking_dict = load_external_tracklets(path_to_detection_file,seqs_to_retrieve ,dataset.nuscenes_handle)

    
    for i in range(len(mot_graph_dataset)):
        scene_token, sample_token = mot_graph_dataset.seq_frame_ixs[i]
        mot_graph = mot_graph_dataset.get_from_frame_and_seq(scene_token, sample_token,True,False)
        frames_per_graph :int = mot_graph.max_frame_dist
        # # OUR TRACKS ---------------------------------------------------------------------------------------------------
        # # Get boxes from samples i and j
        sample_token_boxes_list_ours  = select_all_boxes_from_k_sample(sample_token,2,dataset.nuscenes_handle,ours_tracking_dict )
        
        # # THEIRS TRACKS ---------------------------------------------------------------------------------------------------
        # Get boxes from samples i and j
        sample_token_boxes_list_theirs  = select_all_boxes_from_k_sample(sample_token,frames_per_graph,dataset.nuscenes_handle,theirs_tracking_dict )

        # GT ANNOTATION TRACKS ---------------------------------------------------------------------------------------------------



        #  Check if list is empty, if empty do nothing
        # if sample_token_boxes_list_ours and sample_token_boxes_list_theirs:
        geometry_list = []
        offset = mot_graph.SPATIAL_SHIFT_TIMEFRAMES
        # OUR TRACKS ---------------------------------------------------------------------------------------------------
        # Find corresponding edges betweem sample i and j
        _ , boxes_i_ours = sample_token_boxes_list_ours[0]
        _ , boxes_j_ours = sample_token_boxes_list_ours[1]
        all_boxes_ours, edge_indices_ours = find_correspondences_between_tracking_boxes(boxes_i_ours,boxes_j_ours)
        geometry_list.extend( build_geometries_tracking_boxes(sample_token_boxes_list_ours, edge_indices_ours, offset=offset, edge_color=BLUE ,tracking_box_color=BLACK))
        # # THEIRS TRACKS ---------------------------------------------------------------------------------------------------
        _ , boxes_i_theirs = sample_token_boxes_list_theirs[0]
        _ , boxes_j_theirs = sample_token_boxes_list_theirs[1]
        all_boxes_theirs, edge_indices_theirs = find_correspondences_between_tracking_boxes(boxes_i_theirs,boxes_j_theirs)
        geometry_list.extend( build_geometries_tracking_boxes(sample_token_boxes_list_theirs, edge_indices_theirs, offset=offset, edge_color=RED ,tracking_box_color=RED))
        # GT ANNOTATION TRACKS ---------------------------------------------------------------------------------------------------
        edge_labels:torch.Tensor = mot_graph.graph_obj.edge_labels
        edge_indices:torch.Tensor = mot_graph.graph_obj.edge_index
        true_positive_edge_indices:torch.Tensor = edge_indices[:,edge_labels>0]
        true_positive_edge_indices:np.ndarray = true_positive_edge_indices.cpu().numpy()
        sample_token_boxes_list_gt = []

        
        
        for i in range(frames_per_graph):
            sample_token, _ = sample_token_boxes_list_theirs[i]
            sample_token_boxes_list_gt.append( (sample_token, mot_graph.graph_dataframe["boxes_dict"][i]) )

        _ , boxes_i_gt = sample_token_boxes_list_gt[0]
        _ , boxes_j_gt = sample_token_boxes_list_gt[1]

        all_boxes_gt, edge_indices_gt = find_correspondences_between_GT_annotation_boxes(boxes_i_gt, boxes_j_gt,dataset.nuscenes_handle)

        
        geometry_list.extend( build_geometries_tracking_boxes(sample_token_boxes_list_gt, edge_indices_gt, offset=offset, edge_color=GREEN ,tracking_box_color=GREEN))
        
        
        # geometry_list.extend( build_geometries_input_graph_w_pointcloud_w_Bboxes)
        visualize_geometry_list(geometry_list)

        # geometry_list = visualize_output_graph(mot_graph)
        # visualize_geometry_list(geometry_list)
