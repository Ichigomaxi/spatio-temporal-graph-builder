import argparse
import os.path as osp
from typing import List

import torch
# import pickle5 as pickle
import pickle
# from datasets.NuscenesDataset import NuscenesDataset

from visualization.visualize_graph import visualize_geometry_list, visualize_eval_graph
from utils.misc import load_pickle
from datasets.nuscenes_mot_graph import NuscenesMotGraph
if __name__ == "__main__":
    
    # load mot graphs from pickle file
    dirpath = r"C:\Users\maxil\Documents\projects\master_thesis\nuscenes_tracking_results"
    file_name = r"05-23_18_06_evaluation_single_graphs\inferred_mot_graphs.pkl"

    file_path = osp.join(dirpath,file_name)

    data = None
    with open(file_path, "rb") as fh:
        # data = pickle.load(fh)
        data = torch.load(f=fh,map_location=torch.device('cpu'))
    
    inferred_mot_graphs: List[NuscenesMotGraph] = data

    for mot_graph in inferred_mot_graphs:
        geometry_list = visualize_eval_graph(mot_graph)
        visualize_geometry_list(geometry_list)

