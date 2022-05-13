import argparse
import os.path as osp
from typing import List
# from datasets.NuscenesDataset import NuscenesDataset
from visualization.visualize_graph import visualize_output_graph
from utils.misc import load_pickle
from datasets.nuscenes_mot_graph import NuscenesMotGraph
if __name__ == "__main__":
    
    # load mot graphs from pickle file
    dirpath = r"C:\Users\maxil\Documents\projects\master_thesis\nuscenes_tracking_results"
    file_name = r"05-13_18_59_evaluation_single_graphs\inferred_mot_graphs.pkl"

    file_path = osp.join(dirpath,file_name)
    inferred_mot_graphs: List[NuscenesMotGraph] = load_pickle(file_path)

