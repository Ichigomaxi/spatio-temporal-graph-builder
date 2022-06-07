import os.path as osp
from typing import List

import torch
# import pickle5 as pickle
import pickle
# from datasets.NuscenesDataset import NuscenesDataset
from datasets.mot_graph import Graph

from visualization.visualize_graph import visualize_geometry_list, visualize_graph_obj_without_GT_selected_edges,visualize_eval_graph, visualize_input_graph,visualize_output_graph,visualize_graph_obj_without_GT
from utils.misc import load_pickle
from datasets.nuscenes_mot_graph import NuscenesMotGraph
if __name__ == "__main__":
    
    # load mot graphs from pickle file
    dirpath = r"C:\Users\maxil\Documents\projects\master_thesis\nuscenes_tracking_results\tracking_error_cases\val_2"
    file_name_graph  = r"error_graph.pkl"
    file_name_tuple_incoming_edges = r"error_incoming_edges_tuple.pkl"

    file_path_graph = osp.join(dirpath,file_name_graph)
    file_path_tuple = osp.join(dirpath,file_name_tuple_incoming_edges)

    graph_obj = None
    with open(file_path_graph, "rb") as fh:
        # data = pickle.load(fh)
        graph_obj = torch.load(f=fh,map_location=torch.device('cpu'))

    graph_obj : Graph = graph_obj

    with open(file_path_tuple, "rb") as fh:
        # data = pickle.load(fh)
        tuple_incoming_edges = torch.load(f=fh,map_location=torch.device('cpu'))

    tuple_incoming_edges: tuple = tuple_incoming_edges

    incoming_edges = tuple_incoming_edges[0]
    incoming_edges_without_spatial_connections = tuple_incoming_edges[1]
    print(graph_obj.num_features)
    print(graph_obj.num_nodes)
    # incoming_edges_index_without_spatial_connections = graph_obj.edge_index[:,incoming_edges_without_spatial_connections]
    geometry_list = visualize_graph_obj_without_GT_selected_edges(graph_obj, incoming_edges_without_spatial_connections)
    visualize_geometry_list(geometry_list)
    # incoming_edges_index = graph_obj.edge_index[incoming_edges]
    geometry_list = visualize_graph_obj_without_GT(graph_obj)
    visualize_geometry_list(geometry_list)

   



