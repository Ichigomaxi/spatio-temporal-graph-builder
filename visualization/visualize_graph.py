import numpy as np
import torch
import sys
# sys.path.append("datasets")
# import datasets.NuscenesDataset
from datasets.NuscenesDataset import NuscenesDataset
from datasets.nuscenes_mot_graph import NuscenesMotGraph

import open3d as o3d
from open3d import geometry
def add_pointcloud(points:torch.Tensor, color:np.ndarray =None):
    """
    Inputs:
    points.shape = (n,3)
    color.shape = (3)
    """
    line_set_sequences = [] 
    if color is None:
        color = [[0.5, 0.5, 0.5] for i in range(points.shape[0])]
    else: 
        color = np.tile(color, (points.shape[0], 1))

    pcd = o3d.geometry.PointCloud()
    np_points = points.cpu().numpy().reshape(-1,3)
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(color)

    line_set_sequences.append(pcd)
    return line_set_sequences

def add_line_set(nodes:torch.Tensor, edge_indices: torch.Tensor,edge_labels: torch.Tensor = None):
    '''
    nodes.shape = (N, 3) only xyz values 
    edge_indices.shape = (2, E)
    edge_labels.shape = (E, 1)
    '''
    line_set_sequences = []
    mask_edge_label = edge_labels == 1
    
    colors = []
    if edge_labels is None:
        colors= [[1, 0, 0] for i in range(edge_indices.shape[1])]
    else:
        for edge_label in edge_labels:
            if edge_label == 0:
                colors.append([1, 0, 0])
            else:
                colors.append([0, 1, 0])

    # make sure that edge_labels has same length as edge indices
    assert len(colors) == edge_indices.shape[1]

    # #Transport onto CPU and Transform into numpy array
    # Vector3dVector only takes in up to 3 dimensions. Get only xyz from node_features
    np_nodes = nodes.cpu().numpy().reshape(-1, 3)
    np_edge_indices = edge_indices.cpu().numpy().reshape(-1, 2)
    np_edge_indices.astype(np.int32) #Vector2iVector takes in int32 type
    # Transpose if Graph connectivity in COO format with shape :obj:`[2, num_edges]
    if np_edge_indices.shape[0] == 2:
        np_edge_indices = np_edge_indices.T

    line_set = geometry.LineSet(points=o3d.utility.Vector3dVector(np_nodes),
        lines=o3d.utility.Vector2iVector(np_edge_indices))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    line_set_sequences.append(line_set) 

    # colors = [[1, 0, 0] for i in range(len(spatial_pointpairs1))]
    # line_set1 = geometry.LineSet(points=o3d.utility.Vector3dVector(centers1),
    #     lines=o3d.utility.Vector2iVector(spatial_pointpairs1))
    # line_set1.colors = o3d.utility.Vector3dVector(colors)

    # colors = [[1, 0, 0] for i in range(len(spatial_pointpairs2))]
    # line_set2 = geometry.LineSet(points=o3d.utility.Vector3dVector(centers2),
    #     lines=o3d.utility.Vector2iVector(spatial_pointpairs2))
    # line_set2.colors = o3d.utility.Vector3dVector(colors)

    return line_set_sequences

def visualize_input_graph(mot_graph:NuscenesMotGraph):
    geometry_list = []
    # mot_graph.edge_attr

    return geometry_list

def visualize_output_graph(mot_graph:NuscenesMotGraph):

    geometry_list = []

    #----------------------------------------
    # Include reference frame
    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
                size=5, origin=[0, 0, 0])  # create coordinate frame
    geometry_list += [mesh_frame]

    #----------------------------------------
    # Temporal Edges 
    temporal_edges_mask = mot_graph.graph_obj.temporal_edges_mask
    temporal_edges = mot_graph.graph_obj.edge_index.T[temporal_edges_mask].reshape(-1,2)
    
    line_set_sequence_temporal_connections = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
                                    edge_indices= temporal_edges,
                                    edge_labels = None)

    geometry_list += line_set_sequence_temporal_connections
    
    #----------------------------------------
    # Spatial Edges
    spatial_edges_mask = ~ mot_graph.graph_obj.temporal_edges_mask
    temporal_edges = mot_graph.graph_obj.edge_index.T[spatial_edges_mask].reshape(-1,2)
    line_set_sequence_spatial_connections = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
                                    edge_indices= mot_graph.graph_obj.edge_index,
                                    edge_labels = None)
    geometry_list += line_set_sequence_spatial_connections

    #----------------------------------------
    # Color Points/Nodes
    point_sequence = add_pointcloud(mot_graph.graph_obj.x[:,:3],
                                    color= None)
    geometry_list += point_sequence
    #----------------------------------------
    # Draw Graph/Edges with Lineset
    # Spatial Edges Red Edges

    # line_set_sequence = add_line_set(nodes= mot_graph.graph_obj.x[:,:3],
    #                                 edge_indices= mot_graph.graph_obj.edge_index,
    #                                 edge_labels = None)
    # geometry_list += line_set_sequence
    
    #----------------------------------------

    return geometry_list


def main():
    dataset = NuscenesDataset('v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes")
    MotGraphList = dataset.get_nuscenes_mot_graph_list_debugging(True,
                            specific_device=None,
                            label_type="binary",
                            filterBoxes_categoryQuery="vehicle.car")

    for i, mot_graph in enumerate(MotGraphList):
        if i == 0:
            geometry_list_input = visualize_input_graph(mot_graph)
            geometry_list_output = visualize_output_graph(mot_graph)
            if len(geometry_list_input) != 0:
                o3d.visualization.draw_geometries(geometry_list_input)
            if len(geometry_list_output) != 0:   
                print("starting visualization!")
                o3d.visualization.draw_geometries(geometry_list_output)
                print("stoped visualization!")

if __name__ == "__main__":
    main()