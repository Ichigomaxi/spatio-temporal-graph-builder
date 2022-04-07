from collections import defaultdict
from enum import Enum
from turtle import distance
from typing import Dict

from sklearn.neighbors import NearestNeighbors

from matplotlib.pyplot import axis
from utility import get_box_centers

import numpy as np
import torch

class Timeframe(Enum):
    t0 = 0
    t1 = 1
    t2 = 2

class Graph(object):
    """ Graph data structure, undirected by default.
    Taken from https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
    After consideration of different Data Structures: https://en.wikipedia.org/wiki/Graph_(abstract_data_type)#Representations
    """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

class SpatioTemporalGraph(Graph):
    """ Special Graph representation for spatio-temporal graphs
    Graph data structure, undirected by default.
    Taken from https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
    After consideration of different Data Structures: https://en.wikipedia.org/wiki/Graph_(abstract_data_type)#Representations
    """
    def __init__(self,box_center_list , connections, directed=False, ):
        super().__init__(connections, directed)
        
        self._center_points = []
        self._center_points_stacked = np.empty((0,3))

        for boxes in box_center_list:
            box_center_points = get_box_centers(boxes)
            self._center_points.append([box_center_points])
            np.append(self._center_points_stacked,box_center_points, axis = 0)

    def __init__(self, connections, directed=False, ):
        super().__init__(connections, directed)
        
        self._center_points = []
        self._center_points_stacked = np.empty((0,3))

    def get_spatial_pointpairs(self, timeframe: Timeframe):
        spatial_pointpairs = []
        for reference_node in self._graph:
            if(reference_node[0]== timeframe):
                for neighbor_node in self._graph[reference_node]:
                    # print(neighbor_index[0])
                    timestep, idx = neighbor_node[0],neighbor_node[1]
                    if timestep == timeframe:
                        spatial_pointpairs.append([reference_node[1],idx])
        return spatial_pointpairs
        
    def get_temporal_pointpairs(self):
        temporal_pairs_indices = []
        for reference_node in self._graph:
            reference_timeframe = reference_node[0]
            # Find corresponding indices in global centers list
            point_a = self.get_points(reference_node)
            reference_idx_global = np.argwhere(self._center_points_stacked == point_a)[0,0]

            for neighbor_node in self._graph[reference_node]:
                # print(neighbor_index[0])
                neighbor_timeframe, neighbor_idx = neighbor_node[0],neighbor_node[1]
                if neighbor_timeframe != reference_timeframe:
                    # Find corresponding indices in global centers list
                    point_b = self.get_points(neighbor_node)
                    neighbor_idx_global = np.argwhere(self._center_points_stacked == point_b)[0,0]
                    #Append global indices into list
                    temporal_pairs_indices.append([reference_idx_global,neighbor_idx_global])
        return temporal_pairs_indices
        
    def get_points(self,reference_node):
        if(reference_node[0]== Timeframe.t0):
            return self._center_points[0][reference_node[1]]
        elif (reference_node[0]== Timeframe.t1):
            return self._center_points[1][reference_node[1]]
        elif (reference_node[0]== Timeframe.t2):
            return self._center_points[2][reference_node[1]]
        else:
            return AttributeError

def add_general_centers(centers_dict,spatial_shift_timeframes):

    if len(centers_dict) == 3:
        _ ,centers0 =  centers_dict[0]
        _ ,centers1 =  centers_dict[1]
        _ ,centers2 =  centers_dict[2]

        # Boxes 1 must be translated up by l meters
        centers1 += np.array([0,0,spatial_shift_timeframes])

        # Boxes 2 must be translated up by 2*l meters
        centers2 += np.array([0,0,2*spatial_shift_timeframes])

        # Add all centroids into one array
        centers = np.empty((0,3))
        centers = np.append(centers, centers0, axis=0)
        centers = np.append(centers, centers1, axis=0)
        centers = np.append(centers, centers2, axis=0)

        centers_dict["all"] = centers
        

def get_and_compute_temporal_edge_indices( centers_dict:Dict,\
    knn_param:int, use_cuda:bool=True)-> torch.Tensor:
    '''
    Returns indices of edges in reference to the stacked center indices.
    Returns:
    t_temporal_pointpairs:torch interger tensor (num_edges,2)
    '''

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    temporal_pointpairs = []

    # print(centers_dict)

    _,centers0 =  centers_dict[0]
    _ ,centers1 =  centers_dict[1]
    _ ,centers2 =  centers_dict[2]

    # # Boxes 1 must be translated up by l meters
    # centers1 += np.array([0,0,spatial_shift_timeframes])

    # # Boxes 2 must be translated up by 2*l meters
    # centers2 += np.array([0,0,2*spatial_shift_timeframes])

    # # Add all centroids into one array
    # centers = np.empty((0,3))
    # centers = np.append(centers, centers0, axis=0)
    # centers = np.append(centers, centers1, axis=0)
    # centers = np.append(centers, centers2, axis=0)
    centers = centers_dict["all"]

    for i in range(len(centers0)):
        center = centers0[i]
        center = np.expand_dims(center,axis=0)
        temp = np.append(centers1,center,axis=0)
        #Find nearest_neigbor
        nearest_neigbor = NearestNeighbors(n_neighbors=knn_param, algorithm='ball_tree').fit(temp)
        temporal_distances, temporal_indices = nearest_neigbor.kneighbors(temp)
        #Add indices into a list
        for index in temporal_indices[-1]:
            #adapt the index to the global indexing
            # temporal_pointpairs.append([i, index + len(centers0)])

            # find global indices and append them
            reference_node_global_index = np.argwhere(centers == center)[0,0]
            neighbor_node_global_index = np.argwhere(centers == temp[index])[0,0] 
            temporal_pointpairs.append([reference_node_global_index ,\
                neighbor_node_global_index ])

    # connect frame-0-nodes with frame-2-nodes
    for i in range(len(centers0)):
        center = centers0[i]
        center = np.expand_dims(center,axis=0)
        temp = np.append(centers2,center,axis=0)
        #Find nearest_neigbor
        nearest_neigbor = NearestNeighbors(n_neighbors=knn_param, algorithm='ball_tree').fit(temp)
        temporal_distances, temporal_indices = nearest_neigbor.kneighbors(temp)
        #Add indices into a list (The last entry belongs to center!)
        for index in temporal_indices[-1]:
            #adapt the index to the global indexing
            # temporal_pointpairs.append([i, index + len(centers0)])

            # find global indices and append them
            reference_node_global_index = np.argwhere(centers == center)[0,0]
            neighbor_node_global_index = np.argwhere(centers == temp[index])[0,0] 
            temporal_pointpairs.append([reference_node_global_index ,\
                neighbor_node_global_index ])

    # connect frame-1-nodes with frame-2-nodes
    for i in range(len(centers1)):
        center = centers1[i]
        center = np.expand_dims(center,axis=0)
        temp = np.append(centers2,center,axis=0)
        nearest_neigbor = NearestNeighbors(n_neighbors=knn_param, algorithm='ball_tree').fit(temp)
        temporal_distances, temporal_indices = nearest_neigbor.kneighbors(temp)

        # Test if the last input is the appended center point
        # assert (temporal_distances[-1] == temporal_distances[np.argwhere(temp == center)[0,0]]).all()

        for index in temporal_indices[-1]:
            #adapt the index to the global indexing
            # temporal_pointpairs.append([i + len(centers0) , index + len(centers0) + len(centers1) ])
            
            # find global indices and append them
            reference_node_global_index = np.argwhere(centers == center)[0,0]
            neighbor_node_global_index = np.argwhere(centers == temp[index])[0,0] 
            temporal_pointpairs.append([reference_node_global_index ,\
                neighbor_node_global_index ])

    np_temporal_pointpairs = np.asarray(temporal_pointpairs)
    t_temporal_pointpairs = torch.from_numpy(np_temporal_pointpairs).to(device)
    
    return t_temporal_pointpairs

def compute_edge_feats_dict(edge_ixs,centers_dict, use_cuda):
    """
    Computes a dictionary of edge features among pairs of detections
    Args:
        edge_ixs: Edges tensor with shape (2, num_edges)
        use_cuda: bool, determines whether operations must be performed in GPU
    Returns:
        Dict where edge key is a string referring to the attr name, and each val is a tensor of shape (num_edges)
        with vals of that attribute for each edge.

    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    relative_vectors = []
    centers = centers_dict["all"]
    for edge in edge_ixs:
        reference_ind = edge[0]
        neighbor_ind = edge[1]
        reference_node = centers[reference_ind]
        neighbor_node = centers[neighbor_ind]
        relative_vector =  reference_node - neighbor_node
        relative_vectors.append(relative_vector) # List of numpy arrays
    np_relative_vectors = np.asarray(relative_vectors)
    t_relative_vectors = torch.from_numpy(np_relative_vectors).to(device)
    
    edge_feats_dict = {'relative_vectors': t_relative_vectors
        }

    return edge_feats_dict