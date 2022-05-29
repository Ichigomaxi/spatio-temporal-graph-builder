from collections import defaultdict

from enum import Enum
from typing import Dict,List

from sklearn.neighbors import NearestNeighbors

from utility import get_box_centers

from enum import IntEnum
import numpy as np
import torch

class edge_types(IntEnum):
    spatial_edges = 0
    temporal_edges = 1

EDGE_FEATURE_COMPUTATION_MODE = {"relative_position", "edge_type"}


class Timeframe(Enum):
    """
    deprecated
    """
    t0 = 0
    t1 = 1
    t2 = 2


class Graph(object):
    """ 
    deprecated
    Graph data structure, undirected by default.
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
    """
    deprecated
    Special Graph representation for spatio-temporal graphs
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
    """
    deprecated
    """
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

def transform_knn_matrix_2_neighborhood_list_new(t_knn_matrix:torch.Tensor) -> torch.Tensor:
    """
    Returns indices of edges in reference to the stacked center indices.
    No selfreferencing!
    t_knn_matrix: tensor. first column contains current node_idx, all following k coloumns contain the knn-neighbors-indices
    
    """
    num_nodes = t_knn_matrix.shape[0]
    k = t_knn_matrix.shape[1] - 1 
    edge_indices = torch.empty(num_nodes * k, 2)

    edge_indices_list = []
    current_node_idx:torch.Tensor = t_knn_matrix[:,0]
    for i in range(k):
        l_th_neighbor = i+1
        edge_indices_l = torch.stack([current_node_idx, t_knn_matrix[:,l_th_neighbor]], dim=1)
        edge_indices_list.append(edge_indices_l)

    edge_indices = torch.cat(edge_indices_list,dim = 0)
    return edge_indices

def transform_knn_matrix_2_neighborhood_list(t_knn_matrix:torch.Tensor,
            node_list_length:int) -> torch.Tensor:
    '''
    Returns indices of edges in reference to the stacked center indices.
    Arg: 
    t_knn_matrix: torch.Tensor(num_edges_i, knn_parameter== num_neighbors)
    node_list: torch.Tensor(num_edges_i, num_features)
    Returns:
    t_spatial_pointpairs:torch interger tensor (num_spatial_edges,2)
    '''
    neighborhood_list = []
    for reference_edge_index in range(node_list_length):
        # tensors of shape (1 x knn)
        t_current_neighbor_indices = t_knn_matrix[reference_edge_index]
        t_reference_edge_index = torch.ones_like(t_current_neighbor_indices) * reference_edge_index

        t_edge_indices_pairs = torch.stack([t_reference_edge_index,t_current_neighbor_indices]).T

        neighborhood_list.append(t_edge_indices_pairs)

    t_neighborhood_list = torch.cat(neighborhood_list, dim = 0)
    return t_neighborhood_list

def is_invalid_frame(graph_dataframe:Dict,knn_param:int):
    """
    Returns Dict of Bools, which determine if number of 
    object centers is lower than knn-parameter
    Adopts keys from graph_dataframe["centers_dict"] or graph_dataframe["boxes_dict"]
    """
    invalid_frames= {}

    for key in graph_dataframe["boxes_dict"]:
        boxes_i = graph_dataframe["boxes_dict"][key]
        num_object_in_frame_i = len(boxes_i)
        if num_object_in_frame_i < knn_param:
            invalid_frames[key] = True
        else:
            invalid_frames[key] = False

    return invalid_frames

def compare_two_edge_indices_matrices(edge_indices_a:torch.Tensor,edge_indices_b:torch.Tensor):
    list_a = edge_indices_a.tolist()
    list_b = edge_indices_b.tolist()

    set_a = set([tuple(indices_pair) for indices_pair in list_a])
    set_b = set([tuple(indices_pair) for indices_pair in list_b])

    set_difference = set_a - set_b

    if set_difference:
        return False
    else:
        return True

def get_and_compute_spatial_edge_indices_new( 
        frames_per_graph:int , 
        graph_dataframe:Dict,\
        knn_param:int, 
        self_referencing_edges:bool = False,
        adapt_knn_param = False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )-> torch.Tensor:
    '''
    Returns indices of edges in reference to the stacked center indices.
    If all frames have only one object then an empty torch tensor should be returned
    Returns:
    t_spatial_pointpairs:torch interger tensor (num_spatial_edges,2)
    '''
    # increase knn param because one column will be removed since it 
    # is the self referencing edge connection
    knn_param_temp = None
    if(self_referencing_edges==False):
        knn_param_temp = knn_param + 1
    else:
        knn_param_temp = knn_param

    # check if number of object centers is less than the number of neighbors k needed for  KNN
    invalid_frames = None
    if adapt_knn_param:
        invalid_frames = is_invalid_frame(graph_dataframe, knn_param)
    #Init indices list
    spatial_indices = []

    num_spatial_nodes_considered = 0 
    for i in range(frames_per_graph):
        ############################################
        # Build spatial graph for timeframe i
        
        # Get individual nodes for timeframe i
        _, centers = graph_dataframe["centers_dict"][i]

        # Adapt K-NN parameter
        knn_param_temp_i = knn_param_temp
        if invalid_frames is not None:
            # Check if frame i has less than k objects
            if (invalid_frames[i]):
                # if num_object < k 
                # then k = num_object
                # Therefore, KNN-algorithm will look for the (k-1) neighbors close to the k objects
                # Because the k also includes a selfreference
                knn_param_temp_i = len(centers)
        
        #Compute K nearest neighbors
        t_edge_indices_i = None
        num_spatial_nodes_i = centers.shape[0]

        if len(centers) == 1:
            t_edge_indices_i = torch.tensor([],dtype=torch.long).to(device) # There are no spatial connections if there is only one object
        else:
            # Compute KNN-Estimator
            nbrs_i = NearestNeighbors(n_neighbors=knn_param_temp_i, algorithm='ball_tree').fit(centers)

            # Given the same data to estimate the K-Neighbors, 
            # the first and closest neighbors are always the nodes itself.
            # Interpretation:
            # Array contains the nodes id in the first column and 
            # all knn-neighbor-indices in the following columns 
            spatial_indices_i:np.ndarray = nbrs_i.kneighbors(centers, return_distance=False)
            
            # New way to compute the indices
            spatial_indices_i_old = spatial_indices_i
            t_spatial_indices_i_old:torch.Tensor = torch.from_numpy(spatial_indices_i_old).to(device)
            t_edge_indices_i_new = transform_knn_matrix_2_neighborhood_list_new(t_spatial_indices_i_old)

            # Old way to compute the indices
            #Remove the self referencing edge connection
            if (self_referencing_edges==False):
                spatial_indices_i = spatial_indices_i[:, 1:]
            t_spatial_indices_i:torch.Tensor = torch.from_numpy(spatial_indices_i).to(device)
            t_edge_indices_i = transform_knn_matrix_2_neighborhood_list(t_spatial_indices_i, num_spatial_nodes_i).to(device)
            t_edge_indices_i = t_edge_indices_i.long()
            assert compare_two_edge_indices_matrices(t_edge_indices_i,t_edge_indices_i_new)

        # Increase Edge_indices by the number of past nodes that already have been considered
        t_edge_indices_i += num_spatial_nodes_considered
        spatial_indices.append(t_edge_indices_i)
        num_spatial_nodes_considered += num_spatial_nodes_i
    
    t_spatial_indices = torch.cat(spatial_indices, dim=0).to(device)
    # If all frames have only one object then an empty torch tensor should be returned
    return t_spatial_indices.to(torch.long)
    

def get_and_compute_spatial_edge_indices( 
        frames_per_graph:int , 
        graph_dataframe:Dict,
        knn_param:int, 
        self_referencing_edges:bool = False,
        adapt_knn_param = False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )-> torch.Tensor:
    '''
    Returns indices of edges in reference to the stacked center indices.
    Returns:
    t_spatial_pointpairs:torch interger tensor (num_spatial_edges,2)
    '''
    # increase knn param because one column will be removed since it 
    # is the self referencing edge connection
    knn_param_temp = None
    if(self_referencing_edges==False):
        knn_param_temp = knn_param + 1
    else:
        knn_param_temp = knn_param

    # check if number of object centers is less than the number of neighbors k needed for  KNN
    invalid_frames = None
    if adapt_knn_param:
        invalid_frames = is_invalid_frame(graph_dataframe, knn_param)

    #Init indices list
    spatial_indices = []

    # Get individual centers
    _, centers0 = graph_dataframe["centers_dict"][0]
    _, centers1 = graph_dataframe["centers_dict"][1]
    _, centers2 = graph_dataframe["centers_dict"][2]

    # Frame t0
    # Adapt K-NN parameter
    knn_param_temp_0 = knn_param_temp
    if invalid_frames is not None:
        if (invalid_frames[0]):
            knn_param_temp_0 = len(centers0)
    
    #Compute K nearest neighbors
    # print("knn_param_temp_0: ",knn_param_temp_0)
    # print("length num obj target: ",len(centers0))
    nbrs_0 = NearestNeighbors(n_neighbors=knn_param_temp_0, algorithm='ball_tree').fit(centers0)
    spatial_indices_0 = nbrs_0.kneighbors(centers0, return_distance=False)
    #Remove the self referencing edge connection
    if (self_referencing_edges==False):
        spatial_indices_0 = spatial_indices_0[:, 1:]

    t_spatial_indices_0 = torch.from_numpy(spatial_indices_0).to(device)
    num_spatial_nodes_0 = centers0.shape[0]
    t_edge_indices_0 = transform_knn_matrix_2_neighborhood_list(t_spatial_indices_0, num_spatial_nodes_0).to(device)
    spatial_indices.append(t_edge_indices_0)
    
    #Frame t1
    # Adapt K-NN parameter
    knn_param_temp_1 = knn_param_temp
    if invalid_frames is not None:
        if (invalid_frames[1]) :
            knn_param_temp_1 = len(centers1)
        
    #Compute K nearest neighbors
    # print("knn_param_temp_1: ",knn_param_temp_1)
    # print("length num obj target: ",len(centers1))
    nbrs_1 = NearestNeighbors(n_neighbors=knn_param_temp_1, algorithm='ball_tree').fit(centers1)
    spatial_indices_1 = nbrs_1.kneighbors(centers1, return_distance=False)
    #Remove the self referencing edge connection
    if (self_referencing_edges==False):
        spatial_indices_1 = spatial_indices_1[:, 1:]

    t_spatial_indices_1 = torch.from_numpy(spatial_indices_1).to(device)
    num_spatial_nodes_1 = centers1.shape[0]
    t_edge_indices_1 = transform_knn_matrix_2_neighborhood_list(t_spatial_indices_1, num_spatial_nodes_1).to(device)
    t_edge_indices_1 += (num_spatial_nodes_0)
    spatial_indices.append(t_edge_indices_1)

    #Frame t2
    # Adapt K-NN parameter
    knn_param_temp_2 = knn_param_temp
    if invalid_frames is not None:
        if (invalid_frames[2]):
            knn_param_temp_2 = len(centers2)
        
    
    #Compute K nearest neighbors
    # print("knn_param_temp_2: ",knn_param_temp_2)
    # print("length num obj target: ",len(centers2))

    nbrs_2 = NearestNeighbors(n_neighbors=knn_param_temp_2, algorithm='ball_tree').fit(centers2)
    spatial_indices_2 = nbrs_2.kneighbors(centers2, return_distance=False)
    #Remove the self referencing edge connection
    if (self_referencing_edges==False):
        spatial_indices_2 = spatial_indices_2[:, 1:]
    
    t_spatial_indices_2 = torch.from_numpy(spatial_indices_2).to(device)
    num_spatial_nodes_2 = centers2.shape[0]
    t_edge_indices_2 = transform_knn_matrix_2_neighborhood_list(t_spatial_indices_2, num_spatial_nodes_2).to(device)
    t_edge_indices_2 += (num_spatial_nodes_1) + (num_spatial_nodes_0)
    spatial_indices.append(t_edge_indices_2)

    t_spatial_indices = torch.cat(spatial_indices, dim=0).to(device)

    return t_spatial_indices

def build_temporal_connections(
        temporal_pointpairs:List[np.ndarray], 
        all_nodes:np.ndarray ,
        current_nodes_i:np.ndarray, 
        following_nodes_j :np.ndarray, 
        knn_param:int,
        self_referencing_edges:bool = False):
    
    for i in range(len(current_nodes_i)):
        center = current_nodes_i[i]
        center = np.expand_dims(center,axis=0)
        temp = np.append(following_nodes_j,center,axis=0)
        #Find nearest_neigbor
        nearest_neigbor = NearestNeighbors(n_neighbors=knn_param, algorithm='ball_tree').fit(temp)
        temporal_indices = nearest_neigbor.kneighbors(temp, return_distance=False)
        #Remove the self referencing edge connection
        if (self_referencing_edges==False):
            temporal_indices = temporal_indices[:,1:]
        #Add indices into a list
        for index in temporal_indices[-1]:
            #adapt the index to the global indexing
            # temporal_pointpairs.append([i, index + len(current_nodes_i)])

            # find global indices and append them
            reference_node_global_index:np.int64 = np.argwhere(all_nodes == center)[0,0]
            neighbor_node_global_index:np.int64 = np.argwhere(all_nodes == temp[index])[0,0] 
            temporal_pointpairs.append([reference_node_global_index ,\
                neighbor_node_global_index ])

def get_and_compute_temporal_edge_indices_new(
        frames_per_graph:int , 
        graph_dataframe:Dict,
        knn_param:int, self_referencing_edges:bool = False,
        adapt_knn_param = False,
        max_length_temporal_edges:int = 2,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )-> torch.Tensor:
    '''
    Returns indices of edges in reference to the stacked center indices.
    Returns:
    t_temporal_pointpairs:torch interger tensor (num_edges,2)
    '''
    # increase knn param because one column will be removed since it 
    # is the self referencing edge connection
    knn_param_temp = None
    if(self_referencing_edges==False):
        knn_param_temp = knn_param + 1
    else:
        knn_param_temp = knn_param
    
    # check if number of object centers is less than the number of neighbors k needed for  KNN
    invalid_frames = None
    if adapt_knn_param:
        invalid_frames = is_invalid_frame(graph_dataframe,knn_param)
        print('invalid_frames',invalid_frames)

    temporal_pointpairs = []

    # Connect the first (frames_per_graph-1) frames to their following frames
    maximium_frame_id = (frames_per_graph - 1)
    all_nodes = graph_dataframe["centers_list_all"].cpu().numpy()
    for current_frame_id in range(frames_per_graph-1):
        _, current_nodes_i = graph_dataframe["centers_dict"][current_frame_id]
        # Connect nodes with temporal_edge_length = i 
        for temporal_edge_length in range(1,max_length_temporal_edges+1,1): # 1,2,3 
            next_frame_id = current_frame_id + temporal_edge_length
            # Break this iteration if there arent any future frames left to connect to
            if next_frame_id > maximium_frame_id:
                break
            ## Do building#######################
            # Get individual centers
            _, following_nodes_j =  graph_dataframe["centers_dict"][next_frame_id]

            # Adapt K-NN parameter according to target time-frame 
            target_frame = next_frame_id
            if invalid_frames is not None:
                if (invalid_frames[target_frame]):
                    knn_param_temp = len(following_nodes_j) + 1

            # connect frame-0-nodes with frame-1-nodes
            build_temporal_connections(temporal_pointpairs, 
                all_nodes ,
                current_nodes_i,
                following_nodes_j, 
                knn_param_temp, 
                self_referencing_edges)

    np_temporal_pointpairs:np.ndarray = np.asarray(temporal_pointpairs)
    t_temporal_pointpairs = torch.from_numpy(np_temporal_pointpairs).to(device)
    
    return t_temporal_pointpairs

def get_and_compute_temporal_edge_indices(
        frames_per_graph:int , 
        graph_dataframe:Dict,
        knn_param:int, self_referencing_edges:bool = False,
        adapt_knn_param = False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )-> torch.Tensor:
    '''
    Returns indices of edges in reference to the stacked center indices.
    Returns:
    t_temporal_pointpairs:torch interger tensor (num_edges,2)
    '''
    # increase knn param because one column will be removed since it 
    # is the self referencing edge connection
    knn_param_temp = None
    if(self_referencing_edges==False):
        knn_param_temp = knn_param + 1
    else:
        knn_param_temp = knn_param
    
    # check if number of object centers is less than the number of neighbors k needed for  KNN
    invalid_frames = None
    if adapt_knn_param:
        invalid_frames = is_invalid_frame(graph_dataframe,knn_param)
        print('invalid_frames',invalid_frames)

    temporal_pointpairs = []

    # print(centers_dict)
    # Get individual centers
    _, centers0 = graph_dataframe["centers_dict"][0]
    _, centers1 = graph_dataframe["centers_dict"][1]
    _, centers2 = graph_dataframe["centers_dict"][2]

    # centers = centers_dict["all"]
    centers = graph_dataframe["centers_list_all"].cpu().numpy()

    # connect frame-0-nodes with frame-1-nodes
    
    # Adapt K-NN parameter according to target time-frame 
    target_frame = 1
    knn_param_temp_0_to_1 = knn_param_temp
    if invalid_frames is not None:
        if (invalid_frames[target_frame]):
            _, centers_target = graph_dataframe["centers_dict"][target_frame]
            knn_param_temp_0_to_1 = len(centers_target) + 1 
            # print("knn_param_temp_1_to_2: ",knn_param_temp_0_to_1)
            # print("length num obj target: ",len(graph_dataframe["centers_dict"][target_frame]))

    for i in range(len(centers0)):
        center = centers0[i]
        center = np.expand_dims(center,axis=0)
        temp = np.append(centers1,center,axis=0)
        #Find nearest_neigbor
        nearest_neigbor = NearestNeighbors(n_neighbors=knn_param_temp_0_to_1, algorithm='ball_tree').fit(temp)
        temporal_indices = nearest_neigbor.kneighbors(temp, return_distance=False)
        #Remove the self referencing edge connection
        if (self_referencing_edges==False):
            temporal_indices = temporal_indices[:,1:]
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
    # Adapt K-NN parameter according to target time-frame 
    target_frame =  2
    
    knn_param_temp_0_to_2 = knn_param_temp
    if invalid_frames is not None:
        if (invalid_frames[target_frame]):
            _, centers_target = graph_dataframe["centers_dict"][target_frame]
            knn_param_temp_0_to_2 = len(centers_target) + 1 
            # print("knn_param_temp_0_to_2: ",knn_param_temp_0_to_2)
            # print("length num obj target: ",len(graph_dataframe["centers_dict"][target_frame]))


    for i in range(len(centers0)):
        center = centers0[i]
        center = np.expand_dims(center,axis=0)
        temp = np.append(centers2,center,axis=0)
        #Find nearest_neigbor
        nearest_neigbor = NearestNeighbors(n_neighbors=knn_param_temp_0_to_2, algorithm='ball_tree').fit(temp)
        temporal_indices = nearest_neigbor.kneighbors(temp, return_distance=False)
        #Remove the self referencing edge connection
        if (self_referencing_edges==False):
            temporal_indices = temporal_indices[:,1:]
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
    target_frame =  2

    knn_param_temp_1_to_2 = knn_param_temp
    if invalid_frames is not None:
        if (invalid_frames[target_frame]):
            _, centers_target = graph_dataframe["centers_dict"][target_frame]
            knn_param_temp_1_to_2 = len(centers_target) + 1 
            # print("knn_param_temp_1_to_2: ",knn_param_temp_1_to_2)
            # print("length num obj target: ",len(graph_dataframe["centers_dict"][target_frame]))

    for i in range(len(centers1)):
        center = centers1[i]
        center = np.expand_dims(center,axis=0)
        temp = np.append(centers2,center,axis=0)
        nearest_neigbor = NearestNeighbors(n_neighbors=knn_param_temp_1_to_2, algorithm='ball_tree').fit(temp)
        temporal_indices = nearest_neigbor.kneighbors(temp, return_distance=False)
        #Remove the self referencing edge connection
        if (self_referencing_edges==False):
            temporal_indices = temporal_indices[:,1:]
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

def compute_edge_feats_dict(edge_ixs_dict:Dict[str,torch.Tensor],
            graph_dataframe:Dict[str,Dict[str,torch.Tensor]],
            mode:str, 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))\
            -> Dict[str, torch.Tensor]:
    """
    Computes a dictionary of edge features among pairs of detections
    Args:
        edge_ixs: Edges tensor with shape (2, num_edges)
        use_cuda: bool, determines whether operations must be performed in GPU
    Returns:
        edge_feats_dict: Dictionary where edge key is a string referring to the attr name, and each val is a 
        torch.tensor of shape (num_edges, num_edge_features)
        with vals of that attribute for each edge.

    """
    assert mode in EDGE_FEATURE_COMPUTATION_MODE,\
        'Incorrect mode string. Please use any of these keywords: {}'.format(EDGE_FEATURE_COMPUTATION_MODE)
    # if mode not in EDGE_FEATURE_COMPUTATION_MODE:
    #     raise ValueError('Incorrect mode string. Please use any of these keywords: {}'.format(EDGE_FEATURE_COMPUTATION_MODE))

    t_edge_ixs = edge_ixs_dict["edges"]
    t_temporal_edges_mask = edge_ixs_dict["temporal_edges_mask"]

    edge_feats_dict = {}
    # Compute features
    t_relative_vectors = None
    if (mode == "relative_position"):
        centers = graph_dataframe["centers_list_all"].cpu().numpy()
        t_relative_vectors = compute_relative_position(centers, t_edge_ixs, device)
        edge_feats_dict['relative_position']= t_relative_vectors

    elif(mode == "edge_type"):
        t_edge_types_one_hot = encode_edge_types(t_temporal_edges_mask, device=device)
        edge_feats_dict['edge_type']= t_edge_types_one_hot
    
    return edge_feats_dict

def compute_relative_position(nodes :np.ndarray,
            t_edge_ixs: torch.Tensor ,
            device:torch.device)-> torch.Tensor:
    relative_vectors = []
        
    for edge in t_edge_ixs:
        reference_ind = edge[0]
        neighbor_ind = edge[1]
        reference_node = nodes[reference_ind]
        neighbor_node = nodes[neighbor_ind]
        relative_vector =  reference_node - neighbor_node
        relative_vectors.append(relative_vector) # List of numpy arrays
    np_relative_vectors = np.asarray(relative_vectors)
    t_relative_vectors = torch.from_numpy(np_relative_vectors).to(device)

    return t_relative_vectors

def encode_edge_types(
            edge_type_mask:torch.Tensor, 
            device:torch.device)-> torch.Tensor:
    # Init one hot encoding tensor
    t_edge_types_one_hot = torch.zeros(edge_type_mask.shape[0],len(edge_types), dtype=torch.float32).to(device)
    # Get one dimensional tensor(num_edge, edge_type)
    t_edge_types = torch.where(edge_type_mask[:,0]==True,1,0).to(device)
    t_edge_types.unsqueeze_(1)

    t_edge_types_one_hot.scatter_(1, t_edge_types, 1)
    return t_edge_types_one_hot
