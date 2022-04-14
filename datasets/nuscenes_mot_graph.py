from ctypes import Union
from typing import Dict, List

import numpy as np
from sklearn.utils import deprecated
import torch
import torch.nn.functional as F
from graph.graph_generation import (add_general_centers,
                                    compute_edge_feats_dict,
                                    get_and_compute_temporal_edge_indices)
from groundtruth_generation.nuscenes_create_gt import (
    generate_edge_label_one_hot, generate_flow_labels)
from nuscenes import NuScenes
from torch_geometric.transforms.to_undirected import ToUndirected
from torch_scatter import scatter_min
from utility import filter_boxes, get_box_centers, is_same_instance
from utils.nuscenes_helper_functions import is_valid_box, is_valid_box_torch
from zmq import device

from datasets.mot_graph import Graph, MOTGraph

# from utils.graph import get_knn_mask

class NuscenesMotGraph(object):

    SPATIAL_SHIFT_TIMEFRAMES = 20
    KNN_PARAM_TEMPORAL = 3

    def __init__(self,nuscenes_handle:NuScenes, start_frame:str , max_frame_dist:int = 3,
                    filterBoxes_categoryQuery:str = None,
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.max_frame_dist = max_frame_dist
        self.nuscenes_handle = nuscenes_handle
        self.start_frame = start_frame
        self.is_possible2construct:bool = self._is_possible2construct()
        self.filterBoxes_categoryQuery = filterBoxes_categoryQuery # Is often 'vehicle.car'
        self.device = device

        # Data-child object for pytorch
        self.graph_obj:Graph = None
        # Dataframe: is a Dict that contains all necessary extra information
        #  for pre and post-processing apart from pytorch computations
        self.graph_dataframe:Dict = None
        if self.is_possible2construct:
            self.graph_dataframe = self._construct_graph_dataframe()

    def _construct_graph_dataframe(self):
        """
        Determines which frames will be in the graph, and creates a DataFrame with its detection's information.

        Args:

        Returns:
            graph_df: DataFrame with rows of scene_df between the selected frames
        """
        graph_dataframe = {}
        # Load Center points for features from LIDAR pointcloud frame of reference
        sensor = 'LIDAR_TOP'
        sample_token = self.start_frame

        # Compute Dict of Lists of Box-objects mapped by integer value that references the timeframe
        # append dict to graph_dataframe
        boxes_dict= {}
        for i in range(self.max_frame_dist):
            # Append new boxes
            sample = self.nuscenes_handle.get('sample', sample_token)
            lidar_top_data = self.nuscenes_handle.get('sample_data', sample['data'][sensor])
            _, boxes, _= self.nuscenes_handle.get_sample_data(lidar_top_data['token'], selected_anntokens=None, use_flat_vehicle_coordinates =False)
            # filter out all object that are not of class self.filterBoxes_categoryQuery
            if( self.filterBoxes_categoryQuery is not None):
                boxes = filter_boxes(self.nuscenes_handle, boxes= boxes, categoryQuery= self.filterBoxes_categoryQuery)
            boxes_dict[i] = boxes

            #Move to next sample
            sample_token = sample["next"]

        graph_dataframe["boxes_dict"] = boxes_dict
        
        # Compute Dict of Lists of Box-objects mapped by integer value that references the timeframe
        # append dict to graph_dataframe
        # Box memory is shared, so no new box memory space is allocated
        centers_dict = {} 
        for box_timeframe, box_list in boxes_dict.items():

            # car_boxes = filter_boxes(self.nuscenes_handle, boxes= box_list, categoryQuery= 'vehicle.car')
            car_boxes = box_list
            centers = get_box_centers(car_boxes)
            centers_dict[box_timeframe] = (car_boxes,centers)

        graph_dataframe["centers_dict"] = centers_dict

        # Combine all lists within boxes Dict into one list
        # Add in chronological order
        # append dict to graph_dataframe
        box_list = []
        for box_list_i_key in range(self.max_frame_dist):
            box_list_i = graph_dataframe["boxes_dict"][box_list_i_key]
            box_list = box_list + box_list_i

        graph_dataframe["boxes_list_all"] = box_list

        # Combine all lists within centers Dict into one list
        # Add in chronological order
        # Shift the different Timeframes by a defined constant NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES * timeframe
        # Ensure that the memory is different than that from the Box-objects
        # List is torch.Tensor
        # append dict to graph_dataframe
        t_centers_list = torch.empty((0,3)).to(self.device)
        for centers_list_i_key in range(self.max_frame_dist):
            _ ,centers_list_i = graph_dataframe["centers_dict"][centers_list_i_key]
            centers_list_i = centers_list_i.copy()
            t_centers_list_i = torch.from_numpy(centers_list_i).to(self.device)
            t_centers_list_i += torch.tensor([0,0,NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES * centers_list_i_key]).to(self.device)
            t_centers_list = torch.cat([t_centers_list, t_centers_list_i], dim = 0 ).to(self.device)

        graph_dataframe["centers_list_all"] = t_centers_list

        return graph_dataframe


    def _get_edge_ixs(self, centers_dict):
        """
        Constructs graph edges by taking pairs of nodes with valid time connections (not in same frame, not too far
        apart in time) and perhaps taking KNNs according to reid embeddings.
        Args:
            centers_dict: torch.tensor with shape (num_nodes, reid_embeds_dim)

        Returns:
            edge_ixs: torch.tensor withs shape (2, num_edges) describes indices of edges, 
            edge_feats_dict: dict with edge features, mainly torch.Tensors e.g (num_edges, num_edge_features)
        """

        add_general_centers(centers_dict,\
                    NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES)
        # print(centers_dict)

        edge_ixs = get_and_compute_temporal_edge_indices(centers_dict,\
                    NuscenesMotGraph.KNN_PARAM_TEMPORAL, device= self.device)

        edge_feats_dict = None
        
        
        # print(centers_dict.keys())

        edge_feats_dict = compute_edge_feats_dict(edge_ixs= edge_ixs,
                            centers_dict=centers_dict, device=self.device)

        return edge_ixs, edge_feats_dict

    def _identify_new_instances_within_graph(self):
        """
        Returns a list of instance_tokens and corresponding node_indices that are considered new instances within the graph-scene
        The instances of the 
        """

        # Safe list of base object instances
        base_key = 0
        base_box_list = self.graph_dataframe["boxes_dict"][base_key]
        # base_center = self.graph_dataframe['centers_dict'][base_key]

        # Put all base tokens into one list for comparison later
        base_box_list_sample_annotation_tokens = [base_box.token for base_box in base_box_list]

        base_box_list_instance_tokens = []
        for sample_annotation_token in base_box_list_sample_annotation_tokens:
            sample_annotation = self.nuscenes_handle.get('sample_annotation', sample_annotation_token)
            instance_token = sample_annotation['instance_token']
            base_box_list_instance_tokens.append(instance_token)
            
        # Init list of new tokens
        new_instance_token_list = []
        new_instance_box_list = []

        for box_list_i_key in self.graph_dataframe["boxes_dict"]:

            box_list_i = self.graph_dataframe["boxes_dict"][box_list_i_key]

            # Extract list of new objects
            if (box_list_i_key != base_key):
                for box in box_list_i:
                    sample_annotation = self.nuscenes_handle.get('sample_annotation', box.token)
                    instance_token = sample_annotation['instance_token']
                    if ((instance_token not in base_box_list_instance_tokens)
                        and
                        (instance_token not in new_instance_token_list)) :
                        new_instance_token_list.append(instance_token)
                        new_instance_box_list.append(box)
        
        #If needed you can remove duplicates with the following
        # new_instance_token_set = set(new_instance_token_list)
        # new_instance_token_list = list(new_instance_token_set)

        return new_instance_token_list, new_instance_box_list
        # return new_instance_token_list

    def assign_edge_labels(self, label_type:str):
        '''
        Generates Edge labels for each edge
        There are two kinds of labels: binary, multiclass
        '''

        label_types = {"binary", "multiclass"}
        if label_type not in label_types:
            raise ValueError('Incorrect label_type string. Please use either "binary" or "multiclass"')

        box_list = self.graph_dataframe["boxes_list_all"]
        centers = self.graph_dataframe['centers_list_all']
        
        new_instance_token_list = None
        if label_type == "multiclass":
            new_instance_token_list, _ = self._identify_new_instances_within_graph()

        flow_labels = []

        #TODO Change the way how to iterate through edges
        t_edges = self.graph_obj.edge_index.T

        for edge in t_edges:
            node_a_center = centers[edge[0]]
            node_b_center = centers[edge[1]]

            node_a_box = box_list[edge[0]]
            node_b_box = box_list[edge[1]]

            # Check that car_box and car_centers match
            if not (is_valid_box_torch(node_a_box,node_a_center,
                    spatial_shift_timeframes= NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES,
                    device= self.device)\
                    and 
                    is_valid_box_torch(node_b_box,node_b_center,
                    spatial_shift_timeframes= NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES,
                    device = self.device)
                    ):
                raise ValueError('A box does not correspond to a selected center')

            str_node_a_sample_annotation = node_a_box.token
            str_node_b_sample_annotation = node_b_box.token

            if label_type == "binary":
                edge_label = self._assign_edge_labels_binary(
                                    str_node_a_sample_annotation,
                                    str_node_b_sample_annotation)

            elif label_type == "multiclass":
                edge_label = self._assign_edge_labels_one_hot(
                                str_node_a_sample_annotation,
                                str_node_b_sample_annotation,
                                new_instances_token_list = new_instance_token_list)

            flow_labels.append(edge_label)

        # Concatenate list of edge_labels
        if label_type == "binary":
            flow_labels = torch.tensor(flow_labels,dtype=torch.uint8)
        elif label_type == "multiclass":
            flow_labels = torch.stack(flow_labels, dim = 0)

        # Transfere to GPU if available
        # flow_labels = torch.FloatTensor(flow_labels)
        flow_labels = flow_labels.to(self.device)

        self.graph_obj.edge_labels = flow_labels

    def _assign_edge_labels_one_hot(self,
                    sample_annotation_token_a:str,
                    sample_annotation_token_b:str,
                    new_instances_token_list:List[str])->torch.Tensor:

        edge_label = generate_edge_label_one_hot(self.nuscenes_handle,
                            sample_annotation_token_a,
                            sample_annotation_token_b,
                            new_instances_token_list = new_instances_token_list,
                            device= self.device)

        return edge_label

    def _assign_edge_labels_binary(self,
                    str_node_a_sample_annotation:str,
                    str_node_b_sample_annotation:str):

        if (is_same_instance(self.nuscenes_handle,
                        str_node_a_sample_annotation,
                        str_node_b_sample_annotation)):
            return 1
        else:
            return 0

    
    @deprecated("This function is obsolete, use 'assign_edge_labels' with \'multiclass\'-label instead")
    def assign_edge_labels_one_hot(self):

        box_list = self.graph_dataframe["boxes_list_all"]
        centers = self.graph_dataframe['centers_list_all']
        
        new_instance_token_list, new_instance_box_list = self._identify_new_instances_within_graph()

        flow_labels = []

        #TODO Change the way how to iterate through edges
        t_edges = self.graph_obj.edge_index.T

        for edge in t_edges:
            node_a_center = centers[edge[0]]
            node_b_center = centers[edge[1]]

            node_a_box = box_list[edge[0]]
            node_b_box = box_list[edge[1]]

            # Check that car_box and car_centers match
            if not (is_valid_box(node_a_box,node_a_center,
                    spatial_shift_timeframes= NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES)\
                    and is_valid_box(node_b_box,node_b_center,
                    spatial_shift_timeframes= NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES)
                    ):
                raise ValueError('A box does not correspond to a selected center')

            str_node_a_sample_annotation = node_a_box.token
            str_node_b_sample_annotation = node_b_box.token

            edge_label = generate_edge_label_one_hot(self.nuscenes_handle,
                            str_node_a_sample_annotation,
                            str_node_b_sample_annotation, 
                            new_instances_token_list = new_instance_token_list,
                            device = self.device)

            flow_labels.append(edge_label)

        # Concatenate list of edge_labels
        flow_labels = torch.cat(flow_labels, dim = 0)

        # Transfere to GPU
        # flow_labels = torch.FloatTensor(flow_labels)
        flow_labels = flow_labels.to(self.device)

        self.graph_obj.edge_labels = flow_labels

    @deprecated("This function is obsolete, use 'assign_edge_labels' with \'binary\'-label instead")
    def assign_edge_labels_binary(self):
        """
        Assigns self.graph_obj edge labels (tensor with shape (num_edges,)), with labels defined according to the
        network flow MOT formulation
        """

        box_list = []
        for box_list_i_key in self.graph_dataframe["boxes_dict"]:
            # 
            box_list_i = self.graph_dataframe["boxes_dict"][box_list_i_key]
            box_list = box_list + box_list_i

        # flow_labels = generate_flow_labels(nuscenes_handle = self.nuscenes_handle,
        #                     temporal_pointpairs = self.graph_obj.edge_attr,
        #                     car_box_list= box_list, centers= self.graph_dataframe['centers_dict']["all"])
        
        # Transfere and compute information on cpu
        nuscenes_handle = self.nuscenes_handle
        temporal_pointpairs = (self.graph_obj.edge_index).cpu().T
        centers = self.graph_dataframe['centers_dict']["all"]
        car_box_list = box_list

        flow_labels = []
        for point_pair in temporal_pointpairs:
            node_a_center = centers[point_pair[0]]
            node_b_center = centers[point_pair[1]]

            node_a_box = car_box_list[point_pair[0]]
            node_b_box = car_box_list[point_pair[1]]

            # Check that car_box and car_centers match
            if not (is_valid_box(node_a_box,node_a_center,
                    spatial_shift_timeframes= NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES)\
                    and is_valid_box(node_b_box,node_b_center,
                    spatial_shift_timeframes= NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES)
                    ):
                raise ValueError('A box does not correspond to a selected center')

            str_node_a_sample_annotation = node_a_box.token
            str_node_b_sample_annotation = node_b_box.token

            if (is_same_instance(nuscenes_handle,str_node_a_sample_annotation \
                                    ,str_node_b_sample_annotation)):
                flow_labels.append(1)
            else:
                flow_labels.append(0)


        # Transfere to GPU           
        flow_labels = torch.FloatTensor(flow_labels)
        flow_labels = flow_labels.to(self.device)

        self.graph_obj.edge_labels = flow_labels

    def _is_possible2construct(self):
        init_sample = self.nuscenes_handle.get('sample',self.start_frame)
        scene_token = init_sample["scene_token"]
        scene = self.nuscenes_handle.get("scene",scene_token)

        last_sample_token =""
        sample_token = self.start_frame
        i= 0
        while(last_sample_token == ""):
            sample = self.nuscenes_handle.get('sample', sample_token)
            sample_token = sample["next"]
            i += 1
            if(sample["token"]== scene['last_sample_token']):
                last_sample_token = scene['last_sample_token']
        if i < self.max_frame_dist:
            return False
        else:
            return True
        

    def construct_graph_object(self):
        """
        Constructs the entire Graph object to serve as input to the MPN, and stores it in self.graph_obj,
        """
        centers_dict = self.graph_dataframe["centers_dict"]
        # Determine graph connectivity (i.e. edges) and compute edge features
        edge_ixs, edge_feats_dict = self._get_edge_ixs(centers_dict)

        # Prepare Inputs/ bring into apropiate shape to generate graph/object
        centers = centers_dict["all"]
        t_centers = torch.from_numpy(centers).to(self.device)

        edge_feats = edge_feats_dict['relative_vectors']

        # Transpose if not Graph connectivity in COO format with shape :obj:`[2, num_edges]
        if edge_ixs.shape[1] == 2:
            edge_ixs = edge_ixs.T

        # print('edge_feats:',edge_feats.shape)
        edge_feats = torch.cat((edge_feats, edge_feats), dim = 0)
        # print('edge_feats:',edge_feats.shape)
        # print('edge_ixs:',edge_ixs.shape)
        edge_ixs = torch.cat((edge_ixs, torch.stack((edge_ixs[1], edge_ixs[0]))), dim=1)
        # print('edge_ixs:',edge_ixs.shape)

        self.graph_obj = Graph(x = t_centers,
                               edge_attr = edge_feats,
                               edge_index = edge_ixs)
        
        # Ensure that graph is undirected.
        if self.graph_obj.is_directed():
            print(self.graph_obj)
            undirectTransfomer = ToUndirected()
            self.graph_obj = undirectTransfomer(self.graph_obj)
            print(self.graph_obj)
        
        self.graph_obj.to(self.device)
