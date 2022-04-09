from typing import Dict
import numpy as np
import torch
import  torch.nn.functional as F
from nuscenes import NuScenes
from torch_scatter import scatter_min

from utility import get_box_centers, filter_boxes
from graph.graph_generation import get_and_compute_temporal_edge_indices,\
                             add_general_centers, compute_edge_feats_dict

from groundtruth_generation.nuscenes_create_gt import generate_flow_labels
from utils.nuscenes_helper_functions import is_valid_box
from utility import is_same_instance

from datasets.mot_graph import MOTGraph, Graph
# from utils.graph import get_knn_mask

class NuscenesMotGraph(object):

    SPATIAL_SHIFT_TIMEFRAMES = 20
    KNN_PARAM_TEMPORAL = 3

    def __init__(self,nuscenes_handle:NuScenes, start_frame:str , max_frame_dist = 3):
        self.max_frame_dist = max_frame_dist
        self.nuscenes_handle = nuscenes_handle
        self.start_frame = start_frame
        self.is_possible2construct:bool = self._is_possible2construct()
        
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
            valid_frames: list of selected frames

        """
        graph_dataframe = {}
        # Load Center points for features from LIDAR pointcloud frame of reference
        sensor = 'LIDAR_TOP'
        sample_token = self.start_frame

        boxes_dict= {}
        for i in range(self.max_frame_dist):
            # Append new boxes
            sample = self.nuscenes_handle.get('sample', sample_token)
            lidar_top_data = self.nuscenes_handle.get('sample_data', sample['data'][sensor])
            _, boxes, _= self.nuscenes_handle.get_sample_data(lidar_top_data['token'], selected_anntokens=None, use_flat_vehicle_coordinates =False)
            boxes_dict[i] = boxes

            #Move to next sample
            sample_token = sample["next"]

        graph_dataframe["boxes_dict"] = boxes_dict

        centers_dict = {} 
        for box_timeframe, box_list in boxes_dict.items():

            car_boxes = filter_boxes(self.nuscenes_handle, boxes= box_list, categoryQuery= 'vehicle.car')
            centers = get_box_centers(car_boxes)
            centers_dict[box_timeframe] = (car_boxes,centers)

        graph_dataframe["centers_dict"] = centers_dict

        return graph_dataframe


    def _get_edge_ixs(self, centers_dict):
        """
        Constructs graph edges by taking pairs of nodes with valid time connections (not in same frame, not too far
        apart in time) and perhaps taking KNNs according to reid embeddings.
        Args:
            centers_dict: torch.tensor with shape (num_nodes, reid_embeds_dim)

        Returns:
            torch.tensor withs shape (2, num_edges)
        """
        use_cuda = True

        add_general_centers(centers_dict,\
                    NuscenesMotGraph.SPATIAL_SHIFT_TIMEFRAMES)
        # print(centers_dict)

        edge_ixs = get_and_compute_temporal_edge_indices(centers_dict,\
                    NuscenesMotGraph.KNN_PARAM_TEMPORAL, use_cuda=use_cuda)

        edge_feats_dict = None
        
        
        # print(centers_dict.keys())

        edge_feats_dict = compute_edge_feats_dict(edge_ixs= edge_ixs,
                            centers_dict=centers_dict, use_cuda=use_cuda)

        return edge_ixs, edge_feats_dict

    def assign_edge_labels(self):
        """
        Assigns self.graph_obj edge labels (tensor with shape (num_edges,)), with labels defined according to the
        network flow MOT formulation
        """

        box_list = []
        for box_list_i_key in self.graph_dataframe["boxes_dict"]:
            box_list.append(self.graph_dataframe["boxes_dict"][box_list_i_key])

        # flow_labels = generate_flow_labels(nuscenes_handle = self.nuscenes_handle,
        #                     temporal_pointpairs = self.graph_obj.edge_attr,
        #                     car_box_list= box_list, centers= self.graph_dataframe['centers_dict']["all"])
        nuscenes_handle = self.nuscenes_handle
        temporal_pointpairs = (self.graph_obj.edge_index).cpu()
        centers = self.graph_dataframe['centers_dict']["all"]
        # centers = torch.from_numpy(centers.copy())
        car_box_list = box_list

        # print(temporal_pointpairs.device)
        # print(centers.device)
        # print(car_box_list.device)

        flow_labels = []
        for point_pair in temporal_pointpairs:
            node_a_center = centers[point_pair[0]]
            node_b_center = centers[point_pair[1]]
            # print(node_a_center)
            # print(node_b_center)

            # node_a_box = get_box(car_box_list, node_a_center)
            # node_b_box = get_box(car_box_list, node_b_center)

            node_a_box = car_box_list[point_pair[0]]
            node_b_box = car_box_list[point_pair[1]]

            if not (is_valid_box(node_a_box,node_a_center) and is_valid_box(node_b_box,node_b_center)):
                print('invalid boxes!!!')
                raise ValueError('A box does not correspond to a selected center')

            str_node_a_sample_annotation = node_a_box.token
            str_node_b_sample_annotation = node_b_box.token

            if (is_same_instance(nuscenes_handle,str_node_a_sample_annotation \
                                    ,str_node_b_sample_annotation)):
                flow_labels.append(1)
            else:
                flow_labels.append(0)

        # Check that labels are torch.Tensor
        if type(flow_labels) == np.ndarray:
            flow_labels = torch.from_numpy(flow_labels)

        # Transfere to GPU                     
        flow_labels = flow_labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.graph_obj.edge_labels = flow_labels

        # #Get unique instance id of object
        # ids = torch.as_tensor(self.graph_df.id.values, device=self.graph_obj.edge_index.device)
        # # Get the connected inst-ids of each edge  
        # per_edge_ids = torch.stack([ids[self.graph_obj.edge_index[0]], ids[self.graph_obj.edge_index[1]]])
        # # compare if the connected ids are the same -> if they connect the same object instance
        # same_id = (per_edge_ids[0] == per_edge_ids[1]) & (per_edge_ids[0] != -1)
        # #safe edge ids connecting the same object (active edge)
        # same_ids_ixs = torch.where(same_id)
        # # Get the same-id-edges
        # same_id_edges = self.graph_obj.edge_index.T[same_id].T

        # time_dists = torch.abs(same_id_edges[0] - same_id_edges[1])

        # # For every node, we get the index of the node in the future (resp. past) with the same id that is closest in time
        # future_mask = same_id_edges[0] < same_id_edges[1]
        # active_fut_edges = scatter_min(time_dists[future_mask], same_id_edges[0][future_mask], dim=0, dim_size=self.graph_obj.num_nodes)[1]
        # original_node_ixs = torch.cat((same_id_edges[1][future_mask], torch.as_tensor([-1], device = same_id.device))) # -1 at the end for nodes that were not present
        # active_fut_edges = original_node_ixs[active_fut_edges] # Recover the node id of the corresponding
        # fut_edge_is_active = active_fut_edges[same_id_edges[0]] == same_id_edges[1]

        # # Analogous for past edges
        # past_mask = same_id_edges[0] > same_id_edges[1]
        # active_past_edges = scatter_min(time_dists[past_mask], same_id_edges[0][past_mask], dim = 0, dim_size=self.graph_obj.num_nodes)[1]
        # original_node_ixs = torch.cat((same_id_edges[1][past_mask], torch.as_tensor([-1], device = same_id.device))) # -1 at the end for nodes that were not present
        # active_past_edges = original_node_ixs[active_past_edges]
        # past_edge_is_active = active_past_edges[same_id_edges[0]] == same_id_edges[1]

        # # Recover the ixs of active edges in the original edge_index tensor o
        # active_edge_ixs = same_ids_ixs[0][past_edge_is_active | fut_edge_is_active]
        # self.graph_obj.edge_labels = torch.zeros_like(same_id, dtype = torch.float)
        # self.graph_obj.edge_labels[active_edge_ixs] = 1

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

        # # Load Center points for features from LIDAR pointcloud frame of reference
        # sensor = 'LIDAR_TOP'
        # sample_token = self.start_frame
            
        # boxes_dict= {}
        # for i in range(self.max_frame_dist):
        #     # Append new boxes
        #     sample = self.nuscenes_handle.get('sample', sample_token)
        #     lidar_top_data = self.nuscenes_handle.get('sample_data', sample['data'][sensor])
        #     _, boxes, _= self.nuscenes_handle.get_sample_data(lidar_top_data['token'], selected_anntokens=None, use_flat_vehicle_coordinates =False)
        #     boxes_dict[i] = boxes

        #     #Move to next sample
        #     sample_token = sample["next"]

        # centers_dict = {} 
        # for box_timeframe, box_list in boxes_dict.items():

        #     car_boxes = filter_boxes(self.nuscenes_handle, boxes= box_list, categoryQuery= 'vehicle.car')
        #     centers = get_box_centers(car_boxes)
        #     centers_dict[box_timeframe] = (car_boxes,centers)

        centers_dict = self.graph_dataframe["centers_dict"]
        # Determine graph connectivity (i.e. edges) and compute edge features
        edge_ixs, edge_feats_dict = self._get_edge_ixs(centers_dict)

        centers = centers_dict["all"]
        t_centers = torch.from_numpy(centers)

        edge_feats = edge_feats_dict['relative_vectors']

        # self.graph_obj = Graph(x = centers,
        #                        edge_attr = torch.cat((edge_feats, edge_feats), dim = 0),
        #                        edge_index = torch.cat((edge_ixs, torch.stack((edge_ixs[1], edge_ixs[0]))), dim=1))
        self.graph_obj = Graph(x = t_centers,
                               edge_attr = edge_feats,
                               edge_index = edge_ixs)

        self.graph_obj.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
