# from ctypes import Union
from pickle import TRUE
from tkinter.tix import Tree
from turtle import shape
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from graph.graph_generation import (compare_two_edge_indices_matrices,
                                    compute_edge_feats_dict,
                                    get_and_compute_spatial_edge_indices,
                                    get_and_compute_spatial_edge_indices_new,
                                    get_and_compute_temporal_edge_indices,
                                    get_and_compute_temporal_edge_indices_new)
from groundtruth_generation.nuscenes_create_gt import (
    generate_edge_label_one_hot, generate_flow_labels)
from matplotlib.pyplot import box
from matplotlib.style import available
from pyquaternion import Quaternion
from sklearn.utils import deprecated
from torch_geometric.transforms.to_undirected import ToUndirected
from utility import filter_boxes, get_box_centers, is_same_instance
from utils.nuscenes_helper_functions import (determine_class_id,
                                             get_sample_data_table,
                                             is_valid_box, is_valid_box_torch,
                                             skip_sample_token, transform_boxes_from_world_2_sensor, transform_detections_lidar2world_frame)

# For dummy objects
from datasets.mot_graph import Graph
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix

# from utils.graph import get_knn_mask

class NuscenesMotGraph(object):

    NODE_FEATURE_MODES = {"only_centers", "centers_and_time"}
    EDGE_LABEL_TYPES = {"binary", "multiclass"}
    DUMMY_TOKEN = "dummy_token"

    def __init__(self,nuscenes_handle:NuScenes, start_frame:str , max_frame_dist:int = 3,
                    filterBoxes_categoryQuery:Union[str,List[str]] = None,
                    construction_possibility_checked = True,
                    adapt_knn_param = False,
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    dataset_params: dict = None,
                    detection_dict: Dict[str,Any] = None,
                    inference_mode: bool = False):
        
        self.max_frame_dist = max_frame_dist
        self.nuscenes_handle = nuscenes_handle
        self.start_frame = start_frame
        self.is_possible2construct:bool = self._is_possible2construct()
        self.filterBoxes_categoryQuery = filterBoxes_categoryQuery # Is often 'vehicle.car'
        self.adapt_knn_param = adapt_knn_param
        self.device = device
        self.use_gt_detections:bool = True
        self.det_file_path:str = None
        self.detection_dict: Dict[str,Any] = None
        self.inference_mode:bool = inference_mode


        self.SPATIAL_SHIFT_TIMEFRAMES, self.KNN_PARAM_TEMPORAL , self.KNN_PARAM_SPATIAL = None, None, None
        if dataset_params is not None:
            self.KNN_PARAM_SPATIAL:int = dataset_params["graph_construction_params"]["spatial_knn_num_neighbors"]
            self.KNN_PARAM_TEMPORAL:int = dataset_params["graph_construction_params"]["temporal_knn_num_neighbors"]
            self.SPATIAL_SHIFT_TIMEFRAMES:int = dataset_params["graph_construction_params"]["spatial_shift_timeframes"]

            self.MAX_TEMPORAL_EDGE_LENGTH:int = self.max_frame_dist - 1
            if (("max_temporal_edge_length" in  dataset_params["graph_construction_params"]) \
                and (dataset_params["graph_construction_params"]["max_temporal_edge_length"] is not None)):
                self.MAX_TEMPORAL_EDGE_LENGTH:int = dataset_params["graph_construction_params"]["max_temporal_edge_length"]

            if "use_gt_detections" in dataset_params:
                self.use_gt_detections:bool = dataset_params["use_gt_detections"]
                if "det_file_path" in dataset_params:
                    self.det_file_path:str = dataset_params["det_file_path"]
                    self.detection_dict = detection_dict
                    
        else:
            self.SPATIAL_SHIFT_TIMEFRAMES = 20
            self.KNN_PARAM_TEMPORAL = 3
            self.KNN_PARAM_SPATIAL = 3
            self.MAX_TEMPORAL_EDGE_LENGTH:int = 2
            
        assert (self.use_gt_detections == False and self.detection_dict is not None) \
                or (self.use_gt_detections == True and self.detection_dict is None)

        # Data-child object for pytorch
        self.graph_obj:Graph = None
        # Dataframe: is a Dict that contains all necessary extra information
        #  for pre and post-processing apart from pytorch computations
        self.graph_dataframe:Dict = None

        if construction_possibility_checked:
            assert self.is_possible2construct, "Graph object cannot be be built! Probably there are not enough frames available before the scene ends"
        
        if self.is_possible2construct:
            self.graph_dataframe = self._construct_graph_dataframe()
        


    def _construct_dummy_boxes(self, num_needed_boxes: int) -> List[Box]:
        boxes = []
        for i in range(num_needed_boxes):
            center: List[float] = [0, 0, 0]
            size: List[float] = [-1, -1, -1] #width, length, height.
            orientation:Quaternion = Quaternion([0,0,0,0])
            label: int = -1
            name: str = "dummy"
            token: str = self.DUMMY_TOKEN
            box = Box(center,
                            size,
                            orientation,
                            label = label,
                            name=name,
                            token=token)
            # print(box)
            boxes.append(box)
        return boxes

    def _load_detections_from_frame_from_sensor(self, sample_token:str,
                        sensor_channel:str, 
                        load_given_detections:bool) -> List[Box]:
        """
        Returns a list of Box objects
        Loads either sample_annotations as detections from nuscenes Database 
            or from given Detections (e.g. baseline detections)
        """
        if load_given_detections:
            if sample_token in self.detection_dict:
                # Boxes are already filtered to only include detections from the nuscenes tracking challenge
                boxes:List[Box] = self.detection_dict[sample_token]
                initial_boxes:List[Box] = [box.copy() for box in boxes]
                ##################################################################
                # Only need to filter out specific classes if wanted... but not necessary for challenge
                # filter_detection_boxes()
                ###################################################################
                # Transforms detections boxes from world frame into sensor (LIDAR) frame
                # Get transforms
                transform_boxes_from_world_2_sensor(boxes, self.nuscenes_handle, sensor_channel, sample_token)
                for i,box in enumerate(boxes):
                    translation_world, orientation_world = transform_detections_lidar2world_frame(self.nuscenes_handle,box.center.tolist(),box.orientation, sample_token)
                    intial_box: Box = initial_boxes[i]
                    assert np.sum(intial_box.center - np.asarray(translation_world)) < 1e-8
                    assert orientation_world.absolute_distance(orientation_world, intial_box.orientation ) < 1e-8
            else:
                boxes = []
        else:
            # Append new boxes
            # sample = self.nuscenes_handle.get('sample', sample_token)
            # lidar_top_data_old = self.nuscenes_handle.get('sample_data', sample['data'][sensor_channel])
            lidar_top_data = get_sample_data_table(self.nuscenes_handle, sensor_channel, sample_token)

            _, boxes, _= self.nuscenes_handle.get_sample_data(lidar_top_data['token'], selected_anntokens=None, use_flat_vehicle_coordinates =False)
        
            # filter out all object that are not of class self.filterBoxes_categoryQuery
            if( self.filterBoxes_categoryQuery is not None):
                boxes = filter_boxes(self.nuscenes_handle, boxes= boxes, categoryQuery= self.filterBoxes_categoryQuery)
        
        # If the graph is allowed to change and adapt to scenes with less than k objects, 
        # then timeframes with at least one detection can be used to build a graph
        if(self.adapt_knn_param == True and len(boxes)==0):
            boxes.extend(self._construct_dummy_boxes(1))
        # Embed dummy objects if number of objects is smaller then any knn-Parameter
        # this will also catch cases where no objects are left after filtering 
        # there must be at least k + 1 elements such that one element can have k neighbors
        if ((len(boxes) < (self.KNN_PARAM_SPATIAL + 1)) \
            or (len(boxes) < (self.KNN_PARAM_TEMPORAL + 1)))\
            and (self.adapt_knn_param == False):
            spatial_difference = self.KNN_PARAM_SPATIAL + 1 - len(boxes)
            temporal_difference = self.KNN_PARAM_TEMPORAL + 1 - len(boxes)
            num_needed_boxes = max(spatial_difference, temporal_difference)
            boxes.extend(self._construct_dummy_boxes(num_needed_boxes))

        return boxes

    def _load_detections(self, graph_dataframe:dict ):
        # Load Center points for features from LIDAR pointcloud frame of reference
        sensor_channel = 'LIDAR_TOP'
        sample_token = self.start_frame

        # Decide if load from given detection file
        load_given_detections = True
        if self.use_gt_detections: 
            load_given_detections = False

        # Compute Dict of Lists of Box-objects mapped by integer value that references the timeframe
        # append dict to graph_dataframe
        boxes_dict= {}
        for i in range(self.max_frame_dist):
            boxes = self._load_detections_from_frame_from_sensor(sample_token, sensor_channel, load_given_detections)
            boxes_dict[i] = boxes
            #Move to next sample
            sample_token = skip_sample_token(sample_token,0,self.nuscenes_handle)

        graph_dataframe["boxes_dict"] = boxes_dict

        # Combine all lists within boxes Dict into one list
        # Add in chronological order
        # append dict to graph_dataframe
        box_list: List[Box] = []
        for box_list_i_key in range(self.max_frame_dist):
            box_list_i = graph_dataframe["boxes_dict"][box_list_i_key]
            box_list = box_list + box_list_i

        graph_dataframe["boxes_list_all"] = box_list

    def _construct_graph_dataframe(self):
        """
        Determines which frames will be in the graph, and creates a DataFrame with its detection's information.

        Args:

        Returns:
            graph_df: DataFrame with rows of scene_df between the selected frames
        """
        graph_dataframe = {}
        self._load_detections(graph_dataframe)
        
        boxes_dict = graph_dataframe["boxes_dict"]

        # Compute Dict of Lists of Box-objects mapped by integer value that references the timeframe
        # append dict to graph_dataframe
        # Box memory is shared, so no new box memory space is allocated
        centers_dict = {} 
        for box_timeframe, box_list in boxes_dict.items():
            car_boxes = box_list
            centers = get_box_centers(car_boxes)
            centers_dict[box_timeframe] = (car_boxes,centers)

        graph_dataframe["centers_dict"] = centers_dict
        # print("centers_dict",centers_dict)
        # print("centers_dict",len(centers_dict))

        # Combine all lists within centers Dict into one list
        # Add in chronological order
        # Shift the different Timeframes by a defined constant self.SPATIAL_SHIFT_TIMEFRAMES * timeframe
        # Ensure that the memory is different than that from the Box-objects
        # List is torch.Tensor
        # append dict to graph_dataframe
        computed_differences = 0
        t_centers_list:torch.Tensor = torch.empty((0,3)).to(self.device)
        for centers_list_i_key in range(self.max_frame_dist):
            _ ,centers_list_i = graph_dataframe["centers_dict"][centers_list_i_key]
            centers_list_i:np.ndarray = centers_list_i.copy()
            t_centers_list_i:torch.Tensor = torch.from_numpy(centers_list_i).to(self.device)
            # print("t_centers_list_i",t_centers_list_i)
            # print("t_centers_list_i.shape",t_centers_list_i.shape)
            t_centers_list_i += torch.tensor([0,0,self.SPATIAL_SHIFT_TIMEFRAMES * centers_list_i_key]).to(self.device)
            t_centers_list = torch.cat([t_centers_list, t_centers_list_i], dim = 0 ).to(self.device)

            computed_differences = torch.mean(t_centers_list_i[:,2]) - computed_differences
        
        assert computed_differences > (self.SPATIAL_SHIFT_TIMEFRAMES - 5) \
                and computed_differences < (self.SPATIAL_SHIFT_TIMEFRAMES + 5)
        graph_dataframe["centers_list_all"] = t_centers_list

        # Add tensor that encodes
        t_frame_number = torch.zeros((t_centers_list.shape[0],1), dtype=torch.int8).to(self.device)
        current_row = 0
        for frame_i in range(self.max_frame_dist):
            num_samples_frame_i = len(graph_dataframe["boxes_dict"][frame_i])
            timeframe_i = torch.ones(num_samples_frame_i,1, dtype=torch.int8).to(self.device) * frame_i
            next_row = current_row + num_samples_frame_i
            t_frame_number[current_row:next_row,:] = timeframe_i
            current_row += num_samples_frame_i
            current_row == next_row

        graph_dataframe["timeframes_all"] = t_frame_number
        
        # Add object class according to tracking own classes 
        class_ids =  torch.zeros((t_centers_list.shape[0],1), dtype=torch.int8).to(self.device)
        list_of_boxes:List[Box] = graph_dataframe["boxes_list_all"]
        for i,box in enumerate(list_of_boxes):
            if self.use_gt_detections:
                class_id = determine_class_id(box.name)
            else:
                class_id = box.label
            class_ids [i] = class_id
        graph_dataframe["class_ids"] = class_ids
        assert (graph_dataframe["class_ids"]!=0).all(), "some nodes were not assigned a suitable class id"
        
        # Detection scores 
        list_of_boxes:List[Box] = graph_dataframe["boxes_list_all"]
        detection_scores = torch.ones(len(list_of_boxes)).to(self.device)
        for i,box in enumerate(list_of_boxes):
            if not self.use_gt_detections:
                detection_scores[i] = box.score
        graph_dataframe["detection_scores"] = detection_scores

        # Add list of available sample_tokens in this graph
        available_sample_tokens: List[str] = []
        sample_token = self.start_frame
        for frame_i in range(self.max_frame_dist):
            available_sample_tokens.append(sample_token)
            sample = self.nuscenes_handle.get('sample', sample_token)
            sample_token = sample["next"]
        assert available_sample_tokens[0] != available_sample_tokens[1]
        graph_dataframe["available_sample_tokens"] = available_sample_tokens

        # Add sample_token for each node. Important for evaluation
        sample_tokens = []
        for timeframe_i in range(self.max_frame_dist):
            boxes = graph_dataframe["boxes_dict"][timeframe_i]
            num_boxes_timeframe_i = len(boxes)
            current_sample_token = graph_dataframe["available_sample_tokens"][timeframe_i]
            current_sample_list :List[str]= [current_sample_token for i in range(num_boxes_timeframe_i)]
            sample_tokens.extend(current_sample_list)
        assert len(sample_tokens) == len(graph_dataframe["boxes_list_all"])
        graph_dataframe["sample_tokens"] = sample_tokens

        return graph_dataframe


    def _get_edge_ixs(self, mode:str):
        """
        Constructs graph edges by taking pairs of nodes with valid time connections (not in same frame, not too far
        apart in time) and perhaps taking KNNs according to reid embeddings.
        Args:
        Returns:
            edge_ixs: torch.tensor withs shape (2, num_edges) describes indices of edges, 
            edge_feats_dict: dict with edge features, mainly torch.Tensors e.g (num_edges, num_edge_features)
        """
        # Compute Spatial Edges
        t_spatial_edge_ixs = None
        # t_spatial_edge_ixs = get_and_compute_spatial_edge_indices(
        #             self.max_frame_dist,
        #             self.graph_dataframe,
        #             self.KNN_PARAM_SPATIAL,
        #             adapt_knn_param = self.adapt_knn_param,
        #             device= self.device)
        t_spatial_edge_ixs_new = get_and_compute_spatial_edge_indices_new(self.max_frame_dist,
                    self.graph_dataframe,
                    self.KNN_PARAM_SPATIAL,
                    adapt_knn_param = self.adapt_knn_param,
                    device= self.device)
        t_spatial_edge_ixs = t_spatial_edge_ixs_new
        # assert compare_two_edge_indices_matrices(t_spatial_edge_ixs_new,t_spatial_edge_ixs),"New method does not return the same edge indices as the old method!!!"

        # Compute Temporal Edges
        t_temporal_edge_ixs = None
        # t_temporal_edge_ixs = get_and_compute_temporal_edge_indices(
        #             self.max_frame_dist,
        #             self.graph_dataframe,
        #             self.KNN_PARAM_TEMPORAL,
        #             adapt_knn_param = self.adapt_knn_param,
        #             device= self.device)
        t_temporal_edge_ixs_new = get_and_compute_temporal_edge_indices_new(
                    self.max_frame_dist,
                    self.graph_dataframe,
                    self.KNN_PARAM_TEMPORAL,
                    adapt_knn_param = self.adapt_knn_param,
                    device= self.device, max_length_temporal_edges = self.MAX_TEMPORAL_EDGE_LENGTH)
        # assert compare_two_edge_indices_matrices(t_temporal_edge_ixs,t_temporal_edge_ixs_new),"New method does not return the same edge indices as the old method!!!"
        t_temporal_edge_ixs = t_temporal_edge_ixs_new

        #TODO Join temporal and spatial edges but also generate a mask to filter them
        
        # Combine Edge indices into one tensor 
        # and compute a mask to be able to extract temporal edges
        t_edge_ixs = torch.cat([t_temporal_edge_ixs, t_spatial_edge_ixs]).to(self.device)
        t_temporal_mask = torch.ones_like(t_temporal_edge_ixs, dtype=torch.bool).to(self.device)
        t_spatial_mask = torch.zeros_like(t_spatial_edge_ixs, dtype=torch.bool).to(self.device)
        t_temporal_edges_mask = torch.cat([t_temporal_mask, t_spatial_mask]).to(self.device)

        t_test = torch.masked_select(t_edge_ixs, t_temporal_edges_mask)
        t_test_compare = t_temporal_edge_ixs.view((1,-1)).squeeze()
        assert torch.equal(t_test,t_test_compare)
        
        edge_ixs_dict = {}
        edge_ixs_dict["edges"] = t_edge_ixs
        edge_ixs_dict["temporal_edges_mask"] = t_temporal_edges_mask

        edge_feats_dict = None

        edge_feats_dict = compute_edge_feats_dict(edge_ixs_dict,
                            graph_dataframe=self.graph_dataframe,
                            mode= mode, device=self.device)

        

        return edge_ixs_dict, edge_feats_dict

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

    def _contains_dummy_objects(self, boxes:List[Box]=None) -> bool:
        """
        Return True if list of filtered 3D-Detections (3D Bounding Boxes class) contains dummy boxes.
        This is probably due to number of object in a frame being lower than the Knn-Parameter.
        """
        box_list = boxes
        if box_list is None:
            box_list = self.graph_dataframe["boxes_list_all"]
        
        dummyObjectFlag = False
        i = 0 
        while (i < len(box_list)) and (dummyObjectFlag== False):
            box = box_list[i]
            if(box.token == self.DUMMY_TOKEN):
                dummyObjectFlag = True
            i += 1
        return dummyObjectFlag

    def contains_dummy_objects(self, boxes:List[Box]=None) -> bool:
        return self._contains_dummy_objects(boxes=boxes)

    def assign_edge_labels(self, label_type:str):
        '''
        Generates Edge labels for each edge
        There are two kinds of labels: binary, multiclass
        '''
        self.label_type = label_type
        
        
        assert label_type in self.EDGE_LABEL_TYPES, \
            'Incorrect label_type string. Please use either {}'.format(self.EDGE_LABEL_TYPES)
        # if label_type not in label_types:
        #     raise ValueError('Incorrect label_type string. Please use either "binary" or "multiclass"')

        box_list = self.graph_dataframe["boxes_list_all"]
        centers = self.graph_dataframe['centers_list_all']
        
        flow_labels = []

        # Check if invalid Graph containing dummy objects
        dummyObjectFlag = self._contains_dummy_objects()
        
        if dummyObjectFlag:
            t_edges = self.graph_obj.edge_index.T
            num_edges = t_edges.shape[0]
            if label_type == "binary":
                flow_labels = torch.zeros(num_edges, dtype=torch.float32)
            elif label_type == "multiclass":
                flow_labels = torch.zeros(num_edges,3, dtype=torch.float32)
                flow_labels[:,2] + 1 
        else:
            new_instance_token_list = None
            if label_type == "multiclass":
                new_instance_token_list, _ = self._identify_new_instances_within_graph()

            #TODO Change the way how to iterate through edges
            t_edges = self.graph_obj.edge_index.T

            for edge in t_edges:
                node_a_center = centers[edge[0]]
                node_b_center = centers[edge[1]]

                node_a_box = box_list[edge[0]]
                node_b_box = box_list[edge[1]]

                # Check that car_box and car_centers match
                if not (is_valid_box_torch(node_a_box,node_a_center,
                        spatial_shift_timeframes= self.SPATIAL_SHIFT_TIMEFRAMES,
                        device= self.device, num_frames = self.max_frame_dist)\
                        and 
                        is_valid_box_torch(node_b_box,node_b_center,
                        spatial_shift_timeframes= self.SPATIAL_SHIFT_TIMEFRAMES,
                        device = self.device, num_frames = self.max_frame_dist)
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
                flow_labels = torch.tensor(flow_labels,dtype=torch.float32)
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

    def _load_node_features(self, node_feature_mode:str) -> torch.Tensor:
        '''
        Returns the corresponding features from self.graphdataframe 
        depending on the given mode
        Args:
        node_feature_mode: string that defines mode
        Returns:
        t_node_features: torch.Tensor(num_object_samples_over_time, num_node_features) with all features 
        '''
        
        if(node_feature_mode not in NuscenesMotGraph.NODE_FEATURE_MODES):
            str_error_message = 'Incorrect label_type string!\n'\
                    + ' Please use any of these Keywords: {}'.format(NuscenesMotGraph.NODE_FEATURE_MODES)
            raise ValueError(str_error_message)

        t_node_features = None

        # num_node_features = 3 (x,y,z)
        if(node_feature_mode == "only_centers"):
            t_node_features = self.graph_dataframe["centers_list_all"]
        
        # num_node_features = 4 (x,y,z,t)
        elif(node_feature_mode == "centers_and_time"):
            t_centers = self.graph_dataframe["centers_list_all"]
            t_timesteps = self.graph_dataframe["timeframes_all"]
            t_node_features = torch.cat([t_centers,t_timesteps], dim = 1)

        return t_node_features

    def construct_graph_object(self,node_feature_mode="centers_and_time", edge_feature_mode="edge_type"):
        """
        Constructs the entire Graph object to serve as input to the MPN, and stores it in self.graph_obj,
        """
        # Determine graph connectivity (i.e. edges) and compute edge features
        edge_ixs_dict, edge_feats_dict = self._get_edge_ixs(edge_feature_mode)
        t_edge_ixs = edge_ixs_dict["edges"].to(self.device)
        # Transpose if Graph connectivity not in COO format with shape :obj:`[2, num_edges]
        if t_edge_ixs.shape[1] == 2:
            t_edge_ixs = t_edge_ixs.T

        # Prepare Inputs/ bring into apropiate shape to generate graph/object
        common_dtype = torch.float
        # Node Features
        t_node_features = self._load_node_features(node_feature_mode).type(common_dtype)
        
        # Edge Features
        t_edge_feats = edge_feats_dict[edge_feature_mode].type(common_dtype)
        
        # Duplicate Edges to make Graph undirected
        t_edge_feats = torch.cat((t_edge_feats, t_edge_feats), dim = 0).to(self.device)
        t_edge_ixs = torch.cat((t_edge_ixs, \
                        torch.stack((t_edge_ixs[1], t_edge_ixs[0]))),\
                        dim=1).to(self.device)

        t_temporal_edge_mask = torch.cat( [edge_ixs_dict["temporal_edges_mask"],
                                    edge_ixs_dict["temporal_edges_mask"]],
                                    dim= 0).to(self.device)

        # Add information to graph if it contains dummy Objects
        bool_contains_dummies = self._contains_dummy_objects()   

        # Add temporal-data for each node. Important for evaluation puposes
        t_frame_number = self.graph_dataframe["timeframes_all"]
        
        # Build Data-graph object for pytorch model
        self.graph_obj = Graph(x = t_node_features,
                               edge_attr = t_edge_feats,
                               edge_index = t_edge_ixs,
                               temporal_edges_mask = t_temporal_edge_mask,
                               timeframe_number = t_frame_number,
                               contains_dummies = bool_contains_dummies          
                               )
        # self.graph_obj.temporal_edges_mask = t_temporal_edge_mask
        # print('self.graph_obj.contains_dummies:\n',self.graph_obj.contains_dummies)

        # Ensure that graph is undirected.
        if self.graph_obj.is_directed():
            print('Before:\n{}'.format(self.graph_obj))
            undirectTransfomer = ToUndirected()
            self.graph_obj = undirectTransfomer(self.graph_obj)
            print('After:\n{}'.format(self.graph_obj))
        
        self.graph_obj.to(self.device)

class NuscenesMotGraphAnalyzer(NuscenesMotGraph):
    '''
    A subclass that serves to quickly analyze if the MotGraph is valid or not
    Checks if the number of objects is below the given KNN
    '''
    def __init__(self,
                    nuscenes_handle:NuScenes, start_frame:str , max_frame_dist:int = 3,
                    filterBoxes_categoryQuery:Union[str,List[str]] = None,
                    construction_possibility_checked = True,
                    adapt_knn_param = False,
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    dataset_params: dict = None,
                    detection_dict: Dict[str,Any] = None,
                    inference_mode: bool = False
                    ) -> None:
        # Inherit parent init
        super().__init__(
                    nuscenes_handle, start_frame , max_frame_dist,
                    filterBoxes_categoryQuery,
                    construction_possibility_checked,
                    adapt_knn_param,
                    device = device,
                    dataset_params= dataset_params,
                    detection_dict= detection_dict,
                    inference_mode= inference_mode)

    def _construct_graph_dataframe(self):
        graph_dataframe = {}
        self._load_detections(graph_dataframe)

        return graph_dataframe
