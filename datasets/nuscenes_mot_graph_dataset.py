from ctypes import ArgumentError
from typing import List, Tuple
from datasets.nuscenes_mot_graph import NuscenesMotGraph, NuscenesMotGraphAnalyzer
from datasets.NuscenesDataset import NuscenesDataset
from nuscenes.nuscenes import NuScenes
import torch
import time
import pickle
from utils.inputs import load_detections_nuscenes_detection_submission_file

class NuscenesMOTGraphDataset(object):
    """
    Adopted from MOTGraphDataset from https://github.com/dvl-tum/mot_neural_solver
    Main Dataset Class. It is used to sample graphs from a a set of MOT sequences by instantiating MOTGraph objects.
    It is used both for sampling small graphs for training.
    Its main method is 'get_from_frame_and_seq', where given sequence name and a starting frame position, a graph is
    returned.
    """
    def __init__(self, dataset_params, mode, splits =None, logger = None,
                    nuscenes_handle: NuScenes = None,
                    device:str = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        assert mode in NuscenesDataset.ALL_SPLITS, "mode not part of official nuscenes splits: {}".format(NuscenesDataset.ALL_SPLITS)

        self.dataset_params = dataset_params
        self.mode = mode
        self.logger = logger
        self.device = device

        self.nuscenes_dataset = None
        if nuscenes_handle is not None:
            self.nuscenes_dataset = NuscenesDataset(self.dataset_params["dataset_version"],
                                                self.dataset_params["dataroot"],
                                                nuscenes_handle=nuscenes_handle)
        else:
            self.nuscenes_dataset = NuscenesDataset(self.dataset_params["dataset_version"],
                                                self.dataset_params["dataroot"])

        self.nuscenes_handle = self.nuscenes_dataset.nuscenes_handle

        self.seqs_to_retrieve:List[dict] = self._get_seqs_to_retrieve_from_splits(splits)
        self.seq_frame_ixs:List[Tuple[str,str]] = []
        self.frames_to_detection_boxes = None

        # Load Detections from dections submission style file
        if "use_gt_detections" in self.dataset_params \
            and self.dataset_params["use_gt_detections"] == False:
            self.frames_to_detection_boxes = load_detections_nuscenes_detection_submission_file(self.dataset_params["det_file_path"])


        # Sequence the dataset
        if self.seqs_to_retrieve:
            # Index the dataset (i.e. assign a pair (scene, starting frame) to each integer from 0 to len(dataset) -1)
            if dataset_params['load_valid_sequence_sample_list'] == True:
            
                print("##########################################################")
                print("Starting to load sequence sample list from pickle file")
                loading_list_start_time = time.time()
                self.seq_frame_ixs = []
                filepath = None 
                if ("train" in self.mode):
                    assert isinstance(dataset_params['sequence_sample_list_train_path'], str), \
                                        'No string-object was given for train set to \'sequence_sample_list_train_path\'! '
                    filepath_train = dataset_params['sequence_sample_list_train_path']
                    filepath = filepath_train
                elif "val" in self.mode:
                    assert isinstance(dataset_params['sequence_sample_list_val_path'], str), \
                                        'No string-object was given for val set to \'sequence_sample_list_val_path\'! '
                    filepath_val = dataset_params['sequence_sample_list_val_path']
                    filepath = filepath_val
                elif 'test' in self.mode:
                    assert isinstance(dataset_params['sequence_sample_list_test_path'], str), \
                                        'No string-object was given for test set to \'sequence_sample_list_test_path\'! '
                    filepath_test = dataset_params['sequence_sample_list_test_path']
                    filepath = filepath_test
                
                with open(filepath, 'rb') as f:
                    self.seq_frame_ixs = pickle.load(f)
                print("Finished Loading ")
                loading_list_end_time = time.time()
                time_difference = loading_list_end_time - loading_list_start_time
                print("Elapsed Loading time: {}".format(time_difference))
                print("##########################################################")
            else:
                self.seq_frame_ixs = self._index_dataset()

    def _get_seqs_to_retrieve_from_splits(self, splits:dict)-> List[dict]:
        """
        Returns list of sequences(nuscenes scene-objects) corresponding to the mode 
        and to the available sequences to the specific nuscenes handle
        Args:
        splits: dict that lists train, val, and/or test sequences. 
                This is useful if only specific scenes should be taken into scope
                If vanilla nuscenes splits should be used then leave it as None
        """
        seqs_to_retrieve = None
        scene_names = []
        if splits is not None:
            # custom split
            # Get appropiate sequence names depending on mode
            if 'train' in self.mode:
                scene_names = splits['train']
            elif 'val' in self.mode:
                scene_names = splits['val']
            elif 'test' in self.mode:
                scene_names = splits['test']
            print("Loading Custom selected scenes:", scene_names )
        else:
            # official nuscenes split
            dict_splits_to_scene_names = self.nuscenes_dataset.splits_to_scene_names
            split_name = self.mode
            scene_names = dict_splits_to_scene_names[split_name]
        
        # Dict containing all nuscene-scene-tables/dicts reachable by nuscenes_handle
        sequences_by_name = self.nuscenes_dataset.sequences_by_name
        # List respective nuscene-scene-tables/dicts corresponding to its split
        seqs_to_retrieve = [sequences_by_name[scene_name] for scene_name in scene_names]

        return seqs_to_retrieve

    def _index_dataset(self):
        """
        For each sequence in our dataset we consider all valid frame positions (see 'get_last_frame_df()').
        Then, we build a tuple with all pairs (scene, start_frame). The ith element of our dataset corresponds to the
        ith pair in this tuple.
        Returns:
            tuple of tuples of valid (seq_name, frame_num) pairs from which a graph can be created
        """
        print('############################################################')
        print('Starting to Index the dataset\n {}-Datasplit'.format(self.mode))
        split_scene_list = self.seqs_to_retrieve

        # Create List of all sample frames within the retrieved scenes/sequences 
        sample_list_all = []
        for scene in split_scene_list:
            last_sample_token =""
            sample_token = scene['first_sample_token']
            while(last_sample_token == ""):
                sample = self.nuscenes_handle.get('sample', sample_token)
                sample_list_all.append((scene['token'], sample["token"]))
                sample_token = sample["next"]
                if(sample["token"]== scene['last_sample_token']):
                    last_sample_token = scene['last_sample_token']

        # Filter out all non valid (scene_tokens, sample_tokens) that will not be able to build a valid graph with 3 or more frames
        print("First Filtering Process:\n Check if there are still enough time frames left")
        first_filter_start_time = time.time()
        filtered_sample_list = []
        for scene_sample_tuple in sample_list_all:
            # Check sample if enough frames remaining to construct graph
            scene_token, init_sample_token = scene_sample_tuple
            # Get Scenes-object
            scene = self.nuscenes_handle.get("scene",scene_token)
            # Count how many sample frames are remaining until last frame 
            last_sample_token = ""
            sample_token = init_sample_token
            i= 0
            while(last_sample_token == ""):
                sample = self.nuscenes_handle.get('sample', sample_token)
                sample_token = sample["next"]
                i += 1
                if(sample["token"]== scene['last_sample_token']):
                    last_sample_token = scene['last_sample_token']
            # If less than dataset_params['max_frame_dist'] frames counted then we filter it out
            if not (i < self.dataset_params['max_frame_dist']):
                filtered_sample_list.append(scene_sample_tuple)

        print("Finished Filtering:\n Now remaining MOT graph samples should contain {} time frames".format(self.dataset_params['max_frame_dist']))
        first_filter_end_time = time.time()
        print("Elapsed Time for first filtering",first_filter_end_time - first_filter_start_time, "seconds")
        print('---------------------------------------------')

        #TODO
        # Filter if num_objects less than KNN -param or 1 
        print("Filtering Process:\n Check if any Mot Graph are not buildable due to lack of detections")
        start = time.time()
        # take on older configs which do not explicitly contain this new param
        if ("filter_for_buildable_sample_frames" not in self.dataset_params)\
            or ("filter_for_buildable_sample_frames" in self.dataset_params
            and self.dataset_params["filter_for_buildable_sample_frames"]):

            construction_possibility_checked = True # due to previous filtering
            filtered_sample_list_new = []
            for scene_sample_tuple in filtered_sample_list:
                scene_token, init_sample_token = scene_sample_tuple
                start_frame = init_sample_token
                mot_graph_analyzer = NuscenesMotGraphAnalyzer(
                                            nuscenes_handle = self.nuscenes_handle,
                                            start_frame = start_frame,
                                            max_frame_dist = self.dataset_params['max_frame_dist'],
                                            construction_possibility_checked = construction_possibility_checked,
                                            filterBoxes_categoryQuery= self.dataset_params["filterBoxes_categoryQuery"],
                                            adapt_knn_param = self.dataset_params["adapt_knn_param"],
                                            device= self.device,
                                            dataset_params=self.dataset_params,
                                            detection_dict= self.frames_to_detection_boxes)
        
                if not mot_graph_analyzer.contains_dummy_objects():
                    filtered_sample_list_new.append(scene_sample_tuple)
            filtered_sample_list = filtered_sample_list_new
        print("Finished Filtering:\n Now remaining mot graph samples should not contain any dummy boxes")
        end = time.time()
        print("Elapsed Time for filtering",end - start, "seconds")
        print('---------------------------------------------')
        

        return filtered_sample_list

    def get_filtered_samples_from_one_scene(self,searched_scene_token:str) -> List[Tuple[str,str]]:
        """
        Returns a List of all scene_token-sample_token tuples that correspond to a certain scene
        Args:
        scene_token

        """
        list_scene_sample_tuple:List[Tuple[str,str]] = [] 

        for tuple in self.seq_frame_ixs:
            scene_token_current = tuple[0]
            # sample_token_current = tuple[1]
            if(scene_token_current == searched_scene_token):
                list_scene_sample_tuple.append(tuple)

        return list_scene_sample_tuple
        

    def get_nuscenes_handle(self):
        """
        Returns nuscens handle
        """
        return self.nuscenes_handle

    def get_from_frame_and_seq(self, seq_name:str, start_frame:str,
                               return_full_object:bool = False,
                                inference_mode:bool =False):
        """
        Method behind __getitem__ method. We load a graph object of the given sequence name, starting at 'start_frame'.

        Args:
            seq_name: scene_token from nuscenes.Nuscenes
            start_frame: sample_token from nuscenes.Nuscenes
            return_full_object: bool indicating whether we need the whole MOTGraph object or only its Graph object
                                (Graph Network's input)

        Returns:
            mot_graph: output MOTGraph object or Graph object, depending on whethter return full_object == True or not

        """
        mot_graph = NuscenesMotGraph(
                                    nuscenes_handle = self.nuscenes_handle,
                                    start_frame = start_frame,
                                    max_frame_dist = self.dataset_params['max_frame_dist'],
                                    filterBoxes_categoryQuery = self.dataset_params["filterBoxes_categoryQuery"],
                                    adapt_knn_param = self.dataset_params["adapt_knn_param"],
                                    device = self.device,
                                    dataset_params = self.dataset_params, 
                                    detection_dict = self.frames_to_detection_boxes,
                                    inference_mode = inference_mode)

        # Construct the Graph Network's input
        mot_graph.construct_graph_object(
                        node_feature_mode = self.dataset_params['node_feature_mode'],
                        edge_feature_mode = self.dataset_params['edge_feature_mode'])
        
        if self.mode in ('train', 'val', "train_detect", "train_track",
                  "mini_train", "mini_val") \
            and self.dataset_params["use_gt_detections"]:
            mot_graph.assign_edge_labels(self.dataset_params["label_type"])

        if return_full_object:
            return mot_graph

        else:
            return mot_graph.graph_obj

    def __len__(self):
        return len(self.seq_frame_ixs) if hasattr(self, 'seqs_to_retrieve') and self.seqs_to_retrieve else 0

    def __getitem__(self, ix):
        seq_name, start_frame = self.seq_frame_ixs[ix]

        return self.get_from_frame_and_seq(seq_name= seq_name,
                                           start_frame = start_frame,
                                           return_full_object=False,
                                           inference_mode=False
                                           )
