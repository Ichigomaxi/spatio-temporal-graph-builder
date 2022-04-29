
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from datasets.NuscenesDataset import NuscenesDataset

class NuscenesMOTGraphDataset(object):
    """
    Adopted from MOTGraphDataset from https://github.com/dvl-tum/mot_neural_solver
    Main Dataset Class. It is used to sample graphs from a a set of MOT sequences by instantiating MOTGraph objects.
    It is used both for sampling small graphs for training, as well as for loading entire sequence's graphs
    for testing.
    Its main method is 'get_from_frame_and_seq', where given sequence name and a starting frame position, a graph is
    returned.
    """
    def __init__(self, dataset_params, mode, splits =None, logger = None):
        
        assert mode in NuscenesDataset.ALL_SPLITS

        self.dataset_params = dataset_params
        self.mode = mode
        self.logger = logger

        self.nuscenes_dataset = NuscenesDataset(self.dataset_params["dataset_version"],
                                                self.dataset_params["dataroot"])
        
        self.nuscenes_handle = self.nuscenes_dataset.nuscenes_handle

        self.seqs_to_retrieve = self._get_seqs_to_retrieve_from_splits(splits)

        if self.seqs_to_retrieve:
            # Index the dataset (i.e. assign a pair (scene, starting frame) to each integer from 0 to len(dataset) -1)
            self.seq_frame_ixs = self._index_dataset()

    def _get_seqs_to_retrieve_from_splits(self, splits):
        """
        Returns list of sequences(nuscenes scene-objects) corresponding to the mode
        """
        seqs_to_retrieve = None
        if splits is None:
            

            dict_splits_to_scene_names = self.nuscenes_dataset.splits_to_scene_names
            split_name = self.mode
            scene_names = dict_splits_to_scene_names[split_name]
            sequences_by_name = self.nuscenes_dataset.sequences_by_name

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
        
        return filtered_sample_list

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
                                    filterBoxes_categoryQuery= self.dataset_params["filterBoxes_categoryQuery"]
                                    )

        # Construct the Graph Network's input
        mot_graph.construct_graph_object()
        if self.mode in ('train', 'val', "train_detect", "train_track",
                  "mini_train", "mini_val"):
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
