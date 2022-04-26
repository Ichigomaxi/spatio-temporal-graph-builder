
from re import T
from nuscenes.nuscenes import NuScenes
from torch_geometric.data import Dataset

from nuscenes.utils.splits import create_splits_scenes
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from typing import Optional, List, Dict, Set, Any, Iterable, Sequence

import torch

class NuscenesDataset(object):

    ALL_SPLITS = {"train", "val", "test", "train_detect", "train_track",
                  "mini_train", "mini_val"}

    def __init__(self, dataset_version: str='v1.0-trainval', dataroot: str='/media/HDD2/Datasets/mini_nusc'):
        
        
        print(f"Parsing NuScenes {dataset_version} ...")
        
        # Init data handler
        self.nuscenes_handle = NuScenes( \
                                version = dataset_version,\
                                dataroot = dataroot,\
                                verbose=True)
                                
        self.dataroot = dataroot
        self.version = dataset_version
        #Set of strings
        self.splits: Set[str] = set(s for s in self.ALL_SPLITS if s.split("_")[0] in dataset_version)
        self.sequences_by_name: Dict[str, Any] = {
            scene["name"]: scene for scene in self.nuscenes_handle.scene
        }
        self.splits_to_scene_names: Dict[str, List[str]] = create_splits_scenes()
        print("Done parsing")

    
        # # return ['data_1.pt', 'data_2.pt', ...]
        # #Get set of scenes
        # scenes = self.nuscenes_handle.scene
        # #Get first scenes
        # scene_0 = scenes[0]
        # # Get token of first frame
        # first_sample_token = scene_0['first_sample_token']

        # sample_0 = self.nuscenes_handle.get('sample', first_sample_token)

        # # Get LIDAR pointcloud
        # sensor = 'LIDAR_TOP'
        # lidar_top_data_0 = self.nuscenes_handle.get('sample_data', sample_0['data'][sensor])

        # pcl_path, _, _= self.nuscenes_handle.get_sample_data(lidar_top_data_0['token'], selected_anntokens=None, use_flat_vehicle_coordinates =False)

        # return [pcl_path]

    def get_nuscenes_mot_graph_list_debugging(self, 
                                only_first_scene:bool,
                                specific_device:str =None, 
                                label_type:str ='binary',
                                filterBoxes_categoryQuery:str = None
                                ):
        '''
        Return list/sequence of nuscenes_mot_graph objects.
        The processed data is taken from the mini_train split. Therefore it is mainly used for debugging and development

        '''
        # nusc = NuScenes(version='v1.0-mini', dataroot=r"C:\Users\maxil\Documents\projects\master_thesis\mini_nuscenes", verbose=True)
        nusc = self.nuscenes_handle
        split = create_splits_scenes()
        print(split.keys())
        split_scene_list = []
        for scene_name in split['mini_train']:
            for scene in nusc.scene:
                if scene['name']==scene_name:
                    split_scene_list.append(scene)

        sample_dict = {}

        i = 0 
        for scene in split_scene_list:
            last_sample_token =""
            sample_token = scene['first_sample_token']
            while(last_sample_token == ""):
                
                sample = nusc.get('sample', sample_token)
                sample_dict[i] = (scene['token'],sample["token"])
                i += 1
                sample_token = sample["next"]
                if(sample["token"]== scene['last_sample_token']):
                    last_sample_token = scene['last_sample_token']

        #Create List of Graph objects
        #______________________________________________________________#
        # Decide if only first scene should be computed
        only_first_scene = only_first_scene
        scene_token0, sample_token0= sample_dict[0]
        device = None
        if specific_device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                for i in range(torch.cuda.device_count()):
                    if torch.cuda.get_device_name(i) == "GeForce RTX 2080":
                        device = torch.device(i) # RTX 2080 8GB
        else:
            device = specific_device
        print("Device:", device)
        #_______________________________________________________________#

        MotGraphList= []
        for sample_key in sample_dict:
            scene_token_current, sample_token_current= sample_dict[sample_key]
            if(only_first_scene):
                if(scene_token0 == scene_token_current):
                    object = NuscenesMotGraph(nuscenes_handle = nusc,
                                start_frame=sample_token_current,
                                max_frame_dist = 3, 
                                filterBoxes_categoryQuery=filterBoxes_categoryQuery,
                                device= device)
                    is_possible2construct = object.is_possible2construct
                    if is_possible2construct:
                        object.construct_graph_object()
                        object.assign_edge_labels(label_type=label_type)
                        MotGraphList.append(object)
            else:
                object = NuscenesMotGraph(nuscenes_handle = nusc,
                                start_frame=sample_token_current,
                                max_frame_dist = 3,  
                                filterBoxes_categoryQuery=filterBoxes_categoryQuery ,device= device)
                is_possible2construct = object.is_possible2construct
                if is_possible2construct:
                    object.construct_graph_object()
                object.assign_edge_labels(label_type=label_type)
                MotGraphList.append(object)
        
        return MotGraphList
