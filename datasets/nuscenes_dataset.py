
from re import T
from nuscenes.nuscenes import NuScenes
from torch_geometric.data import Dataset

from nuscenes.utils.splits import create_splits_scenes

from typing import Optional, List, Dict, Set, Any, Iterable, Sequence

import torch as t

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
            scene["name"]: scene for scene in self.nusc.scene
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

        
