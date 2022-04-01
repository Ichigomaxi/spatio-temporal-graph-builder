import os.path as osp

import torch
from nuscenes.nuscenes import NuScenes
from torch_geometric.data import Dataset, download_url


class NuscenesDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.nuscenes_handle = NuScenes( \
                                version='v1.0-trainval',\
                                dataroot='/media/HDD2/Datasets/mini_nusc',\
                                verbose=True)
    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        # return ['data_1.pt', 'data_2.pt', ...]
        #Get set of scenes
        scenes = self.nuscenes_handle.scene
        #Get first scenes
        scene_0 = scenes[0]
        # Get token of first frame
        first_sample_token = scene_0['first_sample_token']

        sample_0 = self.nuscenes_handle.get('sample', first_sample_token)

        # Get LIDAR pointcloud
        sensor = 'LIDAR_TOP'
        lidar_top_data_0 = self.nuscenes_handle.get('sample_data', sample_0['data'][sensor])

        pcl_path, _, _= self.nuscenes_handle.get_sample_data(lidar_top_data_0['token'], selected_anntokens=None, use_flat_vehicle_coordinates =False)

        return [pcl_path]

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data