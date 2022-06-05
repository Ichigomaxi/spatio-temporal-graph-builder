
import argparse
import os.path as osp
from typing import List
import json

# Load the summary/ submission 
if __name__ == "__main__":
    
    # change this #####################################################################
    # dirpath ="/media/HDD2/Datasets/nuscenes_baseline_detections"
    # detections_path = "megvii/megvii_val.json" 
    # detections_path = "megvii_test.json"
    # detections_path = "megvii_train.json"     

    # dirpath = "/media/HDD2/Datasets/nuscenes_CBMOT_detections/resources"
    # detections_path = "centertrack_origin.json"
    # detections_path = "centertrack_tracks.json"
    # # CenterPoint Detection centerpoint_voxel_1440 - after bugfix 
    # detections_path = "infos_val_10sweeps_withvelo_filter_True.json"

    # CenterPoint Detection centerpoint_voxel_1440_dcn(flip) - (depreacted, before bugfix)
    dirpath = "/media/HDD2/Datasets/nuscenes_EagerMOT_detections"
    detections_path = "centerpoint_3Ddetections/val/infos_val_10sweeps_withvelo_filter_True.json"
    detections_path = "centerpoint_3Ddetections/val/detections.json"

    ## Leave this ######################################################################
    detections_file_path = osp.join(dirpath,detections_path)
    detections_dict:dict = None
    with open(detections_file_path, 'r') as _f:
        detections_dict = json.load(_f)

    print("Keys:", detections_dict.keys())

    for i, sample_token in enumerate(detections_dict["results"]):
        if i < 1:
            sample_results = detections_dict["results"][sample_token]
            print("sample_token:", sample_token)
            print("sample_results: \n",sample_results)
