
import argparse
import os.path as osp
from typing import List
import json

# Load the summary/ submission 
if __name__ == "__main__":
    
    # change this
    dirpath ="/media/HDD2/Datasets/nuscenes_baseline_detections"
    detections_path = "megvii/megvii_val.json" 
    # detections_path = "megvii_test.json"
    # detections_path = "megvii_train.json"     

    ## Leave this
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
