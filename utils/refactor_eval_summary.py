
import argparse
import os.path as osp
from typing import List
import json

# Load the summary/ submission 
if __name__ == "__main__":
    
    # change this
    dirpath ="/media/HDD2/students/maximilian/spatio-temporal-gnn/experiments"
    file_subdir = "05-30__18-40_evaluation/eval_results"        

    metrics_summary_path = 'metrics_summary.json'
    metric_details_path = "metrics_details.json"

    ## Leave this
    metrics_summary_file_path = osp.join(dirpath,file_subdir,metrics_summary_path)
    metric_details_file_path = osp.join(dirpath,file_subdir,metric_details_path)
    metrics_summary = None
    with open(metrics_summary_file_path, 'r') as _f:
        metrics_summary = json.load(_f)
    metric_details = None
    with open(metric_details_file_path, 'r') as _f:
        metric_details = json.load(_f)

    car_summary = {}
    label_metrics = metrics_summary["label_metrics"]
    for metric in metrics_summary["label_metrics"]:
        metric_dict = label_metrics[metric]
        car_summary[metric] = metric_dict['car']
    print(car_summary)

    summary_results_file = osp.join(dirpath, file_subdir,"car_metric_summary.json")
    with open(summary_results_file, 'w') as f:
        json.dump(car_summary, f, indent=4)

    car_metric_details = metric_details["car"]
    print(metrics_summary)

    details_results_file = osp.join(dirpath, file_subdir,"car_metric_details.json")
    with open(details_results_file, 'w') as f:
        json.dump(car_metric_details, f, indent=4)