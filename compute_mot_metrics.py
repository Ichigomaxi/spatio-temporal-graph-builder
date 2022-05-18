from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox
import json

config_path = "configs/nuscenes_eval/mot_car_evaluation.json"
# config_path = 'configs/nuscenes_eval/tracking_nips_2019.json'

with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))

result_path_ = ""

eval_set_ = "mini_val"

output_dir_=  '/media/HDD2/students/maximilian/spatio-temporal-gnn/mot_metric/'+ eval_set_

version_ = 'v1.0-trainval'

dataroot_ = '/media/HDD2/Datasets/mini_nusc'

verbose_ = True

render_classes_ = None #List[str]  probably the class string writen in cfg_['class_range'] "class_range": {
                                                                                            # "car": 50,
                                                                                            # "truck": 50,
                                                                                            # "bus": 50,
                                                                                            # "trailer": 50,
                                                                                            # "pedestrian": 40,
                                                                                            # "motorcycle": 40,
                                                                                            # "bicycle": 40
                                                                                            # },

nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)

render_curves_:bool = True #computes  PR and TP

metrics_summary = nusc_eval.main(render_curves=render_curves_)

print(metrics_summary)