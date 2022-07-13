# spatio-temporal graph builer

## Architecture 

<!-- ![hustlin_erd](./spatio-temporal-graph-builer/documentation/pipeline_v5.pdf)
<embed src="/documentation/pipeline_v5.pdf" type="application/pdf"> -->
![pipeline_v5](./documentation/pipeline_v5-1.png?raw=true "Visualization of our Pipeline.")
<!-- ![hustlin_erd](./spatio-temporal-graph-builer/documentation/pipeline_v5-1.png) -->

We build on top of the architecture established by [Braso et al. - Learning a Neural Solver for Multiple Object Tracking](https://github.com/dvl-tum/mot_neural_solver).

Therefore, we use [SACRED](https://github.com/IDSIA/sacred/tree/master) to manage the executions (read config, terminal commands) and [pytorch lightning](https://www.pytorchlightning.ai/) to manage the boiler-plate code for the training and the inference.

Our pipeline is composed of 4 parts.
1. Spatio-Temporal Graph Builder
2. Spatio-Temporal Graph Neural Network
3. Greedy Rounding (Greedy projection)
4. Global Tracking

### 1. Spatio-Temporal Graph Builder ###
Get familiar with the **nuscenes_mot_graph** class and its attribute the **graph object** _(nuscenes_mot_graph.graph_obj)_.

A **nuscenes_mot_graph** contains most information connected to the spatio-temporal graph.

A graph object (often denomenated graph_obj) is a class that inherits from the [data class](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) from [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html).
It contains all bare bone information of the graph in a defined way, such that the node and edge features can be processed in a unified way by the pytorch model.
The most important attributes of the graph are:
- node features: graph_obj.x
- edge features: graph_obj.edge_attr
- edge indices: graph_obj.edge_index

The remainig attributes can be seen in `def construct_graph_object(self,node_feature_mode="centers_and_time", edge_feature_mode="edge_type"):` - [nuscenes_mot_graph](./datasets/nuscenes_mot_graph.py) 
```python
# Build Data-graph object for pytorch model
        self.graph_obj = Graph(x = t_node_features,
                               edge_attr = t_edge_feats,
                               edge_index = t_edge_ixs,
                               temporal_edges_mask = t_temporal_edge_mask,
                               timeframe_number = t_frame_number,
                               contains_dummies = bool_contains_dummies          
                               )
```

Important files associated with this:
- [nuscenes_mot_graph](./datasets/nuscenes_mot_graph.py)
- [nuscenes_mot_graph_dataset](nuscenes_mot_graph_dataset.py)
- [helper functions graph ](./graph/graph_generation.py)
- [helper functions groundtruth_generation](./groundtruth_generation/nuscenes_create_gt.py)

### 2. Spatio-Temporal Graph Neural Network ### 
Important files associated with this:
- [linear layers](./model/mlp.py)
- [graph neural network](./model/mpn.py)
- [pytorch lightning module (wraps around torch model and handels training and tracking)](./pl_module/pl_module.py)
- [graph neural network](./model/mpn.py)

### 3. Greedy Rounding (Greedy projection) ### 
- - - -
_**Attention!!!:**_

In parallel to the undirected graph, the undirected graphs edges, edge predictions are transformed to a directed graph.
This is Important to use the Rounding method from Braso et al.
The edges, edge predictions and other attribute begin with `temporal_directed_...` before their name.:
- `temporal_directed_edge_preds`
- `temporal_directed_edge_indices`
This transformation is done in function `to_directed_temporal_graph` of [graph helper functions](./utils/graph.py).
- - - -
Important files associated with this:
- [helper functions for 'local' tracking within graph](./utils/evaluation.py) look at function `project_graph_model_output`
- [Greedy Rounding module from \[Braso et al.\] ](./tracker/projectors.py)
- [graph helper functions](./utils/graph.py)

### 4. Global Tracking ###
We perform window shifting to track a whole scene.
An overview over the algorithm is explained in the [thesis](./documentation/220614_RAIStudentThesis_v12_final.pdf).

Important files associated with this:
- [Tracking module (with window shifting)](./tracker/mpn_tracker.py)

## Installation with conda or pip
We provide the requirements for a pip or for a conda installation.
All requirement files are in the requirements-folder of this repo.

```bash
# using pip
pip install -r ./requirements/pip_requirements.txt
```

```bash
# using Conda
conda create --name <env_name> --file ./requirements/requirements.txt
```
Best Practice:
Create a Conda environment
Then continue with the installation of the required packages.

## Installation from source
1)
Please install the devkit for the nuscenes dataset.
Currently using version nuscenes-devkit 1.1.9

Follow the installation guide on https://github.com/nutonomy/nuscenes-devkit

Usually it is just: 
pip install nuscenes-devkit

2)
Install open3D version 0.14.1+

3)
Install plotly/dash
```bash
pip install dash
pip install jupyter-dash
pip install pandas
```


4) install pytorch and pytorch-geometric
conda list output:
```bash
pyg                       2.0.4           py39_torch_1.10.0_cu113    pyg
pytorch                   1.10.0          py3.9_cuda11.3_cudnn8.2.0_0    pytorch
pytorch-cluster           1.6.0           py39_torch_1.10.0_cu113    pyg
pytorch-mutex             1.0                        cuda    pytorch
pytorch-scatter           2.0.9           py39_torch_1.10.0_cu113    pyg
pytorch-sparse            0.6.13          py39_torch_1.10.0_cu113    pyg
pytorch-spline-conv       1.2.1           py39_torch_1.10.0_cu113    pyg
torch-tb-profiler         0.4.0                    pypi_0    pypi
torchmetrics              0.8.0              pyhd8ed1ab_0    conda-forge
pip list output:
torch                   1.10.0
torch-cluster           1.6.0
torch-geometric         2.0.4
torch-scatter           2.0.9
torch-sparse            0.6.13
torch-spline-conv       1.2.1
torch-tb-profiler       0.4.0
torchmetrics            0.8.0
```

5) install pytorch lightning:
conda list output:
```bash
pytorch-lightning         1.5.10             pyhd8ed1ab_0    conda-forge
```
pip list output:
```bash
pytorch-lightning       1.5.10
```

6) install sacred (at this moment must be installed manually. Download from git and do "python setup.py")

conda list output:
```bash
sacred                    0.8.3                    pypi_0    pypi
```

pip list output:
```bash
sacred                  0.8.3
```

7) install [motmetrics](https://github.com/cheind/py-motmetrics)  <= 1.1.3
<!-- https://github.com/cheind/py-motmetrics -->
<!-- https://pypi.org/project/motmetrics/ -->

```bash
pip install motmetrics==1.1.3
```

```bash
motmetrics                1.1.3                    pypi_0    pypi
```

8) install ujson 5.3.0 
pip install ujson

## Dataset and Object Detections
We perform tracking on the [nuscenes](https://www.nuscenes.org/) dataset.
For training and inference we use the provided ground truth annotations (3D bounding boxes) as object detections.

For our final experiments we compare our method to other state of the art methods, namely [EagerMOT](https://github.com/aleksandrkim61/EagerMOT) and [CBMOT](https://github.com/cogsys-tuebingen/CBMOT) on publicly available object detections given by [CenterPoint](https://github.com/tianweiy/CenterPoint).

To perform the experiments
Download Center Point Detections from [here](https://github.com/tianweiy/CenterPoint/tree/master/configs/nusc)

[EagerMOT](https://github.com/aleksandrkim61/EagerMOT) uses the **deprecated** **centerpoint_voxel_1440_dcn(flip)** detections. Download from [here](https://drive.google.com/drive/folders/1fAz0Hn8hLdmwYZh_JuMQj69O7uEHAjOh)

CBMOT uses the **centerpoint_voxel_1440** detections. Download from [here](https://drive.google.com/drive/folders/1FOfCe9nWQrySUx42PlZyaKWAK2Or0sZQ).



## Checkpoints
We provide the Checkpoints under [here](https://syncandshare.lrz.de/getlink/fi7wXvdpbz2rvhrubn8urQbq/).

The checkpoint we use can be downloaded [here](https://syncandshare.lrz.de/dl/fi7wXvdpbz2rvhrubn8urQbq/results/experiments/05-08_19:58_train_w_default_config/model_checkpoints/epoch=48-step=263080.ckpt).

It is under: 
- Home/master_thesis_M_Listl/results/experiments/05-08_19:58_train_w_default_config/model_checkpoints

## Training
- - - -
_**Attention!!!:**_
In SACRED we define a base config in the script, as seen here:
```python
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

ex = Experiment()

ex.add_config('configs/tracking_cfg.yaml') # Set Base config 
```

In order to avoid changing the script everytime we want to use another config, we can overload the config using terminal commands.
However, this means that if you add new keywords to a config file. **It should be included in the base config!**

We add the word `with` to the terminal command and the path to the new config-file that we want to use.  
```bash
python example_script.py with new_config.yaml
```

- - - -

Base command for training with base config `config/tracking_cfg.yaml`
```bash
python train.py
```

Training with overloaded config `configs/tracking_example_config.yaml`
```bash
python train.py with configs/tracking_example_config.yaml
```
### Config Params ###

```yaml
# Config for modes from the nuscenes split
train_dataset_mode: "train"
eval_dataset_mode: "val"
```
- `train_dataset_mode`: decides the data split for the training split e.g {"train", "val", "test", "train_detect", "train_track","mini_train", "mini_val"}
- `eval_dataset_mode`: same as `train_dataset_mode` but for the validation split


```yaml
gpu_settings:
  device_type: "gpu"
  # Specify which GPUs to use
  device_id: [1] # Specifically takes the gpu associated with cuda:1 
  torch_device: 'cuda:1'
```
`device_type`: determines processing unit to perform pytorch/tensor computations, e.g {"cpu", "gpu"}
`device_id`: is obsolete
`torch_device`: Very important! Determines on which torch.device the pytorch computations are performed e.g {"cpu", "cuda","cuda:0",...}

```yaml
train_params:
  batch_size: 4
  num_epochs: 200
  optimizer:
    type: Adam
    args:
      lr: 0.0001
      weight_decay: 0.001

  lr_scheduler:
    type:
    args:
      step_size: 7
      gamma: 0.5
```
- `batch_size`: Determines mini-batch size. Number of Graphs used per iteration step in optimization
- `num_epochs`: Determines maximum number of epochs
- `optimizer`: Contains Parameters for optimizer
- `lr_scheduler`: Attention!! Contains Parameters for LR lr_scheduler. However it does not work until now.


```yaml
train_params:
  num_workers: 0 # Used for dataloaders
  save_every_epoch: True # Determines if every a checkpoint will be saved for every epoch
  save_epoch_start: 1 # If the arg above is set to True, determines the first epoch after which we start saving ckpts
  tensorboard: True
  num_save_top_k: 2
  include_custom_checkpointing : True
  include_early_stopping : True
  loss_params:
    weighted_loss: True
```
- `num_workers`: Parameters that derives from a pytorchlightning functionality. It does not work until now. So it is always set to 0!
- `tensorboard`: Determines if the logger module is initiated to monitor the training over tensorboard
- `num_save_top_k`: Determines how many of the best performing training checkpoints should be saved. 
- `include_custom_checkpointing`: Determines if training should include a custom function to save every n-th epoch. (Custom callback)
- `save_every_epoch`: Only works if `include_custom_checkpointing`==True!
- `save_epoch_start`: Only works if `include_custom_checkpointing`==True!
- `include_early_stopping`: Determines if training should be stopped preemptively with early-stopping callback from pytorchlightning
- `weighted_loss`: Determines if Loss is weighted to inprove label imbalance (more negatives than positives)


```yaml
dataset_params:
  # Preprocessing ParamsL
  load_valid_sequence_sample_list : True
  # If the above is true then specify the path to the pickle files
  # Must be absolutepath until now
  # if mode not train or val then sequence_sample_list_train_path will be loaded into Dataset object
  sequence_sample_list_train_path : '/media/HDD2/students/maximilian/spatio-temporal-gnn/dataset/preprocess_dataset_nuscenes/sequence_sample_list_train.pkl' 
  sequence_sample_list_val_path : '/media/HDD2/students/maximilian/spatio-temporal-gnn/dataset/preprocess_dataset_nuscenes/sequence_sample_list_val.pkl'

  # Dataset Processing params:
  dataset_version: 'v1.0-trainval'
  dataroot : '/media/HDD2/Datasets/nuscenes2'
  is_windows_path : False

  # Filter Parameter
  # Contains a list of nuscenes class labels.
  # Only detections from these clases will be loaded for the graph
  # filterBoxes_categoryQuery: ['vehicle.car']  
  filterBoxes_categoryQuery: ['vehicle.car'] # ['vehicle.car', 'vehicle.bicycle','vehicle.bus', 'vehicle.motorcycle','human.pedestrian', 'vehicle.trailer', 'vehicle.truck']

  # Graph Construction Parameters
  graph_construction_params:
    spatial_knn_num_neighbors: 4
    temporal_knn_num_neighbors : 4
    spatial_shift_timeframes : 20
    max_temporal_edge_length : 2
    
  max_frame_dist: 3 # Maximum number of frames contained in each graph sampled graph
  # Node Features
  node_feature_mode : "centers_and_time" # determines the included node features 
  # Edge Features
  edge_feature_mode : "edge_type" # determines the included edge features 
  # Edge_Label Type
  label_type: "binary" # "binary" or "multiclass"
  # Choose how Graph construction should be handled
  adapt_knn_param: False # if number of objects is below the KNN-Param then K

  # Data Augmentation Params
  augment: False # Determines whether data augmentation is performed
```
- `load_valid_sequence_sample_list`: Determines if a list of valid sample_tokens can be loaded from a given path. This is allows to bypass the filtering at the beginning. 
- `sequence_sample_list_train_path`: Path to pickle file with list of valid sample_tokens for training split. Is only used if `load_valid_sequence_sample_list` is true!
- `sequence_sample_list_val_path`: Path to pickle file with list of valid sample_tokens for validation split. Is only used if `load_valid_sequence_sample_list` is true!
- `dataset_version`: string with version of nuscenes. Important for nuscenes handle `nusc`.
- `dataroot`: Path to nuscenes dataset Important for nuscenes handle `nusc`.
- `is_windows_path`: obsolete parameter
- `filterBoxes_categoryQuery`: List of strings. Determines the classes of the objects that will be used for training and tracking. All other objects will be filtered out.
- `spatial_knn_num_neighbors`: Determines number of spatial neighbors per node.
- `temporal_knn_num_neighbors`: Determines number of temporal neighbors per node per l-th iteration (1,2,..., max_temporal_edge_length (\beta)).
- `spatial_shift_timeframes`: Determines the length by which the different timeframes are shifted.
- `max_temporal_edge_length`: Determines number of timeframes that the temporal edges can skip. (skip connections)
- `max_frame_dist`: Determines number of timeframes per graph
- `node_feature_mode`: Determines the node features used for the message passing step
- `edge_feature_mode`: Determines the edge features used for the message passing step
- `label_type`: This is a deprecated config. It should remain "binary".
- `adapt_knn_param`: If True it allows to process graphs with a number of objects less than k_{temp} or k_{spatial}.
- `augment`: obsolete


## Tracking Inference/Validation
After training our model we want to perform tracking and validate our results.
For testing the tracking pipeline within a single graph we have use the `evaluate_single_graphs.py` [script](./evaluate_single_graphs.py). (Architecture part 1 - 3, no global tracking) 
For testing the overall tracking performance we the `evaluate.py` [script](./evaluate.py). The global tracking yields a yaml-file which contains the tracking results in the official nuscenes format. In addition, the evaluation of the results is done by the [official nuscenes evaluation script](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/tracking). This computes the MOT-metrics such as AMOTA, AMOTP

### Local Tracking/ Tracking within a single spatio-temporal Graph ###
We have developed a script which allows to test our tracking method on a graph-by-graph basis. (Architecture part 1 - 3, no global tracking) 
```bash
python evaluate_single_graphs.py
```

#### Config Params ####
```yaml
eval_params:
  debbuging_mode : True
  visualize_graph : False
  save_graphs : True
  save_single_graph_submission: False
  use_gt: True
  tracking_threshold: 0.5
```
`use_gt`: Use Ground truth edge labels instead of edge predictions for tracking
`debbuging_mode`: If true then only processes the first 5 graphs.
`visualize_graph`: If true then it visualizes the Graph in 3D.
`save_graphs`: If true then the inferred mot_graphs are saved in a pickle file. This is used for visualizing these graphs on a local machine.
`save_single_graph_submission`: obsolete param.
`tracking_threshold` : Determines the threshold used in the Greedy Rounding method. Typically  0.5 because it is binary classification.

### Global Tracking ###
(Architecture part 1 - 4, with global tracking) 

Base command for eval with base config `configs/evaluate_mini_tracking_cfg.yaml`
```bash
python evaluate.py
```

Evaluate with overloaded config `configs/evaluate_example_cfg.yaml`
```bash
python evaluate.py with configs/evaluate_example_cfg.yaml
```

#### Config Params ####

```yaml
test_dataset_mode: "mini_val"
```
test_dataset_mode: Determines the datasplit that is used for the tracking and the final evaluation. This is similar to `train_dataset_mode`.

```yaml
dataset_params:
  # Preprocessing Params
  filter_for_buildable_sample_frames : True
  # Detection Params:
  use_gt_detections: True # If True loaded detections will equal the sample_annotations from the trainval-set 
  det_file_path: "/media/HDD2/Datasets/nuscenes_EagerMOT_detections/centerpoint_3Ddetections/val/infos_val_10sweeps_withvelo_filter_True.json"
```
- `filter_for_buildable_sample_frames`: Determines if the sample frames should be filtered for ones which do not allow the building of valid spatio-temporal graphs.
- `use_gt_detections`: If True then groundtruth annotations (3D bounding boxes) are used as 3D object detections. If `use_gt_detections`==False then detections from `det_file_path` are loaded.
- `det_file_path`: Path to file with 3D object detections in official nuscenes submission style.

```yaml
eval_params:
  save_submission: True
```
- `save_submission`: If true then the tracked object detections (with their Trackl-IDs) are saved in a .json-file. The summary/submission is in the official nuscenes submission style. It allows evaluation afterwards with the nuscenes eval-script or direct submission to nuscenes server.

## Visualization
In our repo we use 3D Visualizations to check the functionality of our pipeline. (based on open3D)
However, most of the time the whole nuscenes dataset will only be available at a remote workstation. In addition, computing visualization on the workstation and sending them over an ssh connection is challenging.

Therefore, most visualization script are run locally on a laptop which only visualizes a small subset of the entire nuscenes dataset (mini-nuscenes).
Some scripts run the complete pipeline on the laptop with a pretrained GNN. Other scripts rely on mot_graph-objects that are saved in a pickle-file (.pkl).

Nevertheless, it is possible to compute 2D visualization from the nuscenes-devkit on a workstation and visualize them on a local computer. It works over jupyter-notebooks. As seen [here.](./visualize_nuscenes_scenes_for_dev.ipynb)

### 3D Visualization of single graphs from workstation on laptop ###
In [evaluate_single_graphs.py](./spatio-temporal-graph-builer/pl_module/pl_module.py) we can save the mot_graphs which contain the edge predictions in a pickle file.
If we set the config ['eval_params']['save_graphs']=True.
```python
# save objects for visualization
        if(self.hparams['eval_params']['save_graphs']):
            os.makedirs(output_files_dir, exist_ok=True) # Make sure dir exists
            # Save in pickle format
            pickle_file_path = osp.join(output_files_dir,"inferred_mot_graphs.pkl")
            with open(pickle_file_path, 'wb') as f:
                torch.save(inferred_mot_graphs,f, pickle_protocol = 4)
```
Afterwards the pickle-file can be moved to a local computer with a screen, to visualize the results.
For visualization we use the following script: [visualize_eval_graphs.py](./spatio-temporal-graph-builer/visualize_eval_graphs.py)
```bash
python visualize_eval_graphs.py
```

### 3D Visualization of graphs on laptop (Mini-Nuscenes) ###
The following script visualizes the input graph (spatio-temporal graph) and the ideal output graph (visualization of edge labels).
[visualize_spatio_temporal_graph.py](./spatio-temporal-graph-builer/visualize_spatio_temporal_graph.py)

### 3D Visualization of single graphs on laptop (Mini-Nuscenes, Thesis style) ###
We can also run [evaluate_single_graphs.py](./spatio-temporal-graph-builer/pl_module/pl_module.py) locally on computer with a screen, to visualize our matchings.
Our pretrained GNN can be loaded into CPU memory. However, our local computer could only load the mini-nuscenes set.
To activate this function set ['eval_params']['visualize_graph']= True in the config file.
```python
if(self.hparams['eval_params']['visualize_graph']):
                geometry_list = build_geometries_input_graph_w_pointcloud(mot_graph, dataset.get_nuscenes_handle())
                # geometry_list = visualize_eval_graph_new(mot_graph)
                # geometry_list = visualize_eval_graph(mot_graph)
                visualize_geometry_list(geometry_list)
                geometry_list = build_geometries_input_graph_w_pointcloud_w_Bboxes(mot_graph, dataset.get_nuscenes_handle())
                visualize_geometry_list(geometry_list)
                if self.hparams['dataset_params']['use_gt_detections']:
                    geometry_list = visualize_output_graph_new(mot_graph)
                    # geometry_list = visualize_output_graph(mot_graph)
                    visualize_geometry_list(geometry_list)
```
This results in the following picture, which can be seen in the Thesis as well.

![Input Graph](./documentation/input_scene_0103_sample_0_3e8750f331d7499e9b5123e9eb70f2e2_w_BBoxes_pose_1.png?raw=true "Visualization of Comparison.")

![Inferred Graph](./documentation/scene_0103_sample_0_3e8750f331d7499e9b5123e9eb70f2e2_pose_1.png?raw=true "Visualization of inferred graph.")

### 3D Visualization of Comparison between tracking results ours vs other methods (Mini-Nuscenes, CenterPoint detections) ###
The following script visualizes both tracked objects from our and from another method's results-file.
[visualize_submissions_comparison_centerpoint.py](./spatio-temporal-graph-builer/visualize_submissions_comparison_centerpoint.py)

This results in the following image:
![Comparison image](./documentation/scene_0_frame_0_pose_15_modified.png?raw=true "Visualization of Comparison.")
