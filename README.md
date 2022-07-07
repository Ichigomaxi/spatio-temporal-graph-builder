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


## Tracking Inference/Validation
After training our model we want to perform tracking and validate our results.
For testing the tracking pipeline within a single graph we have use the `evaluate_single_graphs.py` [script](./evaluate_single_graphs.py). (Architecture part 1 - 3, no global tracking) 
For testing the overall tracking performance we the `evaluate.py` [script](./evaluate.py). The global tracking yields a yaml-file which contains the tracking results in the official nuscenes format. In addition, the evaluation of the results is done by the [official nuscenes evaluation script](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/tracking). This computes the MOT-metrics such as AMOTA, AMOTP

### Local Tracking/ Tracking within a single spatio-temporal Graph ###
We have developed a script which allows to test our tracking method on a graph-by-graph basis. (Architecture part 1 - 3, no global tracking) 
```bash
python evaluate_single_graphs.py
```

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

## Visualization




