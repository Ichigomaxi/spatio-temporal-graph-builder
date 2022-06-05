# spatio-temporal graph builer
Installation:

Best Practice:
Create a Conda environment
Then continue with the installation of the required packages.

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
pip install dash
pip install jupyter-dash
pip install pandas

4) install pytorch and pytorch-geometric
conda list output:
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

5) install pytorch lightning:
conda list output:
pytorch-lightning         1.5.10             pyhd8ed1ab_0    conda-forge
pip list output:
pytorch-lightning       1.5.10

6) install sacred (at this moment must be installed manually. Download from git and do "python setup.py")
conda list output:
sacred                    0.8.3                    pypi_0    pypi
pip list output:
sacred                  0.8.3

7) install motmetrics  <= 1.1.3
https://github.com/cheind/py-motmetrics
https://pypi.org/project/motmetrics/
pip install motmetrics==1.1.3
motmetrics                1.1.3                    pypi_0    pypi

8) install ujson 5.3.0 
pip install ujson

9) Center Point Detections from https://github.com/tianweiy/CenterPoint/tree/master/configs/nusc