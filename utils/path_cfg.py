'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import os.path as osp
import pathlib


########################################################################################
# Specification of paths where dataset and output results will be stored
########################################################################################

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()

# Absolute path where datasets and processed data (e.g. precomputed embeddings) will be stored
DATA_PATH = '/media/HDD2/Datasets/mini_nusc'
# DATA_PATH = '/media/HDD2/Datasets/nuscenes2'

# Absolute path where results (e.g. output result files, model weights) will be stored
OUTPUT_PATH = '/media/HDD2/students/maximilian/output'

if DATA_PATH is None:
    DATA_PATH = osp.join(PROJECT_PATH, 'data')

if OUTPUT_PATH is None:
    OUTPUT_PATH = osp.join(PROJECT_PATH, 'output')
