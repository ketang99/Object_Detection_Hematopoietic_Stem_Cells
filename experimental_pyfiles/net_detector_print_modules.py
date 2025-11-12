import numpy as np
import h5py
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import math
import pandas as pd
import csv
from matplotlib import pyplot as plt
from importlib import import_module
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.init as init
import sys

epochs = 25
home_dir = '/data/kgupta/registration_testing'
os.chdir(home_dir)
root_dir = home_dir + '/red_dsb'
ims_dir = home_dir + '/h5_files'
train_filename = 'RED_DSB_trainsplit.h5'
mname = f'test_red_dsb_adam_epoch{epochs}_pneq_2patch__e-1'
model_filename = 'singlechannel_red'
model_dir = home_dir + f'/models/{mname}'
model_save_path = f'{model_dir}/epoch_models'
loss_save_path = f'{model_dir}/loss'
sys.path.append(root_dir)
sys.path.append(root_dir + '/training')
sys.path.append(root_dir + '/training/classifier')
print('Added dirs to path')

import reg_functions as reg
import data_red_dsb as dsb
from layers import *
import net_detector_3 as nd
# contains the model
import trainval_detector as det
# contains the function that performs training
from config_training import config as config_training
from split_combine import SplitComb

from utils import *

nodmodel = import_module('net_detector_3')
config, nod_net, loss, get_pbb = nodmodel.get_model()

print('Model imported')
print(type(nod_net))

nod_net = torch.nn.DataParallel(nod_net)
print(type(nod_net))

print(nod_net.modules())

for layer in nod_net.modules():
    print(type(layer))
    if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose3d):
        # Initialize Conv3d layers as desired (e.g., He initialization)
        init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)  # Initialize biases to zeros
    # elif isinstance(layer, nn.BatchNorm3d):
    #     # Initialize BatchNorm3d layers as desired (e.g., Xavier initialization)
    #     init.xavier_uniform_(layer.weight)
    #     init.constant_(layer.bias, 0)  # Initialize biases to zeros
    # Add more conditions for other layer types as needed

print('Inits done')