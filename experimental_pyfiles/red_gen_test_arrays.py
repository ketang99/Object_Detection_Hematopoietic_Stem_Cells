import numpy as np
import h5py
import os
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
import sys

home_dir = '/home/kgupta/data/registration_testing'
os.chdir(home_dir)
root_dir = home_dir + '/red_dsb'
ims_dir = home_dir + '/h5_files'
train_filename = 'RED_DSB_trainsplit.h5'
model_filename = 'singlechannel_red'
epochs = 50
mname = f'red_dsb_{epochs}'
save_path = home_dir + f'/models/{mname}/epoch_models'
sys.path.append(root_dir)
sys.path.append(root_dir + '/training')
sys.path.append(root_dir + '/training/classifier')
print('')
print('Added dirs to path')

model_checkpoint = f'singlechannel_red_{epochs-1}_state_dict.pth' 

import reg_functions as reg
import data_red_dsb as dsb
from layers import *
import net_detector_3 as nd
# contains the model
import trainval_detector as det
# contains the function that performs training
from config_training import config as config_training
from split_combine import SplitComb

test_dir = 'red_test_arrays'

nodmodel = import_module('net_detector_3')
config, nod_net, loss, get_pbb = nodmodel.get_model()
# nod_net = torch.nn.parallel.DistributedDataParallel(nod_net)
#if torch.cuda.is_available():
    #nod_net = nod_net.cuda()


checkpoint_path = f'{save_path}/{model_checkpoint}'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# If the model was trained using DataParallel, you may need to modify the keys of the state_dict
# checkpoint['model_state_dict'] to checkpoint['state_dict']

# print(checkpoint.keys())

# Load the state_dict into the model
# nod_net.load_state_dict(checkpoint)

model_path = f'singlechannel_red_{epochs-1}.pth'
nod_net = torch.load(f'{save_path}/{model_path}',  map_location=torch.device('cpu'))

print('nod net type: ', type(nod_net))
nod_net = nod_net.module
print('nod net type after .module: ', type(nod_net))


print('model device: ', next(nod_net.parameters()).device)
print('Trained model loaded to cpu')

# Step 3: Set the model to evaluation mode
nod_net.eval()

only_one = False
# read the test phase of the data file and generate a numpy array for each 
with h5py.File(f'{ims_dir}/{train_filename}', 'r') as f:
    pg = f['Patches/Test']
    test_patches = list(pg.keys())
    # print(test_patches)
    for i, pnum in enumerate(test_patches):
        if only_one:
            if i>0:
                break
        else:
            # print(pnum)
            # print(type(pnum))
            patch = pg[f'{pnum}/Patch'][()]
            bbox = pg[f'{pnum}/Patch info/bbox'][()]
            # convert patch to a torch tensor
            patch = torch.from_numpy(patch.astype(np.uint8)).to(torch.float).unsqueeze(0)
            # print('patch device: ', patch.device)
            # run the trained network on the patch
            with torch.no_grad():
                _,out = nod_net(patch)

            # print(out.shape)
            # convert out to a numpy array and save it as a .npy file
            pid = pnum.split()[-1]
            pid = int(pid)
            np.save(f'models/{mname}/testing/test_{pid}.npy', out.numpy())
            np.save(f'models/{mname}/testing/bbox_test_{pid}.npy', bbox)
            
            print(f'Array saved for {pnum}')

print('*****')
print('All test patches processed and arrays saved')
print('*****')
