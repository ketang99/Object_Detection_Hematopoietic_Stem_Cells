import numpy as np
import h5py
import os
import math
import pandas as pd
import csv
from matplotlib import pyplot as plt
import reg_functions as reg
import random

homedir = '/data/kgupta/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'  # all imaris files are stored here
filenames = ['12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims',
             '2w_5_filtered_DP_HSC+annotated.ims',
             '4w_3A_filtered_DP_HSC+annotated.ims',
             '6w_9A_filtered_DP_HSC+annotated.ims']
patch_file = 'RED_DSB_trainsplit.h5'
#filenames = [filenames[0]]
s_p = np.array([192,192,192])
# step_size = [176, 150, 150]
r_scaling = 4
# num_desired = [950,69,1250,1250]
chans = [0,2,4]

phases = ['Train', 'Test', 'Val']
bb_histograms = []


with h5py.File(f'{filedir}/{patch_file}', 'r') as f:
    for p in phases:
        pg = f[f'Patches/{p}']

