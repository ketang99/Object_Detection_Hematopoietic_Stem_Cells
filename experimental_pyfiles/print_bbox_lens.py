import numpy as np
import h5py
import os
import math
import pandas as pd
import csv
from matplotlib import pyplot as plt
import reg_functions as reg
import random
from split_data_gen import SplitData

homedir = '/data/kgupta/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'  # all imaris files are stored here
filenames = ['12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims',
             '2w_5_filtered_DP_HSC+annotated.ims',
             '4w_3A_filtered_DP_HSC+annotated.ims',
             '6w_9A_filtered_DP_HSC+annotated.ims']

def get_meta_bbox(filenames):
    metadata = []
    bbox0 = []
    bbox1 = []
    for file in filenames:
        # print(file)
        meta = reg.get_ims_metadata(f'{homedir}/{filedir}/{file}', r_scaling=4, desired_cnames = ['HLF tdT'])
        metadata.append(meta)
        extmin = meta['extmin_zyx'][-1::-1]
        extmax = meta['extmax_zyx'][-1::-1]
        imcrop_dims = meta['imdims_zyx']
        # with h5py.File(f'{homedir}/{filedir}/{file}', 'r') as f:
        with h5py.File(f'{homedir}/{filedir}/{file}', 'r') as f:
            p0 = {}
            p0['CoordsXYZR'] = f[f'Scene/Content/Points0/CoordsXYZR'][:]
            p1 = {}
            p1['CoordsXYZR'] = f[f'Scene/Content/Points1/CoordsXYZR'][:]

            bbox0.append(reg.coords_to_3d(p0, extmin, extmax, imcrop_dims[-1::-1], scaling_factor=4))
            bbox1.append(reg.coords_to_3d(p1, extmin, extmax, imcrop_dims[-1::-1], scaling_factor=4))

    return bbox0, bbox1

bbox0, bbox1 = get_meta_bbox(filenames)
for i in range(len(filenames)):
    print(filenames[i])
    print(len(bbox1[i]))
