import numpy as np
import h5py
import os
import math
import pandas as pd
import csv
import random
from PIL import Image
import yaml
import argparse
import json

import sys
homedir = '/home/kgupta/data/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'  # all imaris files are stored here
filenames = ['12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims',
             '2w_5_filtered_DP_HSC+annotated.ims',
             '4w_3A_filtered_DP_HSC+annotated.ims',
             '6w_9A_filtered_DP_HSC+annotated.ims']
# filenames = [filenames[0]]

sys.path.append(homedir)
sys.path.append(f'{homedir}/dsb')

import reg_functions as reg

def dict_to_h5(pg, save_dict):
    for key, value in save_dict.items():
        # print(key)
        if isinstance(value, str):
            pg.create_dataset(key, data=np.array([value]).astype('S'))
        else:
            pg.create_dataset(key, data=value)

# create an h5 file for the bboxes
if not os.path.isfile(f'{filedir}/all_bboxes.h5'):
    with h5py.File(f'{filedir}/all_bboxes.h5', 'w') as f:
        for i,filename in enumerate(filenames):
            ig = f.create_group(f'Image {i}')

for i,filename in enumerate(filenames):
    print('*******')
    print('Getting bboxes for:')
    print(filename)
    print('')
    # read the metadata and combo_img
    for ir,r_scaling in enumerate([1,2,3,4]):
        combo_img, bbox0, bbox1, metadata = reg.get_image(f'{filedir}/{filename}', r_scaling = r_scaling, return_img = False)
        # print(metadata)
        with h5py.File(f'{filedir}/all_bboxes.h5', 'a') as f:
            print(metadata)
            ig = f[f'Image {i}']
            if ir == 0:
                dict_to_h5(ig, metadata)
            ig.create_dataset(f'bbox0_{r_scaling}', data=bbox0)
            ig.create_dataset(f'bbox1_{r_scaling}', data=bbox1)

        print(f'bboxes saved for r_scaling = {r_scaling}')

print('Making json file')
with h5py.File(f'{homedir}/{filedir}/all_bboxes.h5', 'r') as f:
# with h5py.File('12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims', 'r') as f:
    # get the tree structure of the file
    tree = reg.print_tree('all_bboxes.h5', f)

    # write the tree to a JSON file
    with open(f'{homedir}/{filedir}/JSON_all_bboxes.json', 'w') as file:
        json.dump(tree, file, indent=4)

print('json file saved')
