import numpy as np
import h5py
import os
import math
import pandas as pd
import csv
import reg_functions as reg
import random
from PIL import Image
import yaml

homedir = '/data/kgupta/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'  # all imaris files are stored here
filename = '12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims'
yolo_dir = 'test_yolo_dataset'
patches_dir = 'images'
full_dir = f'{homedir}/{yolo_dir}/{patches_dir}'

all_imgs = os.listdir(full_dir)
print(all_imgs[:10])

all_names = []
for phase in ['Train','Test','Val']:
    with open(f'{yolo_dir}/{phase}.txt', 'r') as f:
        lines = f.readlines()
        all_names.append(lines)

    print(phase)
    print(lines[:10])
    print('')



