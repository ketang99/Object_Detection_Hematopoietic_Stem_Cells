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
#filenames = [filenames[0]]
s_p = np.array([192,192,192])
# step_size = [176, 150, 150]
r_scaling = 4
# num_desired = [950,69,1250,1250]
chans = [0,2,4]

for filename in filenames:
    print('')
    print(filename)
    combo_img, bbox0, bbox1, metadata = reg.get_image(f'{filedir}/{filename}', return_img = True)
    for (i,ch) in enumerate(chans):
        print('Channel ID: ', ch)
        print('    Min intensity: ', np.min(combo_img[i]))
        print('    Max intensity: ', np.max(combo_img[i]))
        
        total_v = np.prod(combo_img[i].shape)
        pos_v = np.count_nonzero(combo_img[i])
        print('    Sparsity ratio = ', (total_v-pos_v)/total_v)

    del combo_img
    print('*****')
