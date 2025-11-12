import numpy as np
import h5py
import os
import math
from importlib import import_module
import time
import sys

homedir = '/data/kgupta/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'  # all imaris files are stored here
filenames = ['12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims',
             '2w_5_filtered_DP_HSC+annotated.ims',
             '4w_3A_filtered_DP_HSC+annotated.ims',
             '6w_9A_filtered_DP_HSC+annotated.ims']
#filenames = [filenames[0]]
h5_name = 'RED_DSB_trainsplit.h5'

print('')
patch_ids = []
names = []
with h5py.File(f'{filedir}/{h5_name}', 'r') as f:
    image_id = f['Metadata/Image ID'][()]
    patch_id = f['Metadata/Patch ID'][()]

    for (i, name) in enumerate(filenames):

        pdims = f['Metadata/Patch dimensions'][()]
        actual_name = f[f'Metadata/Image {i}/filename'][()]
        print('file and actual names: ', name, actual_name)
        
        all_patches = np.where(image_id == i)
        imp = patch_id[all_patches[0]]
        print(f'number of patches for image {i}: ', len(all_patches[0]))
        # print('all_patches: ', len(all_patches[0]), all_patches)
        if len(all_patches[0]) > 0:
            select_patches = np.random.choice(len(all_patches[0]), 3)
            print('selected patch idx: ', select_patches)
            patch_ids.append(imp[select_patches]) 
            names.append(name)

        print('')

        
print('selected: ', patch_ids)
print('Image ID: ', len(image_id))
print(image_id)
print('*************')
print('Patch ID:', len(patch_id))
print(patch_id)
print(names)

print('Comparison start')

phases = ['Train', 'Test', 'Val']

phasenames = []
with h5py.File(f'{filedir}/{h5_name}', 'r') as f:
    for p in phases:
        pg = f[f'Patches/{p}']
        phasenames.append(list(pg.keys()))

    print('')
    for i in range(len(names)):
        readname = names[i]
        print(readname)
        pid = patch_ids[i]
        print(pid)

        pstarts = []
        patches = []
        for j in range(len(pid)):
            # first check where it lies
            for (ip,pn) in enumerate(phasenames):
                print(pn)
                if f'Patch {pid[j]}' in pn:
                    pp = f[f'Patches/{phases[i]}/Patch {pid[j]}']
                    patches.append(pp['Patch'][()])
                    print(pp['Patch info/Start_pos'][()])
                    pstarts.append(pp['Patch info/Start_pos'][()])
                    print('Patch read')
                    # print(pstarts[j])
        
# print(phasenames)


# this apprach is a waste of time
# pick random patches, select 3 each from each filename. Select filename from patch info, delimit with '/' and select the last one
# check if this name == filenames[i], if yes then append the patch and patch start somewhere (a dict may be good)