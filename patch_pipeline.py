'''
Generate patches for all imaris files
'''

import numpy as np
import h5py
import os
import math
import pandas as pd
import tifffile
import csv
import reg_functions as reg
import math
import json

homedir = '/data/kgupta/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'  # all imaris files are stored here
filenames = ['12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims',
             '2w_5_filtered_DP_HSC+annotated.ims',
             '4w_3A_filtered_DP_HSC+annotated.ims',
             '6w_9A_filtered_DP_HSC+annotated.ims']
h5_name = '0308_SINGLE_red_dsb_patches.h5'

filenames = [filenames[0]]

only_red = True
             
short_names = []
for name in filenames:
    short_names.append(name[:name.find('W')+1])

bbox0_strings = []
bbox1_strings = []

# create h5 file for patches and bboxes if it doesn't already exist
# if not os.path.isfile(os.path.join(filedir, h5_name)):
with h5py.File(f'{homedir}/{filedir}/{h5_name}', 'w') as f:
    f.create_group('Patches')
    mg = f.create_group('Metadata')
    for i in range(len(filenames)):
        mg.create_group(f'Image {i+1}')

# find max bb_size, patch size, step size: these must be the same for all imaris files!
## in a separate notebook, find these values for each imaris file and go from there. See image dimensions also
s_p = np.array([200,200,200])
max_bb = np.array([16.0, 42.0, 42.0])
step_size = s_p - max_bb
print(s_p, max_bb, step_size)
# save the above three variables to metadata (applies to all patches)

# write lines to save s_p, max_bb and step_size to metadata
if only_red:
    desired_cnames = ['HLF tdT']
else:
    desired_cnames = ['DAPI', 'Ctnnal1 GFP', 'HLF tdT']

r_scaling = 4
print(f'Desired channel names: {desired_cnames}')

# get combo_img for the three channels of interest as well as other metadata
for img_num, imsfile in enumerate(filenames):
        
    print(f'Getting image, BBs and metadata for {imsfile}')
    combo_img, bbox0, bbox1, metadict = reg.get_image(f'{homedir}/{filedir}/{imsfile}', r_scaling, desired_cnames=desired_cnames, return_img=True)
    
     # select the tomato channel
    if only_red:
        bbox0 = np.array([])
        if len(combo_img.shape) == 3:
            combo_img = np.expand_dims(combo_img, axis=0)

    print('BB shapes: ', len(bbox0), len(bbox1))
    # write line to write in the filename into metadict

    imdims_zyx = metadict['imdims_zyx']
    print(f'Image dimensions: {imdims_zyx}')
    z_starts = reg.find_patch_starts(step_size[0], imdims_zyx[0], s_p[0])
    y_starts = reg.find_patch_starts(step_size[1], imdims_zyx[1], s_p[1])
    x_starts = reg.find_patch_starts(step_size[2], imdims_zyx[2], s_p[2])

    # print(z_starts, y_starts, x_starts)

    # sample just a few of the starts
    z_starts = z_starts[math.floor(len(z_starts)/2) - 1 : math.floor(len(z_starts)/2) + 2]
    y_starts = y_starts[math.floor(len(y_starts)/2) - 1 : math.floor(len(y_starts)/2) + 2]
    x_starts = x_starts[math.floor(len(x_starts)/2) - 1 : math.floor(len(x_starts)/2) + 2]

    print(z_starts, y_starts, x_starts)

    # Generate the patches: tweak the function to output the return the start and end patch numbers
    phase = 'train'
    print('Generating patches')
    patch_range, has_cell = reg.generate_patches(combo_img, h5_name, imsfile, filedir, bbox0, bbox1, s_p, max_bb, step_size, z_starts, y_starts, x_starts)
    print('Done')

    metadict['Patch_range'] = patch_range
    metadict['z_starts'] = z_starts
    metadict['y_starts'] = y_starts
    metadict['x_starts'] = x_starts

    metadict['num_DP'] = len(bbox0)
    metadict['num_HSPC'] = len(bbox1)
    metadict['patch_with_cell'] = has_cell

    with h5py.File(f'{homedir}/{filedir}/{h5_name}', 'a') as f:
        mgi = f[f'Metadata/Image {img_num+1}']
        for key, value in metadict.items():
            print(key, value)
            if isinstance(value, str):
                mgi.create_dataset(key, data=np.array([value]).astype('S'))
            else:
                mgi.create_dataset(key, data=value)

    print('Saved metadata')
    print('\n')

print('Making json file')
with h5py.File(f'{homedir}/{filedir}/{h5_name}', 'r') as f:
# with h5py.File('12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims', 'r') as f:
    # get the tree structure of the file
    tree = reg.print_tree(h5_name, f)

    # write the tree to a JSON file
    with open(f'{homedir}/{filedir}/JSON_{h5_name}.json', 'w') as file:
        json.dump(tree, file, indent=4)

# print status message
print(f'Finished IMS file tree analysis!')

# Establish the max_bb size, patch dimensions, step size and the starting point for all patches
# imdims_zyx = imcrop_dims[-1::-1]
# max0 = get_max_bbox_sizes(bbox0)
# max1 = get_max_bbox_sizes(bbox1)
# max_bb = np.max(np.array([max0,max1]), axis=0)
# s_p = np.array([59,140,140])
# step_size = s_p - max_bb
# z_starts = reg.find_patch_starts(step_size[0], imdims_zyx[0], s_p[0])
# y_starts = reg.find_patch_starts(step_size[1], imdims_zyx[1], s_p[1])
# x_starts = reg.find_patch_starts(step_size[2], imdims_zyx[2], s_p[2])
# patch_vol = np.prod(s_p)