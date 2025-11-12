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
#filenames = [filenames[0]]
h5_name = 'RED_DSB_trainsplit.h5'
s_p = np.array([192,192,192])
# step_size = [176, 150, 150]
r_scaling = 4
num_desired = [950,69,1250,1250]

split = SplitData(filenames, s_p, homedir, filedir, h5_name, num_desired, r_scaling)
print('class initialized, getting bboxes and patch starts')

split.forward()

# split.get_meta_bbox()
# split.get_max_bb(split.bbox1, split.find_max_bb)
# patch_starts = split.get_zyx_patch_starts()

# # for i in range(len(filenames)):
# #     patch_starts[i] = patch_starts[i][np.random.randint(0,len(patch_starts[i]),(2))] 

# print('Finding cells_present')
# cells_present = []
# for i in range(len(filenames)):
#     cells_present.append(split.has_cell(patch_starts[i], i))

# print('Getting indices for train test val')
# allcases = [[],[],[]]
# for i in range(len(filenames)):
#     alls = split.get_patch_indices(cells_present[i])
#     for j in range(3):
#         allcases[j].append(np.concatenate((alls[j][0], alls[j][1])))
#                 # print(len(allcases[j][i]))

# print('Beginning patch extraction and saving')
# # select patches based on the index for each imaris file
# phase_id = [[],[],[]]
# for p, phase in enumerate(['Train','Test','Val']):
#     for i, file in enumerate(filenames):
#         save_patches_idcs = allcases[p][i]
#         save_patches_idcs = save_patches_idcs[np.random.randint(0,len(save_patches_idcs), 2)]

#         print(file)
#         pids = split.save_split_patches(i, patch_starts[i], save_patches_idcs, cells_present[i], phase)
#         print('')
#         phase_id[p].append(pids)

#     with h5py.File(f'{self.homedir}/{self.filedir}/{self.h5name}', 'a') as f:
#         tg = f['Metadata']
#         tg.create_dataset(phase, data=np.array(phase_id[p]))

# print('Making json file')
# with h5py.File(f'{homedir}/{filedir}/{h5_name}', 'r') as f:
# # with h5py.File('12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims', 'r') as f:
#     # get the tree structure of the file
#     tree = reg.print_tree(h5_name, f)

#     # write the tree to a JSON file
#     with open(f'{homedir}/{filedir}/JSON_{h5_name}.json', 'w') as file:
#         json.dump(tree, file, indent=4)

# # print status message
# print(f'Finished IMS file tree analysis!')

# print('End')


