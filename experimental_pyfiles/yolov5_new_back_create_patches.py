'''
This is a modification of yolov5_create_patches.py
It generate patches as jpeg files and ensures that all patches contain at least one DP or HSPC
'''

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
from matplotlib import pyplot as plt

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

# to be parsed in from terminal: ax_fix, dataset_dir

parser = argparse.ArgumentParser(description='YOLO dataset generator')
parser.add_argument('-d', '--datadir', default='yolo_dataset')
parser.add_argument('-a', '--ax_fix', default=0)
parser.add_argument('-s', '--imsave', default=0)
parser.add_argument('-r', '--rscale', default=3)

args = parser.parse_args()
# global args

ax_fix = int(args.ax_fix)
print('ax_fix:')
print(ax_fix)
# print(ax_name)

ax_names = ['Z','Y','X']
ax_name = ax_names[ax_fix]

print('args.imsave:')
print(args.imsave)
imsave = args.imsave
if imsave==1:
    print('is 1')
    imsave = True
elif imsave==0:
    print('is 0')
    imsave = False

print('imsave?')
print(imsave)

if imsave:
    print('trudat')
else:
    print('faldat')

# filename = '12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims'
yolo_dir = f'{args.datadir}_{ax_name}_r{args.rscale}'
patches_dir = 'images'
full_dir = f'{homedir}/{yolo_dir}/{patches_dir}'

print(yolo_dir)

if imsave:
    if yolo_dir not in os.listdir(homedir):
        os.mkdir(yolo_dir)
        os.mkdir(f'{yolo_dir}/images')
        os.mkdir(f'{yolo_dir}/labels')
    print('created dataset dir and subdirs')
    # create the yaml file
    data = {
        'path': f'{homedir}/{yolo_dir}',
        'train': 'full_Train.txt',
        'val': 'full_Val.txt',
        'test': 'full_Test.txt',  # optional, None if not specified
        'names': {
            0: 'DP',
            1: 'HSPC'
        }
    }
    
    with open(f'{yolo_dir}/dataset.yaml', 'w') as file:
        yaml.dump(data, file)


############################################################################################################
# ###########################################################################################################
# FUNCTION DEFINITIONS

# function that will extract all the bounding boxes for a particular plane (generalized)
def retrieve_planar_bboxes(bbox0,bbox1,ax_fix,ax_val):
    bb_lims = []
    bb_lims.append(bbox0[np.logical_and(bbox0[:,2*ax_fix]<=ax_val, bbox0[:,2*ax_fix+1]>=ax_val)])
    bb_lims.append(bbox1[np.logical_and(bbox1[:,2*ax_fix]<=ax_val, bbox1[:,2*ax_fix+1]>=ax_val)])

    return bb_lims

# function to generate patches in 2D
# the returned patches here will be applicable for all the planes along the chosen axis
def find_patch_starts(single_step_size, single_imdim, single_sp):

    starts = [0]
    p_current = starts[0]
    p_current+=single_step_size
    thresh = single_imdim - single_sp

    while thresh >= p_current:
        starts.append(p_current)
        p_current+=single_step_size

    starts = np.asarray(starts)
    
    return starts.astype(int)

def patch_bbox(current_patch, s_p, bb_lims, ax_fix):

    '''
    This function will return an array with 2 entries which correspond to the bboxes of each cell type
    The returned array corresponds to a single patch
    This will truncate bboxes that partially lie in a patch and return them

    bb_lims is all the boxes that lie on a certain plane: each box has 6 terms zl,zu,yl,yu,xl,xu
        This includes all dims so dims corresponding to 2*ax_fix,2*ax_fix+1 must be removed
    current_patch is the 2D patch pixel start location, current_patch + s_p is the upper bound of the patch

    returns patch_bbs as a two term array, each term corresponding to the cell type. Each of the 2 terms is allowed
    to have len == 0
    '''
    
    patch_bbs = []
    # iterate over each cell type
    for i in range(len(bb_lims)):
        patch_bb = []
        if len(bb_lims[i]) != 0:
            # iterate over each individual bbox
            for j in range(len(bb_lims[i])):
                # selects a single bb
                # print(bb_lims[i][j])
                # print('')
                bb_shifted = np.delete(bb_lims[i][j], [2*ax_fix, 2*ax_fix+1])
                # shifts the bb based on the current patch
                bb_shifted[[0,1]] -= current_patch[0]
                bb_shifted[[2,3]] -= current_patch[1]
                # print(bb_shifted)
                
                '''
                bb_shifted contains a 4 term bounding box with ax_fix's entries removed
                '''
                valids = []
                for ax in range(2):
                    # bbshifted[2ax] must be less than [2ax+1] and not equal AND upper bound must be positive
                    if bb_shifted[2*ax] < bb_shifted[2*ax+1] and bb_shifted[2*ax+1] > 0:
                    
                        '''
                        acceptable cases: LOWER BOUND < UPPER BOUND ALREADY GUARANTEED BY THE ABOVE IF STATEMENT
                        1. lower bound < 0 and upper bound <= s_p[ax]
                        2. lower bound >=0 and upper bound <= s_p[ax]
                        3. lower and upper bounds <= s_p[ax]
                        4. lower bound >= 0 and upper bound > s_p[ax[

                        So basically either the lower or the upper need to be in bounds
                        '''
                        
             #           print('current patch: ', current_patch)
              #          print('bb_shifted: ', bb_shifted)
                        if bb_shifted[2*ax+1] > 0 and bb_shifted[2*ax+1] <= s_p[ax]:
                            upper_inbound = True
                        else:
                            upper_inbound = False
                        
                        if bb_shifted[2*ax] >= 0 and bb_shifted[2*ax] <= s_p[ax]:
                            lower_inbound = True
                        else:
                            lower_inbound = False
                            
                        
                        if upper_inbound or lower_inbound:
                            valids.append(True)
                        else:
                            valids.append(False)

                        #if bb_shifted[2*ax] >= 0 and bb_shifted[2*ax+1] <= s_p[ax]: 
                        #    valids.append(True)
                        #elif bb_shifted[2*ax] >= 0 and bb_shifted[2*ax] <= s_p[ax] :
                        #    valids.append(True)
                        # ensures that upper bound is in (0,s_p[ax]]
                        #elif bb_shifted[2*ax+1] > 0 and bb_shifted[2*ax+1] <= s_p[ax]:
                        #    valids.append(True)
                        #else:
                        #    valids.append(False)
                    
                    else:
                        valids.append(False)

                # print(valids)
                
                # below statement is gonna see if the lower or upper limits of bb_shifted are outside patch bounds
                if valids[0] and valids[1]:
                    # print(i)
                    # print('Valid: ', bb_shifted)
                    bbv = []
                    for ax in range(2):
                        # bounds of dim0
                        if bb_shifted[2*ax] < 0:
                            bbv.append(0)
                        else:
                            bbv.append(bb_shifted[2*ax])
                        # bounds of dim1
                        if bb_shifted[2*ax + 1] > s_p[ax]:
                            bbv.append(s_p[ax])
                        else:
                            bbv.append(bb_shifted[2*ax+1])

                    # print('bbv: ', bbv)
                    assert len(bbv) == 4
                    patch_bb.append(bbv)

            patch_bb = np.array(patch_bb)
            #print('patch_bb: ', patch_bb)
            #print('')
        
        else:
            patch_bb = np.array([])

        patch_bbs.append(patch_bb)

    # returns patch_bbs as a two term array, each term corresponding to the cell type
    return patch_bbs

def get_normalized_bboxes(current_patch, s_p, bb_lims, ax_fix):

    '''
    returns normalized boxes in the form [type, c0, c1, w0, w1]
    '''

    patch_bbs = patch_bbox(current_patch, s_p, bb_lims, ax_fix)

    # patch_bbs has taken care of bounding boxes that have bounds outside s_p by assigning the boundary as s_p

    norm_bbs = []
    for i in range(len(patch_bbs)):
        btype = patch_bbs[i]
        for b in range(len(btype)):
            bb = btype[b]
            c0 = 0.5 * (bb[0] + bb[1]) / s_p[0]
            c1 = 0.5 * (bb[2] + bb[3]) / s_p[1]
            w0 = bb[1] - bb[0]
            w0 /= s_p[0]
            w1 = bb[3] - bb[2]
            w1 /= s_p[1]
        
            # ensure that all bounding box widths > 0
            if w0 > 0 and w1 > 0:
                if (c0 + w0/2 <=1.0) and (c1 + w1/2 <= 1.0): 
                    if (c0 - w0/2 >=0.0) and (c1 - w1/2 >= 0.0):
                        norm_bbs.append([i+1, c0, c1, w0, w1])
    
    return norm_bbs


def patch_split(idcs, train_pct=0.8, test_pct=0.1, val_pct=0.1):
    splits = [int(train_pct * len(idcs)), int(test_pct * len(idcs)), int(val_pct * len(idcs))]
    train_cell = idcs[:splits[0]]
    test_cell = idcs[splits[0]:splits[0] + splits[1]]
    val_cell = idcs[splits[0]+splits[1]:]

    return train_cell, test_cell, val_cell

def get_patch_indices(cells_present):

    '''
    Has been modified to only return patch idcs which have >0 cells
    '''
    
    # the np.where() operations yield indexes. These indexes will be split and then be used to retrieve 
    # what is needed from the four big arrays.
    
    num_p = len(cells_present)
    num_cell = np.count_nonzero(cells_present)
    num_nc = round(num_cell * 0.25*1/0.75)
    print('num_cell and num_nocell', [num_cell, num_nc])
    cell_idx = np.where(np.array(cells_present)!=0)
    
    # print(cell_idx)
    no_cell_idx = np.where(np.array(cells_present)==0)
    # print(no_cell_idx, len(no_cell_idx[0]), len(cell_idx))
    cell_idx = cell_idx[0]
    print('len cell idx: ', len(cell_idx))
    no_cell_idx = no_cell_idx[0]
    print(f'num of no cells: {len(no_cell_idx)}, actual no cells = {num_nc}')
    if len(no_cell_idx) > num_nc:
        rc = np.random.choice(len(no_cell_idx), num_nc, replace=False).astype(int)
        no_cell_idx = no_cell_idx[rc]

    # print(type(no_cell_idx), type(cell_idx))

    # print(cell_idx)
    
    np.random.shuffle(cell_idx)
    # now divide the patches with cells among training, testing and validation (80,10,10)
    traincell, testcell, valcell = patch_split(cell_idx)

    # do the same for cells without patches
    # np.random.shuffle(no_cell_idx)
    # trainno, testno, valno = patch_split(no_cell_idx)

    return traincell, testcell, valcell


def generate_grid_boxes(image_height, image_width, box_height, box_width):
    # returns the grid boxes for background in the form yxyx
    all_positions = []
    for y in range(0, image_height, box_height):
        for x in range(0, image_width, box_width):
            all_positions.append([y, x, min(y + box_height, image_height), min(x + box_width, image_width)])
    return all_positions

def bbox_to_corners(y_center, x_center, y_width, x_width):
    """Convert bounding box from center format to corner format."""
    y1 = y_center - y_width / 2
    y2 = y_center + y_width / 2
    x1 = x_center - x_width / 2
    x2 = x_center + x_width / 2
    return np.array([y1, x1, y2, x2])

def intersection_over_union(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.
    
        expects boxes to be of the form y_l, x_l, y_u, x_u
    """

    if len(box1) == 0 or len(box2) == 0:
        return np.array([])
    else:
        y1_1, x1_1, y2_1, x2_1 = box1
        y1_2, x1_2, y2_2, x2_2 = box2
        # Determine the (x, y)-coordinates of the intersection rectangle
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        # Compute the area of intersection rectangle
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        
        # Compute the area of both the prediction and ground-truth rectangles
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Compute the Intersection over Union (IoU)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        
        return iou

############################################################################################################
# ###########################################################################################################

# general flow
s_p = np.array([[192,192],[192,192],[192,192]])

# s_p should be dependent on the fixed axis: if Z then [640,640] else [50,640]

max_bb = np.array([17,50,65])
max_bb = np.delete(max_bb, ax_fix)
mean_bb = np.array([12,40,40])
mean_bb = np.delete(mean_bb, ax_fix)
s_p = s_p[ax_fix]
step_size = s_p - max_bb
r_scaling = args.rscale

background_boxes = np.array(generate_grid_boxes(s_p[0], s_p[1], mean_bb[0], mean_bb[1])).astype(float)
# background_boxes_norm = np.zeros(background_boxes.shape)
print('background boxes:')
print(background_boxes)
background_boxes[:,[0,2]] = background_boxes[:,[0,2]] / s_p[0]
background_boxes[:,[1,3]] = background_boxes[:,[1,3]] / s_p[1]
print('normalized background boxes:')
print(background_boxes)
# background_boxes has now been normalized and is in the form yl,xl,yu,xu

print('max_bb: ', max_bb)
print('s_p: ', s_p)
print('step_size: ', step_size)

# strides = [12, 150, 150]
# stride = strides[ax_fix]
patches_per_file = 3750
num_split_set = 4000
split_pct = [0.8, 0.1, 0.1]  # train/test/val split of dataset

all_backs = []
for back_bb in background_boxes:
    print('\ncurrent back bb')
    print(back_bb)
    all_ious_below = True
    if all_ious_below:
        bbb = [0]
        bbb.append(0.5*(back_bb[2]+back_bb[0]))
        bbb.append(0.5*(back_bb[3]+back_bb[1]))
        bbb.append(back_bb[2]-back_bb[0])
        bbb.append(back_bb[3]-back_bb[1])
        all_backs.append(bbb)

print('all_backs')
print(all_backs)

def main(imsave=False):
    #return 
    for (fnum, filename) in enumerate(filenames):
        print('*******')
        print('Getting patches for:')
        print(fnum, filename)
        print('')
        # read the metadata and combo_img
        if imsave:
            combo_img, bbox0, bbox1, metadata = reg.get_image(f'{filedir}/{filename}', r_scaling, return_img = True)
            num_channels = combo_img.shape[0]
            imdims_zyx = combo_img.shape[1:]
            imdims = np.delete(imdims_zyx, ax_fix)
            assert num_channels == 3
            print('Image has been read from ims file')
        else:
            combo_img, bbox0, bbox1, metadata = reg.get_image(f'{filedir}/{filename}', return_img = False)
            imdims_zyx = metadata['imdims_zyx']
            imdims = np.delete(imdims_zyx, ax_fix)
            print('No image has been read')

        with h5py.File(f'{homedir}/{filedir}/all_bboxes.h5', 'r') as f:
            bbox0 = f[f'Image {fnum}/bbox0_{r_scaling}'][:]
            bbox1 = f[f'Image {fnum}/bbox1_{r_scaling}'][:]
        
       # imdims_zyx = metadata['imdims_zyx']
        print('Image dims full: ', imdims_zyx)
        print('image dims with axfix removed: ', imdims)
       # imdims = np.delete(imdims_zyx, ax_fix)
        print('step_size: ', step_size)
        print('patch dims s_p: ', s_p)
        
        print('bbox shapes: ', bbox0.shape, bbox1.shape)
        if len(bbox0) == 0 or len(bbox1) == 0:
            print('ONE OF THE BBOXES IS 0')
            return

        starts = []
        for i in range(2):
            starts.append(find_patch_starts(step_size[i], imdims[i], s_p[i]))
        
        # create patch_starts array with all patches
        ps = []
        for s0 in starts[0]:
            for s1 in starts[1]:
                ps.append([s0, s1])
        
        patch_starts = np.asarray(ps)

        print('before split/select patch starts len: ', len(patch_starts))
        print(' patch starts shape: ', patch_starts.shape)
        num_p_planar = len(patch_starts)
        # imdims = metadata['imdims_zyx']
        # imdims = np.delete(imdims, ax_fix)

        # all_z = []
        # for z in range(metadata['imdims_zyx'][ax_fix]):
        #     all_z.append(np.ones(num_p_planar))

        # all_z = np.array(all_z)
        # all_z = np.ravel(all_z)

        # z_save = np.random.choice(all_z, 12500, replace=False)

        # slices_per_z = math.floor(patches_per_file / metadata['imdims_zyx'][ax_fix]) 
        # if slices_per_z == 0:
        #     slices_per_z = 1
        #     acceptable_z = np.random.choice(metadata['imdims_zyx'][ax_fix], patches_per_file, replace=False)
        #     use_all_ps = False
        # else:
        #     acceptable_z = np.arange(metadata['imdims_zyx'][ax_fix])
        #     use_all_ps = True

        acceptable_z = np.arange(metadata['imdims_zyx'][ax_fix])
        use_all_ps = False
        print('acceptable z: ', acceptable_z)
        zs = []
        patch_has_cell = []
        norm_bbs = []
        all_patch_starts = []
        patches_per_slice = round(12 * num_split_set / len(acceptable_z))
        print('patches_per_slice: ', patches_per_slice)
        if len(patch_starts) <= patches_per_slice:
            use_all_ps = True
        else:
            use_all_ps = False

        # for z in range(metadata['imdims_zyx'][ax_fix]):
        for z in acceptable_z:
            # get the bounding boxes for that z:
            # print(z, 'getting bboxes for this z')
            bb_lims = retrieve_planar_bboxes(bbox0,bbox1,ax_fix,z)
            # print('got bboxes')

            if use_all_ps:
                selected_patch_starts = patch_starts
            else:
                s_inds = np.random.choice(len(patch_starts), round(20 * num_split_set / len(acceptable_z)))
                selected_patch_starts = patch_starts[s_inds]
                # print('len selected patch starts: ', len(selected_patch_starts))
                # return

            
            for start in selected_patch_starts:
                zs.append(z)
                all_patch_starts.append(start)
                # get normalized bounding boxes
                nbb = get_normalized_bboxes(start, s_p, bb_lims, ax_fix)
                norm_bbs.append(nbb)
                if len(nbb) != 0:
                    patch_has_cell.append(1)
                else:
                    patch_has_cell.append(0)

        '''
        the above for loop gives us three key things:
        1. all_patch_starts and zs
        2. norm_bbs: normalized bbs
        3. patch_has_cell
        These 3 are all based on s_inds and therefore their lengths are the same and the indices are exactly the same 
        '''

        del patch_starts

        print('Patch has cell length before split/select: ', len(patch_has_cell))
        # get the indices for train/test/val
        train_idx, test_idx, val_idx = get_patch_indices(patch_has_cell)
        
        '''
        get_patch_indices HAS BEEN MODIFIED TO ENSURE THAT ONLY PATCHES WITH BBOXES ARE TO BE SAVED
        '''

        # train_idcs = np.concatenate((train_idx[0], train_idx[1]))
        # test_idcs = np.concatenate((test_idx[0], test_idx[1]))
        # val_idcs = np.concatenate((val_idx[0], val_idx[1]))
        train_idcs = np.array(train_idx)
        test_idcs = np.array(test_idx)
        val_idcs = np.array(val_idx)
        del train_idx, test_idx, val_idx

        print('Train test val patches shape before getting desired number: ')
        print(train_idcs.shape, test_idcs.shape, val_idcs.shape)
        #return
        # print(train_idcs[:10])
        
        if len(train_idcs) > int(0.8*num_split_set) and len(test_idcs) > int(0.1*num_split_set) and len(val_idcs) > int(0.1*num_split_set):
            train_idcs = np.random.choice(train_idcs, int(0.8*num_split_set), replace=False)
            test_idcs = np.random.choice(test_idcs, int(0.1*num_split_set), replace=False)
            val_idcs = np.random.choice(val_idcs, int(0.1*num_split_set), replace=False)

        print('Train test val patches shape: ')
        print(train_idcs.shape, test_idcs.shape, val_idcs.shape)
        print('')

        save_idcs = [train_idcs, test_idcs, val_idcs]
        phases = ['Train', 'Test', 'Val']
        
        # for ip, phase in enumerate(phases):
        #     print(phase)
        #     print(save_idcs[ip])
        #     print('\n')
        #     print('*********')

        if imsave:
            if len(save_idcs[0]) == 0:
                print('save idcs has nothing')
                break
            # iterate through each of the three lads above 
            for ip, phase in enumerate(phases):
                # retrieve z, patch_start, norm_bbs, patch_has_cell
                starting_patches = len(os.listdir(f'{yolo_dir}/images'))
                num_patches = len(os.listdir(f'{yolo_dir}/images'))
                phase_names = []
                phase_starts = []
                phase_num = num_split_set * split_pct[ip]
                # iterate through save_idcs for each phase
                for idx in save_idcs[ip]:
                    num_patches+=1

                    if num_patches - starting_patches >= phase_num:
                        break
                    else:
                    
                        plane = zs[idx]
                        patch_start = all_patch_starts[idx]
                        norm_bb = norm_bbs[idx]
                        has_cell = patch_has_cell[idx]
            
                        patchbounds = np.zeros((2,2))
                        # first column of patchbounds is the start, second column is the end i.e. start+s_p
                        patchbounds[:,0] = patch_start
                        patchbounds[:,1] = patch_start + s_p
                        patchbounds = patchbounds.astype(int)
                        # img_shape = combo_img.shape[1:]
            
                        patch_edge = patchbounds[:,1] > imdims
            
                        for i in range(2):
                            if patch_edge[i]:
                                patchbounds[i,1] = imdims[i]
            
                        # patch images are being padded with 0 for edge cases
                        img_save = np.zeros((num_channels, s_p[0], s_p[1])).astype(np.uint8)
                        if ax_fix == 0:
                            crop = combo_img[:, plane, patchbounds[0,0]:patchbounds[0,1], patchbounds[1,0]:patchbounds[1,1]]
                            full_patch_start = [plane, patch_start[0], patch_start[1]]
                        elif ax_fix == 1:
                            crop = combo_img[:, patchbounds[0,0]:patchbounds[0,1], plane, patchbounds[1,0]:patchbounds[1,1]]
                            full_patch_start = [patch_start[0], plane, patch_start[1]]
                        elif ax_fix == 2:
                            crop = combo_img[:, patchbounds[0,0]:patchbounds[0,1], patchbounds[1,0]:patchbounds[1,1], plane]
                            full_patch_start = [patch_start[0], patch_start[1], plane]
                            
                        img_save[:, :crop.shape[1], :crop.shape[2]] = crop
                        img_save = img_save.astype('uint8')
                        img_save = np.transpose(img_save, [1,2,0])
                        # print(img_save.shape)
                        img_name = f'img_{num_patches}'
                        img = Image.fromarray(img_save)
                        img.save(f'{yolo_dir}/images/{img_name}.jpg')
                        phase_names.append(f'{img_name}.jpg')
                        phase_starts.append(full_patch_start)
    
                        # go through the background boxes and save those which have an iou<0.5 as class 0
                        all_backs = []
                        for back_bb in background_boxes:
                            all_ious_below = True
                            for norm_bb_check in norm_bb:
                                nbb = bbox_to_corners(*norm_bb_check[1:])        
                                # find iou with the single norm_bb
                                iou = intersection_over_union(back_bb, nbb)
                                if iou >= 0.5:
                                    all_ious_below = False
                                    break

                            if all_ious_below:
                                bbb = [0]
                                bbb.append(0.5*(back_bb[2]+back_bb[0]))
                                bbb.append(0.5*(back_bb[3]+back_bb[1]))
                                bbb.append(back_bb[2]-back_bb[0])
                                bbb.append(back_bb[3]-back_bb[1])
                                all_backs.append(bbb)
    
                        # norm_bb: save it as a .txt file with each line representing a separate object
                        # saves the normalized bboxes for each patch into the labels directory

                        with open(f'{yolo_dir}/labels/{img_name}.txt', 'w') as f:
                            # saves the bounding boxes for classes 1 and 2 (FOREGROUND)
                            for row in norm_bb:
                                f.write(f"{int(row[0])} {' '.join(map(str, row[1:]))}\n")
                            # saves the bounding boxes for BACKGROUND
                            for row in all_backs:
                                f.write(f"{int(row[0])} {' '.join(map(str, row[1:]))}\n")
            
                print(f'Saved images and labels for {phase}')
                # create a txt file for each phase. This file will contain the full dirs for each phase image
                if f'full_{phase}.txt' not in os.listdir(f'{yolo_dir}'):
                    mode = 'w'
                else:
                    mode = 'a'

                with open(f'{yolo_dir}/full_{phase}.txt', mode) as f:
                    for imgfile in phase_names:
                        f.write(f'{full_dir}/{imgfile}\n')
                

                if f'starts_{phase}.txt' not in os.listdir(f'{yolo_dir}'):
                    mode = 'w'
                else:
                    mode = 'a'

                with open(f'{yolo_dir}/starts_{phase}.txt', mode) as f:
                    for i in range(len(phase_starts)):
                        pss = phase_starts[i]
                        f.write(f'{pss[0]}_{pss[1]}_{pss[2]},')
                        f.write(f'{full_dir}/{phase_names[i]},')
                        f.write(f'{filename}\n')
            
                print('    Saved the txt file')
    
        del combo_img
    
    print('END')

if __name__ == '__main__':
    main(imsave)
