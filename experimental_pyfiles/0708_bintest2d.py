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
from tifffile import imwrite
from dsb import reg_functions as reg

import sys
homedir = '/home/kgupta/data/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'

with h5py.File(f'{homedir}/{filedir}/all_bboxes.h5', 'r') as f:
    bbox0 = f['Image 0/bbox0_3'][:]
    bbox1 = f['Image 0/bbox1_3'][:]
    imdims_zyx = f['Image 0/imdims_zyx'][:]

ax_fix = 0
bboxes = [bbox0, bbox1]

# get total voxel volume of each bbox type
def get_total_bbox_volume(bbox):
    indiv_vols = []
    total_vol = 0

    for bb in bbox:
        # print(bb)
        iter_vol = np.prod([bb[2*i+1]+1-bb[2*i] for i in range(3)])
        # print([bb[2*i+1]-bb[2*i] for i in range(3)], iter_vol)
        total_vol += iter_vol
        indiv_vols.append(iter_vol)
        # print(total_vol)
        # print('')

    return indiv_vols, total_vol

def patch_bbox(current_patch, s_p, bb_lims, ax_fix):

    '''
    This function will return an array with 2 entries which correspond to the bboxes of each cell type
    The returned array corresponds to a single patch
    This will truncate bboxes that partially lie in a patch and return them
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
                bb_shifted = np.delete(bb_lims[i][j], [ax_fix, ax_fix+1])
                # shifts the bb based on the current patch
                bb_shifted[[0,1]] -= current_patch[0]
                bb_shifted[[2,3]] -= current_patch[1]
                # print(bb_shifted)
                
                valids = []
                for ax in range(2):
                    if bb_shifted[2*ax] >= 0 and bb_shifted[2*ax+1] <= s_p[ax]: 
                        valids.append(True)
                    elif bb_shifted[2*ax] >= 0 and bb_shifted[2*ax] <= s_p[ax] :
                        valids.append(True)
                    elif bb_shifted[2*ax+1] >= 0 and bb_shifted[2*ax+1] <= s_p[ax]:
                        valids.append(True)
                    else:
                        valids.append(False)

                # print(valids)
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
        
        else:
            patch_bb = np.array([])

        patch_bbs.append(patch_bb)

    # returns patch_bbs as a two term array, each term corresponding to the cell type
    return patch_bbs

def get_normalized_bboxes(current_patch, s_p, bb_lims, ax_fix):

    patch_bbs = patch_bbox(current_patch, s_p, bb_lims, ax_fix)

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

            norm_bbs.append([i, c0, c1, w0, w1])

    return norm_bbs

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

def convert_bounding_boxes(bboxes):
    # Convert the format [x_center, y_center, x_width, y_width]
    # to [x_begin, x_end, y_begin, y_end]
    converted_bboxes = np.zeros_like(bboxes)
    converted_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_begin
    converted_bboxes[:, 1] = bboxes[:, 0] + bboxes[:, 2] / 2  # x_end
    converted_bboxes[:, 2] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_begin
    converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y_end
    return np.round(converted_bboxes).astype(int)

indiv_vols = []
total_vols = []

for i, bbox in enumerate(bboxes):
    indiv, tot = get_total_bbox_volume(bbox)
    indiv_vols.append(indiv)
    total_vols.append(tot)

# function that will extract all the bounding boxes for a particular plane (generalized)
def retrieve_planar_bboxes(bbox0,bbox1,ax_fix,ax_val):
    bb_lims = []
    bb_lims.append(bbox0[np.logical_and(bbox0[:,2*ax_fix]<=ax_val, bbox0[:,2*ax_fix+1]>=ax_val)])
    bb_lims.append(bbox1[np.logical_and(bbox1[:,2*ax_fix]<=ax_val, bbox1[:,2*ax_fix+1]>=ax_val)])

    return bb_lims

s_p = np.array([[192,192],[48,192],[48,192]])

# s_p should be dependent on the fixed axis: if Z then [640,640] else [50,640]

max_bb = np.array([17,50,65])
# s_p = [48, 192, 192]
max_bb = np.delete(max_bb, ax_fix)
s_p = s_p[ax_fix]
step_size = s_p - max_bb

print('max_bb: ', max_bb)
print('s_p: ', s_p)
print('step_size: ', step_size)
imdims = np.delete(imdims_zyx, ax_fix)
print('image dims with axfix removed: ', imdims)

starts = []
for i in range(2):
    starts.append(find_patch_starts(step_size[i], imdims[i], s_p[i]))

# create patch_starts array with all patches
ps = []
for s0 in starts[0]:
    for s1 in starts[1]:
        ps.append([s0, s1])

patch_starts = np.asarray(ps)

zs = []
patch_has_cell = []
norm_bbs = []
all_patch_starts = []
# for z in range(imdims_zyx[ax_fix]):
num_bboxes = [0,0] 
numcheck_total = [0,0]
new_vols = [0,0]

bin_img = np.zeros((2,imdims_zyx[0],imdims_zyx[1],imdims_zyx[2])).astype(np.uint8)

for z in range(imdims_zyx[ax_fix]):
    # get the bounding boxes for that z:
    # print(z, 'getting bboxes for this z')
    bb_lims = retrieve_planar_bboxes(bbox0,bbox1,ax_fix,z)

    # if z > 1: 
    #     break

    #print(z)
    # print('bb_lims i.e. planar: ', bb_lims)

    selected_patch_starts = patch_starts

    numbb = 0
    for counts, ps in enumerate(selected_patch_starts):
        zs.append(z)
        all_patch_starts.append(ps)
        # get normalized bounding boxes
        
        nbb = get_normalized_bboxes(ps, s_p, bb_lims, ax_fix)
        nbb = np.array(nbb)
        # print('nbb shape: ', nbb.shape)
        # print('nbb: ', nbb)
        # print(z,start)
        if len(nbb) != 0:
            ogbb = patch_bbox(ps, s_p, bb_lims, ax_fix)
            num_bboxes[0] += len(ogbb[0])  
            num_bboxes[1] += len(ogbb[1])
            # print('original bbox: ', ogbb)
            # print('normalized: ', nbb)
            nbb[:,[1,3]] *= s_p[0]
            nbb[:,[2,4]] *= s_p[1]

            #separate the classes of nbb
            nbbs = []
            for jj in range(2):
                bbcheck = nbb[:,0]==jj
                if len(bbcheck) > 0:
                    nbbs.append(nbb[bbcheck])
                    #num_bboxes[jj] += len(nbb[bbcheck])
                else:
                    nbbs.append(np.array([]))
            
            # print('denorm: ', nbb)
            cbb = []
            for jj in range(2):
                if len(nbbs[jj]) > 0:
                    cbb.append(convert_bounding_boxes(nbbs[jj][:,1:]))
                else:
                    cbb.append(np.array([]))
            # print('converted to original: ', cbb)

            # print(len(cbb))
            numcheck = 0
            for jj in range(2):
                # print(f'cbb{jj}: ', cbb[jj])
                numcheck = 0
                if len(ogbb[jj]) > 0:
                    # bbcheck = nbb[:,0]==jj
                    # print(f'bb{jj} check: ', bbcheck)
                    # if len(bbcheck) > 0:
                    #     print('extracted nbb: ', nbb[bbcheck])
                    diffs = [cbb[jj][:,2*j+1] - cbb[jj][:,2*j] for j in [0,1]]
                    # print('diffs: ', diffs)
                    # print(np.prod(diffs,axis=0))
                    new_vols[jj] += np.sum(np.prod(diffs,axis=0))
                    concheck = ogbb[jj] == cbb[jj] 
                    # print(concheck, len(concheck))
                    # print('np all: ', np.all(concheck))
                    if np.all(concheck):
                        numcheck+=len(concheck)
                        # print(f'original bbox{jj}: \n', ogbb[jj])
                        for bb in cbb[jj]:
                            if ax_fix == 0:
                                bin_img[jj, z, ps[0]+bb[0]:ps[0]+bb[1], ps[1]+bb[2]:ps[1]+bb[3]] = 128
                            elif ax_fix == 1:
                                bin_img[jj, ps[0]+bb[0]:ps[0]+bb[1], z, ps[1]+bb[2]:ps[1]+bb[3]] = 128
                            elif ax_fix == 2:
                                bin_img[jj, ps[0]+bb[0]:ps[0]+bb[1], ps[1]+bb[2]:ps[1]+bb[3], z] = 128

                    else:
                        print('np all: ', np.all(concheck))
                        print(f'original bbox{jj}: \n', ogbb[jj])
                        print('converted to original: \n', cbb[jj])
                        print(concheck)
                        print('')

                numcheck_total[jj] += numcheck

            # if ogbb[0].shape[0] > 1 and ogbb[1].shape[0] > 1:
            #     break
        
        # if counts > 0 and len(nbb) != 0:
        #     return
        norm_bbs.append(nbb)
        if len(nbb) != 0:
            patch_has_cell.append(1)
        else:
            patch_has_cell.append(0)


print(num_bboxes, numcheck_total)
print('new vols from all that stuff: ', new_vols)
print('total vols from most basic bbox arrays: ', total_vols)


vols_bin = [0,0]
for i in range(imdims_zyx[0]):
    for c in range(2):
        vols_bin[c] += np.count_nonzero(bin_img[c,i,:,:])

print('vols binary image: ', vols_bin)

imwrite(f'12a_yolo_{ax_fix}_binfrompatch.tif', bin_img)
print('Image saved')
