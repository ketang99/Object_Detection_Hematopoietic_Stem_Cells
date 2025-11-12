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

import sys
homedir = '/home/kgupta/data/registration_testing'
os.chdir(homedir)
filedir = 'h5_files'  # all imaris files are stored here
filenames = ['12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims',
             '2w_5_filtered_DP_HSC+annotated.ims',
             '4w_3A_filtered_DP_HSC+annotated.ims',
             '6w_9A_filtered_DP_HSC+annotated.ims']
filenames = [filenames[0]]

sys.path.append(homedir)
sys.path.append(f'{homedir}/dsb')

import reg_functions as reg

# to be parsed in from terminal: ax_fix, dataset_dir

parser = argparse.ArgumentParser(description='YOLO dataset generator')
parser.add_argument('-d', '--datadir', default='yolo_dataset')
parser.add_argument('-a', '--ax_fix', default=0)
parser.add_argument('-s', '--imsave', default=0)

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
yolo_dir = f'{args.datadir}_{ax_name}'
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

def patch_split(idcs, train_pct=0.8, test_pct=0.1, val_pct=0.1):
    splits = [int(train_pct * len(idcs)), int(test_pct * len(idcs)), int(val_pct * len(idcs))]
    train_cell = idcs[:splits[0]]
    test_cell = idcs[splits[0]:splits[0] + splits[1]]
    val_cell = idcs[splits[0]+splits[1]:splits[0]+splits[1]+splits[2]]

    return train_cell, test_cell, val_cell

def get_patch_indices(cells_present):

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
    np.random.shuffle(no_cell_idx)
    trainno, testno, valno = patch_split(no_cell_idx)

    return [traincell,trainno], [testcell,testno], [valcell,valno]

def convert_bounding_boxes(bboxes):
    # Convert the format [x_center, y_center, x_width, y_width]
    # to [x_begin, x_end, y_begin, y_end]
    converted_bboxes = np.zeros_like(bboxes)
    converted_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_begin
    converted_bboxes[:, 1] = bboxes[:, 0] + bboxes[:, 2] / 2  # x_end
    converted_bboxes[:, 2] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_begin
    converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y_end
    return np.round(converted_bboxes).astype(int)

############################################################################################################
# ###########################################################################################################

# general flow
s_p = np.array([[192,192],[48,192],[48,192]])

# s_p should be dependent on the fixed axis: if Z then [640,640] else [50,640]

max_bb = np.array([17,50,65])
# s_p = [48, 192, 192]
max_bb = np.delete(max_bb, ax_fix)
s_p = s_p[ax_fix]
step_size = s_p - max_bb

# strides = [12, 150, 150]
# stride = strides[ax_fix]
# patches_per_file = 3750
# num_split_set = 1000
# split_pct = [0.8, 0.1, 0.1]  # train/test/val split of dataset

print('max_bb: ', max_bb)
print('s_p: ', s_p)
print('step_size: ', step_size)

def main(imsave=False):
    
    for filenum, filename in enumerate(filenames):
        print('*******')
        print('Getting patches for:')
        print(filename)
        print('')
        # read the metadata and combo_img

        with h5py.File(f'{homedir}/{filedir}/all_bboxes.h5', 'r') as f:
            bbox0 = f['Image 0/bbox0_3'][:]
            bbox1 = f['Image 0/bbox1_3'][:]
            imdims_zyx = f['Image 0/imdims_zyx'][:]
        
       # imdims_zyx = metadata['imdims_zyx']
        print('Image dims full: ', imdims_zyx)
        imdims = np.delete(imdims_zyx, ax_fix)
        print('image dims with axfix removed: ', imdims)
       # imdims = np.delete(imdims_zyx, ax_fix)
        print('step_size: ', step_size)
        print('patch dims s_p: ', s_p)

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

        acceptable_z = range(imdims_zyx[ax_fix])

        # for z in range(metadata['imdims_zyx'][ax_fix]):
        zs = []
        patch_has_cell = []
        norm_bbs = []
        all_patch_starts = []
        print('writing all BBs')
        for z in acceptable_z:
            # get the bounding boxes for that z:
            # print(z, 'getting bboxes for this z')
            bb_lims = retrieve_planar_bboxes(bbox0,bbox1,ax_fix,z)
            # print('got bboxes')

            # if use_all_ps:
            #     selected_patch_starts = patch_starts
            # else:
            #     s_inds = np.random.choice(len(patch_starts), round(12 * num_split_set / len(acceptable_z)))
            #     selected_patch_starts = patch_starts[s_inds]
            #     # print('len selected patch starts: ', len(selected_patch_starts))
            #     # return

            selected_patch_starts = patch_starts
            
            for counts, start in enumerate(selected_patch_starts):
                zs.append(z)
                all_patch_starts.append(start)
                # get normalized bounding boxes
                nbb = get_normalized_bboxes(start, s_p, bb_lims, ax_fix)
                nbb = np.array(nbb)
                # print('nbb shape: ', nbb.shape)
                # print('nbb: ', nbb)
                if len(nbb) != 0:
                    nbb[:,[1,3]] *= s_p[0]
                    nbb[:,[2,4]] *= s_p[1]
                # print('nbb denormalized: ', nbb)
                # if counts > 0 and len(nbb) != 0:
                #     return
                norm_bbs.append(nbb)
                if len(nbb) != 0:
                    patch_has_cell.append(1)
                else:
                    patch_has_cell.append(0)

        
        # del patch_starts
        

        print('number of z values: ', len(acceptable_z))
        print('number of patch starts per z: ', len(patch_starts))
        print('number of all_patch_starts: ', len(all_patch_starts), len(zs))
        print('z values * patch starts per z: ', len(acceptable_z)*len(patch_starts))
        print('writing to bim_img')
        bin_img = np.zeros((2,imdims_zyx[0],imdims_zyx[1],imdims_zyx[2])).astype(np.uint8)
        print('bin_img shape: ', bin_img.shape)

        for i, ps in enumerate(all_patch_starts):
            nbb = norm_bbs[i]
            print('ps: ', ps)
            print('nbb: ', nbb, nbb.shape)
            print('bb0 check: ', nbb[:,0]==0)
            print('bb1 check: ', nbb[:,0]==1)
            #return
            if len(nbb) != 0:
                nbb_ = [nbb[nbb[:,0]==0], nbb[nbb[:,0]==1]]  # contains bbox0 and bbox1 in each entry
            # nbb1_ = nbb[nbb[:,0]==1]
                bbb = []
                for j, bbs in enumerate(nbb_):
                    if len(bbs) != 0:
                    # need to get the bounds in a form that can be used to directly write to bin_img
                        bbb.append(convert_bounding_boxes(bbs[:,1:]))
                    else:
                        bbb.append(np.array([]))

            else:
                bbb = [[],[]]

            #print('bbb: ', bbb)


           if len(bbb[0]) != 0 or len(bbb[1]) != 0:
               return

            for j, bbs in enumerate(bbb):
                if len(bbs) != 0:
                    for bb in bbs:
                        if ax_fix == 0:
                            bin_img[j, zs[i], ps[0]+bb[0]:ps[0]+bb[1], ps[1]+bb[2]:ps[1]+bb[3]] = 128
                        elif ax_fix == 1:
                            bin_img[j, ps[0]+bb[0]:ps[0]+bb[1], zs[i], ps[1]+bb[2]:ps[1]+bb[3]] = 128
                        elif ax_fix == 2:
                            bin_img[j, ps[0]+bb[0]:ps[0]+bb[1], ps[1]+bb[2]:ps[1]+bb[3], zs[i]] = 128

        # save the binary image as a tif file
        imwrite(f'12a_yolo_{ax_fix}_binfrompatch.tif', bin_img)
            
    
    print('END')

if __name__ == '__main__':
    main(imsave)
