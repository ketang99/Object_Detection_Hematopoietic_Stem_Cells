import os

homedir = '/home/kgupta/data/registration_testing'
os.chdir(homedir)

import numpy as np
import reg_functions as reg
import h5py
import argparse

parser = argparse.ArgumentParser(description='YOLO dataset generator')
parser.add_argument('-d', '--datadir', default='yolo_dataset')
parser.add_argument('-a', '--ax_fix', default=0)

args = parser.parse_args()
# global args

ax_fix = int(args.ax_fix)
print('ax_fix:')
print(ax_fix)

filedir = 'h5_files'  # all imaris files are stored here
filenames = ['12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims',
             '2w_5_filtered_DP_HSC+annotated.ims',
             '4w_3A_filtered_DP_HSC+annotated.ims',
             '6w_9A_filtered_DP_HSC+annotated.ims']

# print(os.get_cwd())
############################################################################################################
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
    if num_nc < len(no_cell_idx):
        rc = np.random.choice(len(no_cell_idx), num_nc, replace=False).astype(int)
    else:
        rc = np.random.choice(len(no_cell_idx), size=len(no_cell_idx), replace=False)
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

############################################################################################################

# general flow
s_p = np.array([640,640])
# s_p should be dependent on the fixed axis: if Z then [640,640] else [50,640]

slices = [20, 3000, 5000]
max_bb = np.array([23,67,87])
max_bb = np.delete(max_bb, ax_fix)
step_size = s_p - max_bb
total_patches = 0
all_pat = []

for filename in filenames:
    print(filename)
    combo_img, bbox0, bbox1, metadata = reg.get_image(f'{filedir}/{filename}', return_img = False)
    print('')
    print('imdimx_zyx: ', metadata['imdims_zyx'])
    print('bbox0 len: ', len(bbox0))
    # print(bbox0[:10])
    print('bbox1 len: ', len(bbox1))
    # print(bbox1[:10])

    imdims = np.delete(metadata['imdims_zyx'], ax_fix)
        
    starts = []
    for i in range(2):
        starts.append(find_patch_starts(step_size[i], imdims[i], s_p[i]))
        
    # create patch_starts array with all patches
    ps = []
    for s0 in starts[0]:
        for s1 in starts[1]:
            ps.append([s0, s1])
    
    patch_starts = np.asarray(ps)
    # total_patches+=len(patch_starts) * metadata['imdims_zyx'][ax_fix]
    all_pat.append(len(patch_starts) * metadata['imdims_zyx'][ax_fix])
    
    # imdims = metadata['imdims_zyx']
    # imdims = np.delete(imdims, ax_fix)
    
    zs = []
    patch_has_cell = []
    norm_bbs = []
    all_patch_starts = []

    # instead of entering the below for loop that iterates over all Zs, we just choose a single z=20
    z = slices[ax_fix]
    # get the bounding boxes for that z:
    bb_lims = retrieve_planar_bboxes(bbox0,bbox1,ax_fix,z)
    print('bb_lims dims')
    # print(len(bb_lims[0]), len(bb_lims[0]))
    # print(bb_lims)

    for start in patch_starts:
        zs.append(z)
        all_patch_starts.append(start)
        # get normalized bounding boxes
        nbb = get_normalized_bboxes(start, s_p, bb_lims, ax_fix)
        norm_bbs.append(nbb)
        if len(nbb) != 0:
            patch_has_cell.append(1)
        else:
            patch_has_cell.append(0)

    print('has cell len and nonzeros')
    print(len(patch_has_cell))
    print(np.count_nonzero(patch_has_cell))

    train_idx, test_idx, val_idx = get_patch_indices(patch_has_cell)

    train_idcs = np.concatenate((train_idx[0], train_idx[1]))
    test_idcs = np.concatenate((test_idx[0], test_idx[1]))
    val_idcs = np.concatenate((val_idx[0], val_idx[1]))
    del train_idx, test_idx, val_idx

    print('train test val idcs shape')
    print(train_idcs.shape)
    print(test_idcs.shape)
    print(val_idcs.shape)

    total_patches+= (len(train_idcs)+len(test_idcs)+len(val_idcs))
    
    save_idcs = [train_idcs, test_idcs, val_idcs]
    phases = ['Train', 'Test', 'Val']

    print('')
    print('****')

print('Total number of patches')
print(total_patches)
print(all_pat)