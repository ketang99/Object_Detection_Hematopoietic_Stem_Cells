'''
Functions to create the crops for NoduleNet training
Functinos also generate the ground truth bounding boxes and class labels
'''

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
import time
import collections
import random
import json
import reg_functions as reg
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from split_combine import SplitComb

class HLFBoneMarrowCells(Dataset):

    def __init__(self, datafile, root_dir, config, phase='Train', transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        We will use just one file to train
        Retrieve bbox0, bbox1 as part of the init
        Save the good patches in a separate .npy file and read from here
        """
        self.datafile = datafile
        self.root_dir = 'h5_files'
        self.fileloc = os.path.join(self.root_dir, self.datafile)
        self.phase = phase
        self.label_mapping = LabelMappingAll(config, phase=self.phase)
        self.transform = transform
        # self.max_bb = np.array([23,67,87])
        self.max_bb = np.array([18,52,66])
        self.s_p = [48, 192, 192]
        self.step_size = self.s_p - self.max_bb

        # need to get self.bbox0 and 1, self.combo_img, self.s_p, self.max_bb
        self.combo_img, self.bbox0, self.bbox1, self.metadata = reg.get_image(f'{self.fileloc}', 
                                                                              r_scaling=3, 
                                                                              desired_cnames = ['HLF tdT'],
                                                                              return_img=True)

        print('bbox1 shape: ', self.bbox1.shape)
        # read numpy array with the patch starts: these are selected such that they all have >1 bbox
        self.all_patch_starts = get_zyx_patch_starts(self.s_p, self.step_size, self.metadata['imdims_zyx'])

        # put code here that selects only patch starts that have >=1 cells of type HSPC
        # maybe don't do all that processing here. Prepare the patch starts beforehand and load them directly

        self.good_patches = [[0,0,5084],[0,61,5330],[0, 427, 3936]]
        if self.phase == 'Train':
            self.patch_starts = [self.good_patches[0], self.good_patches[0]]
        elif self.phase == 'Val':
            self.patch_starts = [self.good_patches[2], self.good_patches[2]]

    def __len__(self):

        return len(self.patch_starts)

    def __getitem__(self, idx):
        '''
        nzhw: right now it gives the patch dimensions. 
              Have it contain the number of bboxes n and their respective z,y,x lengths in vx
        bboxes: return these in the form [zc, yz, xc] where c means center

        Returns:
        patch
        bboxes : first three cols are centers for ZYX. Last column is the size along X
        bbox_size : size along Z,Y,X
        celltypes : 
        '''
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ps = self.patch_starts[idx]
        patchdict = {}
        if len(self.combo_img.shape) == 4:
            patch = self.combo_img[-1, ps[0]:ps[0]+self.s_p[0],
                                    ps[1]:ps[1]+self.s_p[1],
                                    ps[2]:ps[2]+self.s_p[2]]
        else:
            patch = self.combo_img[ps[0]:ps[0]+self.s_p[0],
                                    ps[1]:ps[1]+self.s_p[1],
                                    ps[2]:ps[2]+self.s_p[2]]

        # print('patch start: ', ps)
        pbb = patch_bbox(ps, self.s_p, self.bbox1)
        # print('unselected bbox: ', pbb)
        bbox = select_bboxes(pbb, self.max_bb)
        # print('uncentered selected bbox: ', bbox)
        ploc = ps
        # filename = pi['Filename'][0].tobytes().decode('ascii', 'decode')

        # patch, nzhw = self.split_comber.split(patch)
        if len(bbox) == 0:
            bboxes = np.array([])
        else:
            bboxes = np.zeros((bbox.shape[0], 6))
            # bbox_size = np.zeros((bbox.shape[0], 3))
            for i in range(3):
                bboxes[:,i] = 0.5 * (bbox[:,2*i] + bbox[:, 2*i+1])
                bboxes[:,i+3] = bbox[:,2*i+1] - bbox[:,2*i]
                
            bboxes[:,-1] = bbox[:,-1] - bbox[:,-2]
        
        # print('patch id: ', self.pids[idx])
        # print('bboxes: ', bboxes)
        # labels = []
        # for i in range(len(bboxes)):
        #     target = bboxes[i]
        #     label = self.label_mapping(patch.shape[-3:], target, bboxes)
        #     labels.append(label)
        #     del label

        pos,label = self.label_mapping(patch.shape[-3:], bboxes)

        if self.transform:
            sample = self.transform(sample)

        # print('Patch shape: ', patch.shape)
        # print('before norm min and max of patch: ', np.min(patch), np.max(patch))
        patch = (patch-128.0)/128.0
        # print('min and max of patch: ', np.min(patch), np.max(patch))

        return {'patch': torch.from_numpy(patch).to(torch.float), 
                'bboxes': bboxes,
                'label': torch.from_numpy(label),
                'start_pos': ploc, 
                'patch_idx': idx}

def custom_collate_dict(batch):
    collated_batch = {}
    for key in batch[0].keys():
        # print(key)
        if key == 'bboxes' or key == 'start_pos' or key == 'patch_idx':
            # Handle variable number of bounding boxes
            collated_batch[key] = [sample[key] for sample in batch]
        elif key == 'label' or key =='patch':
            collated_batch[key] = torch.stack([sample[key] for sample in batch])
    
    return collated_batch


def custom_collate_dict_new(batch):
    collated_batch = {}
    batch_size = len(batch)  # Get the actual batch size
    # print('loader collate batch size: ', batch_size)
    # Handle keys with variable-length lists
    variable_length_keys = ['bboxes', 'start_pos', 'patch_idx']
    for key in variable_length_keys:
        collated_batch[key] = [sample[key] for sample in batch]

    # Handle keys with fixed-size tensors (e.g., 'label' and 'patch')
    fixed_size_keys = ['label', 'patch']
    for key in fixed_size_keys:
        collated_batch[key] = torch.stack([sample[key] for sample in batch])

    # Ensure that the batch size is consistent across all keys
    assert all(len(collated_batch[key]) == batch_size for key in collated_batch.keys()), "Inconsistent batch size"

    return collated_batch

def get_zyx_patch_starts(s_p, step_size, imdims):
    # patch starts for all images
    patch_starts = []
    
    zs = reg.find_patch_starts(step_size[0], imdims[0], s_p[0])
    ys = reg.find_patch_starts(step_size[1], imdims[1], s_p[1])
    xs = reg.find_patch_starts(step_size[2], imdims[2], s_p[2])

    starts_i = []
    for z in zs:
        for y in ys:
            for x in xs:
                starts_i.append([z,y,x])

    starts_i = np.array(starts_i)

    return starts_i

def patch_bbox(current_patch, s_p, bbox):

    # print('current patch: ', current_patch)
    if len(bbox) != 0:
        bb_shifted = np.copy(bbox)
        bb_shifted[:,[0,1]] -= current_patch[0]
        bb_shifted[:,[2,3]] -= current_patch[1]
        bb_shifted[:,[4,5]] -= current_patch[2]
        # print('BB shifted shape: ',bb_shifted.shape)
        # print(bb_shifted[:5,:])

        valid_bool = []
        # either an upper or a lower bound must lie between 0 and s_p
        for i in range(3):
            ax_bb = bb_shifted[:,[2*i,2*i+1]]
            # low_valid = np.a() 
            low_valid = np.abs(ax_bb[:,0] - s_p[i]/2) <= s_p[i]/2
            up_valid = np.all((ax_bb[:,1] > 0, ax_bb[:,1] <= s_p[i]), axis=0)
            both_valids = np.stack((low_valid, up_valid), axis=0)
            # print(both_valids.shape)
            valid_bool.append(np.any(both_valids, axis=0))

        # valid_all = np.all()
        valid_all = np.stack((valid_bool[0],valid_bool[1],valid_bool[2]), axis=0)
        # print(valid_all.shape)
        bb_shifted = bb_shifted[np.all(valid_all, axis=0)]


        if len(bb_shifted) != 0:
            # print('current patch: ', current_patch) 
            # print('bb shifted shape after extracting valid bbs: ', bb_shifted.shape)
            # print(bb_shifted)
            bb_out = np.zeros(bb_shifted.shape)
            for i in range(3):
                bb = bb_shifted[:,[2*i,2*i+1]]

                bb[:,0][bb[:,0] < 0] = 0   # lower bounds
                bb[:,1][bb[:,1] > s_p[i]] = s_p[i]   # upper bounds
                bb_out[:,[2*i,2*i+1]] = bb
                # print('')
            return bb_out.astype(int)        
        else:
            return np.array([])

def select_bboxes(bboxes, max_bb):
    max_vol = np.prod(max_bb)
    s_bb = []
    for bb in bboxes:
        vol = 1
        for i in range(3):
            vol *= (bb[2*i+1] - bb[2*i])
        # print('max vol, vol:')
        # print(max_vol, vol)
        if vol > 0.25 * max_vol:
            s_bb.append(bb)

    return np.array(s_bb)

class LabelMappingAll(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'Train':
            self.th_pos = config['th_pos_train']
        elif phase == 'Val':
            self.th_pos = config['th_pos_val']
            
    def __call__(self, input_size, bboxes):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos
        struct = generate_binary_structure(3, 1)      
        
        output_size = []
        for i in range(3):
            assert(input_size[i] % stride == 0)
            output_size.append(input_size[i] / stride)

        output_size = [int(x) for x in output_size]
        
        label = np.zeros(output_size + [len(anchors), 7], np.float32)
        offset = (stride.astype('float') - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 1
                label[:, :, :, i, 0] = binary_dilation(label[:, :, :, i, 0].astype('bool'), structure=struct, iterations=1).astype('float32')
        
        label = label - 1

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        # print('labels shape: ', label.shape)
        if len(bboxes) == 0:
            poss = []
        else:
            poss = []
            for i in range(len(bboxes)):
                target = bboxes[i]
                pos = self.find_pos(target, bboxes, oz, oh, ow)
                poss.append(pos)
                # print('target: ', target)
                # print('pos: ', pos)
                
                dz = (target[0] - oz[pos[0]]) / anchors[pos[3]][0]
                dh = (target[1] - oh[pos[1]]) / anchors[pos[3]][1]
                dw = (target[2] - ow[pos[2]]) / anchors[pos[3]][2]
                drz = np.log(target[3] / anchors[pos[3]][0])
                drh = np.log(target[4] / anchors[pos[3]][1])
                drw = np.log(target[5] / anchors[pos[3]][2])
                label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, drz, drh, drw]
                
        return poss, label
    
    def find_pos(self, target, bboxes, oz, oh, ow):
        stride = self.stride
        th_pos = self.th_pos
        anchors = self.anchors
        offset = (stride.astype('float') - 1) / 2
        
        if np.isnan(target[0]) or len(target) == 0:
            return []
            
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True 
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.sum(np.abs(np.log(target[3:6] / anchors)), axis=1))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
    
        return pos

def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, lz, lh, lw = bbox
    max_overlap_z = min(lz, anchor[0])
    max_overlap_h = min(lh, anchor[1])
    max_overlap_w = min(lw, anchor[2])
    
    min_overlap = np.power(max(lz * lh * lw, anchor[0] * anchor[1] * anchor[2]), 1/3) * th / np.min([max_overlap_z, max_overlap_h, max_overlap_w])
    
    if min_overlap > np.min([max_overlap_z, max_overlap_h, max_overlap_w]):
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(lz - anchor[0]) - (max_overlap_z - min_overlap)
        e = z + 0.5 * np.abs(lz - anchor[0]) + (max_overlap_z - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]
        
        s = h - 0.5 * np.abs(lh - anchor[1]) - (max_overlap_h - min_overlap)
        e = h + 0.5 * np.abs(lh - anchor[1]) + (max_overlap_h - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]
            
        s = w - 0.5 * np.abs(lw - anchor[2]) - (max_overlap_w - min_overlap)
        e = w + 0.5 * np.abs(lw - anchor[2]) + (max_overlap_w - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        
        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)
        
        r0 = np.array(anchor) / 2
        s0 = centers - r0
        e0 = centers + r0
        
        r1 = np.array([lz, lh, lw]) / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))
        
        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
        
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor[0] * anchor[1] * anchor[2] + lz * lh * lw - intersection

        iou = intersection / union

        mask = iou >= th
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw

class GetPBB(object):
    def __init__(self, config):
        self.stride = np.asarray(config['stride'])
        self.anchors = config['anchors']

    def __call__(self, output, thresh=-3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (stride - 1) / 2
        
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors[:, 0].reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors[:, 1].reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors[:, 2].reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors[:, 0].reshape((1, 1, 1, -1))
        output[:, :, :, :, 5] = np.exp(output[:, :, :, :, 5]) * anchors[:, 1].reshape((1, 1, 1, -1))
        output[:, :, :, :, 6] = np.exp(output[:, :, :, :, 6]) * anchors[:, 2].reshape((1, 1, 1, -1))
        
        mask = output[..., 0] > thresh
        xx, yy, zz, aa = np.where(mask)
        
        output = output[xx, yy, zz, aa]
        if ismask:
            return output, [xx, yy, zz, aa]
        else:
            return output