import numpy as np
import h5py
import os
import math
import pandas as pd
import tifffile
import csv
import json
import reg_functions as reg

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

# nn contains all the loss functions, as well as Module to create a CNN or FCNN

def normalize_bboxes(bboxes, dims_downsampled, downsampling=4.0):
    '''
    Normalize bbox centers and side lengths to [0,1] in a downsampled grid space

    bboxes: [Nx6] array. First three cols are centers and last three cols are side lengths. Order ZYX
    downsampling: the factor by which the input layer (initial patch) is downsampled in the neural net's output
    '''

    down_bboxes = bboxes / downsampling   # downsamples the bboxes
    normalized_bboxes = down_bboxes / dims_downsampled

    return normalized_bboxes

def generate_anchor_coordinates_3d(base_size, scales, aspect_ratios, grid_size):

    '''
    base size: base size of an anchor box. 3 entries for side lengths. Order ZXY
    scales: scaling factor of base_size
    aspect_ratios: predefine this. Find longest side of BBs. If X and Y equal, then select either as length. Assume X for now
    grid_size: size of the downsampled image space

    Try to store each anchor point as [Az, Ay, Ax, rz, ry, rx]

    Note: if you wanna include scales, then multiply lz,ly,lx by the scale
    '''
    
    coordinates_3d = []
    for z in range(grid_size):
        for y in range(grid_size):
            for x in range(grid_size):
                for scale in scales:
                    for aspect_ratio in aspect_ratios:
                        # Calculate dimensions
                        lz = base_size[0] * np.sqrt(aspect_ratio)   # Z
                        ly = base_size[1] * np.sqrt(aspect_ratio)  # Y
                        lx = base_size[2] / np.sqrt(aspect_ratio)  # X
                        
                        # Calculate normalized coordinates
                        center_x = x / grid_size
                        center_y = y / grid_size
                        center_z = z / grid_size
                        
                        coordinates_3d.append([center_z, center_y, center_x, lz, ly, lx])
    return np.array(coordinates_3d)

class DictBoneMarrowCells(Dataset):

    def __init__(self, datafile, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datafile = datafile
        self.root_dir = root_dir
        self.fileloc = os.path.join(self.root_dir, self.datafile)
        self.transform = transform

    def __len__(self):

        with h5py.File(self.fileloc, 'r') as f:
            return(len(list(f['Patches'].keys())))

    def __getitem__(self, idx):
        '''
        nzhw: right now it gives the patch dimensions. 
              Have it contain the number of bboxes n and their respective z,y,x lengths in vx
        bboxes: return these in the form [zc, yz, xc] where c means center

        Returns:
        patch
        bboxes
        bbox_size : size along Z,Y,X
        celltypes
        '''
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with h5py.File(self.fileloc, 'r') as f:

            # nzhw = np.zeros((4))
            # nzhw[1:] = f['Metadata']['patch_size'][()]
            # nzhw[0] = 1
            
            patchdict = {}
            pg = f[f'Patches/Patch {idx+1}']
            # print(pg['Patch'][()].shape)
            pi = pg['Patch info']

            # no need for patchdict here, if you need it put it in __init__()
            # for key in list(pi.keys()):
            #     # print(key)
            #     value = pi[key][()] 
            #     # print(type(value))
            #     if isinstance(value, np.ndarray):
            #         if len(value) != 0:
            #             print(len(value))
            #             if isinstance(value[0], np.bytes_):
            #                 if len(value) > 1:
            #                     strarr = []
            #                     for i in range(len(value)):
            #                         strarr.append(value[i].tobytes().decode('ascii', 'decode'))
            #                     patchdict[key] = strarr
            #                 else:
            #                     patchdict[key] = value[0].tobytes().decode('ascii', 'decode')
            #             else:
            #                 patchdict[key] = value
            #         else:
            #             patchdict[key] = value
                    
            #     else:
            #         patchdict[key] = value
                # print('\n')

        # self.patchdict = patchdict
        
            patch = pg['Patch'][()]
            bbtypes = pi['Cell_type'][()]
            bbox = pi['bbox'][()]
            ploc = pi['Start_pos'][()]
            filename = pi['Filename'][0].tobytes().decode('ascii', 'decode')
    
        bboxes = np.zeros((bbox.shape[0], 3))
        bbox_size = np.zeros((bbox.shape[0], 3))
        for i in range(3):
            bboxes[:,i] = 0.5 * (bbox[:,2*i] + bbox[:, 2*i+1])
            bbox_size[:,i] = bbox[:,2*i+1] - bbox[:, 2*i]

        # print(bboxes.shape)
            # nzhw = np.zeros((bbox.shape[0], 4))
            # # nzhw[]
            
            # for i in range(3):
            #     bboxes[:,i] = 0.5 * (bbox[:,2*i] + bbox[:,2*i+1])
            #     nzhw
        

        if self.transform:
            sample = self.transform(sample)

        return {'patch': torch.from_numpy(patch.astype(np.uint8)), 
                'bboxes': bboxes, 
                'bbox_size': bbox_size, 
                'cell_type': bbtypes, 
                'start_pos': ploc, 
                'filename': filename}

class HLFBoneMarrowCells(Dataset):

    def __init__(self, datafile, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datafile = datafile
        self.root_dir = root_dir
        self.fileloc = os.path.join(self.root_dir, self.datafile)
        # self.label_mapping = Label
        self.transform = transform

    def __len__(self):

        with h5py.File(self.fileloc, 'r') as f:
            return(len(list(f['Patches'].keys())))

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

        with h5py.File(self.fileloc, 'r') as f:
            
            patchdict = {}
            pg = f[f'Patches/Patch {idx+1}']
            # print(pg['Patch'][()].shape)
            pi = pg['Patch info']
        
            patch = pg['Patch'][()]
            # bbtypes = pi['Cell_type'][()]
            bbox = pi['bbox'][()]
            ploc = pi['Start_pos'][()]
            filename = pi['Filename'][0].tobytes().decode('ascii', 'decode')
    
        bboxes = np.zeros((bbox.shape[0], 4))
        # bbox_size = np.zeros((bbox.shape[0], 3))
        for i in range(3):
            bboxes[:,i] = 0.5 * (bbox[:,2*i] + bbox[:, 2*i+1])
            # for now let's do just one dimension's size: make it X
            
        bboxes[:,-1] = bbox[:,-1] - bbox[:,-2]

        if self.transform:
            sample = self.transform(sample)

        return {'patch': torch.from_numpy(patch.astype(np.uint8)).unsqueeze(0), 
                'bboxes': bboxes, 
                'start_pos': ploc, 
                'filename': filename}