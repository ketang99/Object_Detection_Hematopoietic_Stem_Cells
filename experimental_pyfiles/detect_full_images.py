'''
Get bounding box predictions for a full image
'''


import os
from PIL import Image
import numpy as np
import sys
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import sys
homedir = '/home/kgupta/data/registration_testing'
os.chdir(homedir)
sys.path.append(homedir)
sys.path.append(f'{homedir}/faster_rcnn')
sys.path.append(f'{homedir}/retinanet')
sys.path.append(f'{homedir}/dsb')

import reg_functions as reg

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
    

parser = argparse.ArgumentParser(description='YOLO dataset generator')
parser.add_argument('-d', '--datadir', default='h5_files')
parser.add_argument('-a', '--ax_fix', type=int, default=0)
parser.add_argument('-m', '--model', type=str, default='faster')
parser.add_argument('-t', '--transform', type=int, default=0)
args = parser.parse_args()

run_model = True

axes = ['Z', 'Y', 'X']
ax_id = args.ax_fix
ax_fix = axes[ax_id]

# yolo_dir = home_dir + '/yolov5_z_small'

if args.model == 'retina':
    model_dir = f'/home/kgupta/data/registration_testing/retinanet/models'
    model_dir = model_dir + f'/retinanet_resnet50_correctbigset_allclasses_{ax_fix}_e100_o0_earlystop1'
elif args.model == 'faster':
    model_dir = f'/home/kgupta/data/registration_testing/faster_rcnn/models'
    model_dir = model_dir + f'/faster_rcnn_50_{ax_fix}_correctbigset_3classes_e50_o0_noroi15_earlystop1'

results_dir = model_dir + '/full_predictions'

if 'full_predictions' not in os.listdir(model_dir):
    os.mkdir(results_dir)
    for ax in axes:
        os.mkdir(f'{results_dir}/{ax}')

save_dir = results_dir
# data_dir = 'yolo_dataset_new_Z_r3'
# data_dir = f'ax_fix.lower()_2d_bigdataset_new_{ax_fix}_r3'
# data_dir = f'2d_bigdataset_new_{ax_fix}_r3'

filename = '12A+6W_ev_Region+1_Merged+filtered+ANNOTATED+CROP.ims'
print('model_dir:', model_dir)

# general flow
s_p = np.array([[192,192],[59,192],[59,192]])

# s_p should be dependent on the fixed axis: if Z then [640,640] else [50,640]

max_bb = np.array([17,50,65])
max_bb = np.delete(max_bb, ax_id)
s_p = s_p[ax_id]
step_size = s_p - max_bb

print('max_bb: ', max_bb)
print('s_p: ', s_p)
print('step_size: ', step_size)

def main():
    
    #return
    if args.transform:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std
        ])
    else:
        transform = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(f'{model_dir}/model.pth')
    model.to(device)
    model.eval()

    combo_img, bbox0, bbox1, metadata = reg.get_image(f'h5_files/{filename}', 3, return_img = True)
    num_channels = combo_img.shape[0]
    imdims_zyx = combo_img.shape[1:]
    imdims = np.delete(imdims_zyx, ax_id)
    assert num_channels == 3
    print('Image has been read from ims file')

    img_tensor = torch.from_numpy(combo_img)
    del combo_img

    img_tensor = img_tensor.to(device)

    print('Original shape of image: ', img_tensor.shape)
    permutation_order = [0, ax_id+1] + [i for i in range(1, 4) if i != ax_id + 1]
    img_tensor = img_tensor.permute(permutation_order)
    img_tensor = img_tensor.unsqueeze(0)
    print(f'For ax_fix={ax_fix}, Permuted shape of image: ', img_tensor.shape)
    print('Single slice shape: ', img_tensor[:,:,0].shape)

    img_tensor = img_tensor / 255.0
    
    print('img and model devices:')
    print(img_tensor.device)
    print(next(model.parameters()).device)
    
    # return
    if run_model:
        starts = []
        for i in range(2):
            starts.append(find_patch_starts(step_size[i], imdims[i], s_p[i]))
        
        # create patch_starts array with all patches
        ps = []
        for s0 in starts[0]:
            for s1 in starts[1]:
                ps.append([s0, s1])
        
        patch_starts = np.asarray(ps)
    # # Now iterate through the second axis of the rearranged tensor
        for i in range(img_tensor.shape[2]):
            slice_along_axis = img_tensor[:,:, i]
            # slice_along_axis.to(device)
            for start in patch_starts:
                if ax_id == 0:
                    locstr = f'{i}_{start[0]}_{start[1]}'
                elif ax_id == 1:
                    locstr = f'{start[0]}_{i}_{start[1]}'
                elif ax_id == 2:
                    locstr = f'{start[0]}_{start[1]}_{i}'
                output_file = f'{save_dir}/{locstr}_predictions.txt'

                patch = slice_along_axis[:, :, start[0]:start[0]+s_p[0], start[1]:start[1]+s_p[1]]
                with torch.no_grad():
                    predictions = model(patch)

                    with open(output_file, 'w') as f:
                        for prediction in predictions:
                            boxes = prediction['boxes'].cpu().numpy()
                            labels = prediction['labels'].cpu().numpy()
                            scores = prediction['scores'].cpu().numpy()
                            
                            for box, label, score in zip(boxes, labels, scores):
                                # Write each prediction to the file
                                if score >= 0.3:
                                    f.write(f'{label} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')

    print('\n********************\nDONE')


if __name__ == '__main__':
    main()
