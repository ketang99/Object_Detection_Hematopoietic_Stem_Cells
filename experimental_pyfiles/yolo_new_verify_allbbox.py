import numpy as np
import csv
import pandas as pd

import sys
import os
home_dir = '/data/kgupta/registration_testing'
os.chdir(home_dir)
yolo_dir = home_dir + '/yolov5_z_small'
results_dir = yolo_dir + '/runs/detect/exp11'
data_dir = 'yolo_dataset_new_Z_r3'

sys.path.append(yolo_dir)

# write some functions to streamline the process

def get_bbox_from_txt(image_id, data_dir, results_dir, gt=True):
    filename = f'img_{image_id}.txt'
    if gt:
        dir = f'{data_dir}/labels'
    else:
        dir = f'{results_dir}/labels'
    # print(dir)
    with open(f'{dir}/{filename}', 'r') as f:
        f.seek(0)
        bb_raw = f.readlines()

    return convert_all_lines(bb_raw)

def convert_single_line_to_bb(single_line):
    single_bb = []
    lsplit = single_line.split()
    # print(lsplit)
    for v in lsplit:
        if len(v) == 1:  # the class
            single_bb.append(int(v))
        else:  # the bb values
            single_bb.append(float(v))

    return single_bb

def convert_all_lines(bb_lines):
    bboxes = []
    for (i,line) in enumerate(bb_lines):
        # print(line, type(line))
        sbb = convert_single_line_to_bb(line)
        # print(sbb)
        bboxes.append(sbb)

    return bboxes

def get_class_bboxes(bboxes, celltype):
    # print(len(bboxes))
    # print('np where')
    # print(np.where(np.array(bboxes)[:,0]==celltype))
    if len(bboxes) == 0:
        return np.array(bboxes)
    else:
        class_inds = np.where(np.array(bboxes)[:,0]==celltype)[0]
        return np.array(bboxes)[class_inds, :]


'''
Steps:
1. Get the number of images from the images directory in the dataset
2. Get the number of label files: should be equal to num_images
3. For each image name, read the label file; each file should contain at least one bbox
'''

# get the image IDs
imgnames = os.listdir(f'{data_dir}/images')
labelnames = os.listdir(f'{data_dir}/labels')
# with open(f'{data_dir}/full_Test.txt', 'r') as f:
#     f.seek(0)
#     gt_test_names = f.readlines()
# gt_test_ids = []
# for sname in gt_test_names:
#     gt_test_ids.append(int(sname.split('/')[-1].split('.')[0].split('_')[-1]))

print('Number of images and label files, are they equal?')
print(len(imgnames), len(labelnames), len(imgnames)==len(labelnames))

# get image ID from imgnames
img_ids = []
for i,imgfile in enumerate(imgnames):
    # if i > 0:
    #     break
    # print(type(imgfile), imgfile)
    # print(imgfile[:-4])
    # print(imgfile[:-4].split('_'))
    img_ids.append(int(imgfile[:-4].split('_')[-1]))

print('')
print(img_ids[:10])
print('\nImage ID ints obtained')


num_images = len(imgnames)
num_wbbox = 0
id_no_bbox = []

for i,img_id in enumerate(img_ids):
    # labelf = f'{data_dir}/labels/{labelfile}'
    bbox = get_bbox_from_txt(img_id, data_dir, results_dir, gt=True)
    if len(bbox) != 0:
        num_wbbox += 1
    else:
        id_no_bbox.append(img_id)

print(num_wbbox)
print(num_wbbox == num_images)
