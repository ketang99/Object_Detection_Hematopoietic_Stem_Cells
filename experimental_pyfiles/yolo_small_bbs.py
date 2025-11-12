import numpy as np
import h5py
import os
import math
import pandas as pd
import csv

home_dir = '/data/kgupta/registration_testing'
# os.chdir(home_dir)
data_dir = home_dir + '/yolo_datasets/yolo_dataset_Y'
os.chdir(data_dir)

if 'labels' in os.listdir():
    os.rename('labels', 'labels_old')

print(os.listdir())
if 'labels' not in os.listdir():
    os.mkdir('labels')

print('after renaming and making new folder: ', os.listdir())

labelnames = os.listdir('labels_old')

def get_bbox_from_txt(filename, small_bb=False):
    # print(dir)
    with open(f'labels_old/{filename}', 'r') as f:
        f.seek(0)
        bb_raw = f.readlines()

    if len(bb_raw) != 0:
        all_bbs = convert_all_lines(bb_raw)
        if small_bb:
            for single_bb in all_bbs:
                single_bb[-1] /= 3
                single_bb[-2] /= 3
        return all_bbs

    else:
        return []

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

for ln in labelnames:
    # get bb and divide the bbox size by 3
    # bb = get_bbox_from_txt(ln, small_bb=True)
    with open(f'labels/{ln}', 'w') as f:
        for row in get_bbox_from_txt(ln, small_bb = True):
            f.write(f"{int(row[0])} {' '.join(map(str, row[1:]))}\n")

print('new labels written')
new_labelnames = os.listdir('labels')
print(len(new_labelnames), len(labelnames))