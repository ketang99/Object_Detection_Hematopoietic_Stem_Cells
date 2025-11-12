'''
Script to find the precision, recall F1-score and accuracy from 2D object detection
- Can be used for yolo, retinanet and faster-rcnn
'''

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
phase = 'Val'

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

def bbox_to_corners(y_center, x_center, y_width, x_width):
    """Convert bounding box from center format to corner format."""
    y1 = y_center - y_width / 2
    y2 = y_center + y_width / 2
    x1 = x_center - x_width / 2
    x2 = x_center + x_width / 2
    return y1, x1, y2, x2

def intersection_over_union(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # Convert from [y_center, x_center, y_width, x_width] to [y1, x1, y2, x2]
    y1_1, x1_1, y2_1, x2_1 = bbox_to_corners(*box1)
    y1_2, x1_2, y2_2, x2_2 = bbox_to_corners(*box2)

    # print('corner gt box: ')
    # print(y1_1, x1_1, y2_1, x2_1)
    # print('corner res box:')
    # print(y1_2, x1_2, y2_2, x2_2)
    
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

# function that (for one image) will return the ious of gt and result bboxes
def get_ious_single_image(gt_bboxes, res_bboxes):
    ious = []
    # i will go from 0 to 1, corresponding to each cell type
    # print(len(gt_bboxes), len(res_bboxes))
    for i in range(len(gt_bboxes)):
        curr_gt = gt_bboxes[i, 1:]
        curr_iou = []
        for j in range(len(res_bboxes)):
            curr_res = res_bboxes[j, 1:-1]
            curr_iou.append(intersection_over_union(curr_gt, curr_res))
    
        ious.append(curr_iou) 

    # print(gt_bboxes[0])
    # print(res_bboxes[0])
    # for ctype in range(2):
    #     print('')
    #     print(ctype)
    #     curr_gt = gt_bboxes[ctype]
    #     curr_res = res_bboxes[ctype]
    #     if len(curr_gt) == 0 or len(curr_res) == 0:
    #         ious.append([])
    #     else:
    #         curr_iou = []
    #         for i in range(len(curr_gt)):
    #             for j in range(len(curr_res)):
    #                 curr_iou.append(intersection_over_union(curr_gt[i], curr_res[j]))

    #         ious.append(curr_iou)

    # print('iou: ', ious)
    
    return np.array(ious)

def evaluate_bboxes_pos(gt_bboxes, res_bboxes, iou_th=0.45):

    num_gt = len(gt_bboxes)
    num_res = len(res_bboxes)

    tp = 0
    fp = 0
    fn = 0
    # false negative case
    # if num_gt > 0 and num_res == 0:
    #     fn += num_gt

    # false positive when no ground truth bboxes exist
    if num_res > 0 and num_gt == 0:
        fp += num_res
    # positive cases
    # true positives:
    # print('gt_bboxes: ,', gt_bboxes)
    # print('res_bboxes: ', res_bboxes)
    ious = get_ious_single_image(gt_bboxes, res_bboxes)
    # print(ious.shape)
    print('\nious:')
    print(ious)
    # print(type(ious))
    comp_iou = ious > iou_th
    # print(comp_iou)
    if len(comp_iou) != 0:
        # TP and FN looks along the gt
        pos_iou = np.any(comp_iou, axis=1)
        # print(pos_iou)
        tp += len(np.where(pos_iou)[0])
        # fp += len(np.where(pos_iou==False)[0])
        fn += len(np.where(pos_iou==False)[0])

        # false positive case looks along the results
        pos_iou_res = np.any(comp_iou, axis=0)
        # print('')
        # print(pos_iou_res)
        fp += len(np.where(pos_iou_res==False)[0])

    return tp, fp, fn

'''
pipeline:

get list of all the images in detect results' labels
iterate through this list and get the image names
iterate through the image names
    iterate through the cell types
    get the ground truth bboxes
    get the result bboxes

    using iou find: TP, TN, FP, FN
    iterate through gt bboxes
        get iou with each of the result bboxes
        select highest iou

definitions of TP, TN, FP, FN:
set a threshold of minimum IOU required for a bbox to be considered a positive

TP: iou of gt with a result bbox > threshold

TN: zero bboxes of some type in the results and gt

FP: len(gt) == 0 but len(res) != 0. results have bb that gt does not

FN: len(gt) > 0 but len(res) == 0. bb has gt but results do not
    the number of FNs = len(gt) - len(res)

OVERALL FLOW:

get image names from results' labels: convert the image id to an int  -  int(tfile.split('.')[0].split('_')[-1])
get image names from full_Test.txt in the data_dir: convert ids to ints  -  int(sname.split('/')[-1].split('.')[0].split('_')[-1])
    this will return all the test images including those which have 0 gt bboxes
    
generate some arrays:
    both: contains image IDs that are present in both the gt test set and the results set. TP and FP.
    in_gt: image IDs that are only in the gt test set. FN.
    in_res: ... only in the results set. FP.
'''

all_im_results = os.listdir(f'{results_dir}/labels')
res_ids = []
for tfile in all_im_results:
    res_ids.append(int(tfile.split('.')[0].split('_')[-1]))   
del all_im_results

with open(f'{data_dir}/full_{phase}.txt', 'r') as f:
    f.seek(0)
    gt_test_names = f.readlines()
gt_test_ids = []
for sname in gt_test_names:
    gt_test_ids.append(int(sname.split('/')[-1].split('.')[0].split('_')[-1]))

print(len(res_ids), len(gt_test_ids))

tp = [0,0]
fp = [0,0]
tn = [0,0]
fn = [0,0]

actual_bbs = [0,0]

# to calculate precision and recall, you don't really need true negatives

for gt_val in gt_test_ids:
    # check if the gt_val has gt bboxes
    if os.path.isfile(f'{data_dir}/labels/img_{gt_val}.txt'):
        is_gt_bbox = True
        gt_bb = get_bbox_from_txt(gt_val, data_dir, results_dir, gt=True)
        gt_bboxes = []
        for ctype in [0,1]:
            bbs = get_class_bboxes(gt_bb, ctype)
            # print(gt_val, len(bbs), bbs)
            gt_bboxes.append(bbs)
            actual_bbs[ctype] += len(bbs)
    else:
        is_gt_bbox = False
        gt_bboxes = [[],[]]

    # if len(gt_bboxes[0]) > 0 or len(gt_bboxes[1]) > 0:
    #     is_gt_bbox = True

    # check if the gt_val has result_bboxes
    if os.path.isfile(f'{results_dir}/labels/img_{gt_val}.txt'):
        is_res_bbox = True
        res_bb = get_bbox_from_txt(gt_val, data_dir, results_dir, gt=False)
        res_bboxes = []
        for ctype in [0,1]:
            res_bboxes.append(get_class_bboxes(res_bb, ctype))
    else:
        is_res_bbox = False
        res_bboxes = [[],[]]

    # print(actual_bbs)
    
    gt_and_res = len(res_bboxes[0]) > 0 or len(res_bboxes[1]) > 0
    # if gt_and_res:
    #     print(is_res_bbox, is_gt_bbox)
    #     print('\ngt_bboxes: ,', gt_bboxes)
    #     print('res_bboxes: ', res_bboxes)
    #     print(len(gt_bboxes), len(res_bboxes))
        # break

    if gt_and_res:
        print('\nGT Image ID:')
        print(gt_val)
        print(is_res_bbox, is_gt_bbox)
        for i in range(2):
            print('\ncell type: ', i)
            print('gt_bboxes: ', gt_bboxes[i])
            print('res_bboxes: ', res_bboxes[i])
            print(len(gt_bboxes[i]), len(res_bboxes[i]))
            print('\n\n')
    
    # case where both files exist
    if is_gt_bbox and is_res_bbox:
        for ctype in range(2):
            # ious = get_ious_single_image(gt_bboxes[ctype], res_bboxes[ctype])
            # print('evaluating bbox ious')
            if ctype == 0:
                print(f'******\nctype {ctype}, calculating IOUs and evaluating')
                print('number of boxes of this cell type:')
                print(len(gt_bboxes[ctype]), len(res_bboxes[ctype]))
            tpp, fpp, fnn = evaluate_bboxes_pos(gt_bboxes[ctype], res_bboxes[ctype])
            print('tpp, fpp, fnn:')
            print(tpp, fpp, fnn)
            tp[ctype] += tpp
            fp[ctype] += fpp
            fn[ctype] += fnn
            
    # false negative case
    elif is_gt_bbox and not is_res_bbox:
        for ctype in range(2):
            fn[ctype] += len(gt_bboxes[ctype])
            
    # false positive case
    elif not is_gt_bbox and is_res_bbox:
        for ctype in range(2):
            fp[ctype] += len(res_bboxes[ctype])
    
    
    # if gt_and_res:
    #     break
    # elif not is_gt_bbox and not is_res_bbox:
    #     tn += 1

print('\n*********\nActual BBs')
print(actual_bbs)

print('True positives')
print(tp)
print('False positives')
print(fp)
print('False negatives')
print(fn)

precision = []
recall = []
f1 = []
acc = []
for i in range(2):
    print(f'\nCell type {i}:')
    if tp[i] != 0 or fp[i] != 0:
        precision.append(tp[i]/(tp[i]+fp[i]))
    else:
        precision.append(0)
    recall.append(tp[i]/(tp[i]+fn[i]))
    print(precision[i], recall[i])
    if precision[i] != 0 or recall[i] != 0:
        f1.append(2*(precision[i]*recall[i])/(precision[i]+recall[i]))
    else:
        f1.append(0)
    acc.append(tp[i]/(actual_bbs[i]))

print('Precision: ', precision)
print('Recall: ', recall)
print('F1 scores: ', f1)
print('Accuracy: ', acc)



