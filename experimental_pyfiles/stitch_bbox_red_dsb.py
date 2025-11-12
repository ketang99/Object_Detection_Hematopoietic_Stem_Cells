import numpy as np
import h5py
import os
import math
import pandas as pd
import csv
from matplotlib import pyplot as plt
import reg_functions as reg
import random


'''
Overall flow:
1) select a patch and find its neighbouring patches
2) get the bboxes and starting position of each patch
    - see if patch starts and patch IDs are stored in the big h5 file
    - that is not the case. Make a script that reads:
        - the patch ids
        - the patch starts
        - stores these two in a dict --> json file
3) convert the bboxes rel to each patch to the bounding box rel to the full image dimensions (metadata)
4) for the selected patch, do an iou of each bbox with bboxes from neighbouring patches
    - if the iou is above some threshold, merge the bboxes
        - how to merge? For now take the extrema of the boxes and use those to recalculate the center and bbox size
    - do iou until it is below the threshold for all bboxes (selected and neighbours)
    - remove the bbox from the selected patch's bboxes array
        - this can be done because for a selected patch, it is impossible for one of its bboxes to not be in the direct adjacent neighbourhood
    - append the merged bbox to an array that will contain all the merged bboxes
'''