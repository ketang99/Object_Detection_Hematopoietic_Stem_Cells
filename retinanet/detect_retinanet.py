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
sys.path.append(f'{homedir}/retinanet')

from dataloader_faster_rcnn import FasterRCNN_2D_Dataset

parser = argparse.ArgumentParser(description='YOLO dataset generator')
parser.add_argument('-d', '--datadir', default='yolo_dataset')
parser.add_argument('-e', '--epochs_full', type=int, default=10)
parser.add_argument('-p', '--pretrain', type=int, default=0)
parser.add_argument('-r', '--rscale', type=int, default=3)
parser.add_argument('-t', '--transform', type=int, default=0)
parser.add_argument('-o', '--overfit', type=int, default=0)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-c', '--clip_grad', type=int, default=1)
parser.add_argument('-f', '--freeze_batchnorm', type=int, default=1)
parser.add_argument('--model_folder', type=str, default='faster_rcnn_50')
parser.add_argument('--stop_training', type=int, default=0)
parser.add_argument('-a', '--ax_fix', type=int, default=0)

args = parser.parse_args()

axes = ['Z', 'Y', 'X']
ax_fix = axes[args.ax_fix]
# big_dir = f'yolo_dataset_new_{axes[ax_fix]}_r3'
big_dir = f'2d_bigdataset_new_{ax_fix}_r3'
phase = 'Test'
num_epochs_no_roi = args.epochs_full
num_epochs_full = args.epochs_full
# model_folder = f'retinanet_resnet50_e{num_epochs_full}_o{args.overfit}'
# model_folder = f'retinanet_resnet50_correctbigset_allclasses_{ax_fix}_e{num_epochs_full}_o{args.overfit}'
model_folder = f'retinanet_resnet50_correctbigset_allclasses_{ax_fix}_e100_o0_earlystop1'
model_dir = f'{homedir}/retinanet/models/{model_folder}'
#model_dir = '/home/kgupta/data/registration_testing/faster_rcnn/models/faster_rcnn_50_e50_o0_noroi20_earlystop1'

print('model_dir:', model_dir)

def main():
    
    #return
    if args.transform:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std
        ])
    else:
        transform = None
    
    model = torch.load(f'{model_dir}/model.pth')
    
    # Initialize the test dataset and dataloader
    test_dataset = FasterRCNN_2D_Dataset(big_dir, 'Test', transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)
    
    # Put the model in evaluation mode
    model.eval()
    
    # Ensure the model is on the appropriate device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Directory to save the prediction results
    output_dir = f'{model_dir}/predictions'
    if 'predictions' not in os.listdir(model_dir):
        os.mkdir(output_dir)


    print('Output directory: ', output_dir)
    print('*******\nStarting predictions')
    # Run predictions
    with torch.no_grad():
        for images, label_names in test_loader:
            images = list(image.to(device) for image in images)
            
            # print(label_name)
            # return
            # Get predictions
            predictions = model(images)
            
            for label_name, prediction in zip(label_names, predictions):
                # Extract filename without extension
                # filename = os.path.splitext(os.path.basename(img_path))[0]
                output_file = os.path.join(output_dir, f'{label_name}')
                
                with open(output_file, 'w') as f:
                    boxes = prediction['boxes'].cpu().numpy()
                    labels = prediction['labels'].cpu().numpy()
                    scores = prediction['scores'].cpu().numpy()
                    
                    for box, label, score in zip(boxes, labels, scores):
                        # Write each prediction to the file
                        f.write(f'{label} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')

        print('\n**************\nDONE')

if __name__ == '__main__':
    main()
