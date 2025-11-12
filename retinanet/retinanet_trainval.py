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
from torchvision.models.detection import retinanet_resnet50_fpn

import sys
homedir = '/home/kgupta/data/registration_testing'
os.chdir(homedir)

sys.path.append(homedir)
sys.path.append(f'{homedir}/retinanet')

from dataloader_faster_rcnn import FasterRCNN_2D_Dataset

parser = argparse.ArgumentParser(description='YOLO dataset generator')
parser.add_argument('-d', '--datadir', default='yolo_dataset')
parser.add_argument('-e', '--epochs_full', type=int, default=10)
parser.add_argument('-w', '--epochs_frozen', type=int, default=10)
parser.add_argument('-p', '--pretrain', type=int, default=0)
parser.add_argument('-r', '--rscale', type=int, default=3)
parser.add_argument('-t', '--transform', type=int, default=0)
parser.add_argument('-o', '--overfit', type=int, default=0)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-c', '--clip_grad', type=int, default=1)
parser.add_argument('-f', '--freeze_batchnorm', type=int, default=1)
parser.add_argument('--stop_training', type=int, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('-a', '--ax_fix', type=int, default=0)

args = parser.parse_args()

print('Pretrained? ', args.pretrain)
print('Transform to ImageNet? ', args.transform)
print('Epochs: ', args.epochs_full)
print('Epochs frozen: ', args.epochs_frozen)
print('Batch size: ', args.batch_size)
print('Overfit? ', args.overfit)
print('Clip grads? ', args.clip_grad)
print('Freeze batchnorm? ', args.freeze_batchnorm)
print('Stop training based on val loss? ', args.stop_training)


if args.overfit:
    print('\nOverfit!!\n')

axes = ['Z','Y','X']
ax_fix = axes[args.ax_fix]

print('ax fix: ', ax_fix)

# big_dir = 'yolo_dataset_new_{ax_fix}_r3'
big_dir = f'yolo_bigdataset_new_{ax_fix}_r3'
big_dir = f'2d_bigdataset_new_{ax_fix}_r3'
phase = 'Train'
num_epochs_no_roi = args.epochs_frozen
num_epochs_full = args.epochs_full
model_folder = f'retinanet_resnet50_correctbigset_allclasses_{ax_fix}_e{num_epochs_full}_o{args.overfit}'
if not args.clip_grad:
    model_folder = model_folder + '_noclip'
if args.stop_training:
    print('    early stopping will be done if needed')
    model_folder = model_folder + f'_earlystop1'
model_dir = f'{homedir}/retinanet/models/{model_folder}'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('model directory: ', model_dir)


if model_folder not in os.listdir(f'{homedir}/retinanet/models/'):
    os.mkdir(model_dir)
else:
    if 'details.txt' in os.listdir(model_dir):
        os.remove(f'{model_dir}/details.txt')

# write txt file with training details
with open(f'{model_dir}/details.txt', 'w') as f:
    f.write(f'big_dir: {homedir}/{big_dir}\n')
    f.write(f'model_dir: {model_dir}\n')
    f.write(f'num_epochs_no_roi: {num_epochs_no_roi}\n')
    f.write(f'num_epochs_full: {num_epochs_full}\n')
    f.write(f'batch_size: {args.batch_size}\n')
    f.write(f'Overfit: {args.overfit}\n')
    f.write(f'Pretrained: {args.pretrain}\n')
    f.write(f'Transform to imagenet: {args.transform}\n')
    f.write(f'Clip gradients in full training: {args.clip_grad}\n')
    f.write(f'Freeze batchnorm in full training: {args.freeze_batchnorm}\n')
    f.write(f'Stop based on val loss: {args.stop_training}\n')
    f.write(f'Patience for early stopping: {args.patience}\n')

def main():
    
    #return
    # ResNet50 expects images to be normalized with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]
    # Define the transformations
    if args.transform:
        print('Will transform to imagenet')
        #transform = transforms.Compose([
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std
        ])
    else:
        transform = None

    '''
    The above comment about resnet50 expectations are wrong; our model expects images to be normalized to [0,1]
    The dataloader takes care of this by dividing each image by 255.0
    '''

    # Create the custom anchor generator
    #anchor_sizes = ((8, 16, 32, 48, 64),)

    # Create the RoI align layer
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Load a pretrained ResNet-50 backbone
    # backbone = torchvision.models.resnet50(pretrained=True)
    # backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))

    # Load a pretrained Faster R-CNN model with a ResNet-50 backbone
    #model = fasterrcnn_resnet50_fpn(
    #    pretrained=args.pretrain,
    #    rpn_anchor_generator=anchor_generator
    #)

    model = retinanet_resnet50_fpn(num_classes=3)
    #model.rpn.anchor_generator = anchor_generator
    
    print('\nModel modules')
    for name, module in model.named_modules():
        print(f"Module: {name}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print('Model has been imported')

    # initialize the dataset and dataloader
    train_dataset = FasterRCNN_2D_Dataset(big_dir, 'Train', transform, args.overfit)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))


    # Train the model
    # model.train()
    # Define an optimizer and a learning rate for the backbone and RPN
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)

    # unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    if args.freeze_batchnorm:
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()  # Freeze batch norm layers and prevents mean and var from being updated
        print('Batchnorm frozen')

    #if args.clip_grad:
    #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
       # print('Gradients clipped')

    print('    \nTraining done and ROI heads unfrozen\n\n')

    #return
    # initialize validation dataset
    val_dataset = FasterRCNN_2D_Dataset(big_dir, 'Val', transform, args.overfit)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    # Define a new optimizer for the entire model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


    print('\n Beginning full training\n*********')
    train_losses = []
    val_losses = []
    '''
    To save the losses do the following:
    1. Get the loss_dict and extract the keys: put the key names in an array
    2. loss_dict will have n=4 tensors; each tensor has a single scalar value
    3. Using .item(), get the scalar values as floats
    4. Write these floats to an array called epoch_loss: array will have n=4 terms
    5. Append epoch_loss to train or val losses
    6. After all the epochs are done, convert the two loss arrays to a numpy array and save
    '''
    # train and validate the full model
    best_val_loss = float('inf')
    patience = args.patience
    trigger_times = 0
    stop_epoch = num_epochs_full
    for epoch in range(num_epochs_full):
        model.train()
        total_samples = 0
        epoch_train_losses = []
        
        print(f'\n*****\nTraining\n\nEpoch {epoch+1}')
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            if args.overfit:
                print('targets: ')
                print(targets)

            batch_size = len(images)
            total_samples += batch_size
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
    
            #if args.overfit:
            #    print('loss stuff')
            #    print('summed losses: ', losses)
            #     print(loss_dict)
            #     print([loss for loss in loss_dict.values()]) 
            #     print([loss.size for loss in loss_dict.values()])
            #     print([type(loss) for loss in loss_dict.values()])


            losses_array = [loss.item() * batch_size for loss in loss_dict.values()]
            epoch_train_losses.append(losses_array)  # will get the four losses for each batch

            optimizer.zero_grad()
            losses.backward()
            
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()

        # epoch_train_losses contains num_batch arrays of 4 values each
        print('epoch train losses shape before summing along number of batches: ')
        print(np.array(epoch_train_losses).shape)
        epoch_train_losses = np.sum(np.array(epoch_train_losses), axis=0)  # sum along the number of batches
        print('shape after summing: ')
        print(epoch_train_losses.shape)  
        epoch_train_losses /= total_samples # divide by total number of samples to get mean that accounts for diff batch sizes
        if epoch_train_losses.shape[-1] == len(loss_dict):
            sumax = -1
        else:
            sumax = 0
        train_losses.append(epoch_train_losses)
        print(f'\n\nEpoch {epoch+1}/{num_epochs_full}, Train total loss: {np.sum(epoch_train_losses, axis=sumax)}')
        print(f'{list(loss_dict.keys())}: ')
        print(epoch_train_losses)
        # epoch_train_losses = np.sum(epoch_train_losses, axis=sumax) # sums losses of the batches
        #train_losses.append(epoch_train_losses)

        # Validation: model is kept in training mode as in eval mode it will not return the losses; it will return the detection results
        print('************\nValidation')
        model.train()
        val_loss = 0
        total_samples = 0
        epoch_val_losses = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                if args.overfit:
                    print('targets: ')
                    print(targets)


                batch_size = len(images)
                total_samples += batch_size

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()

                losses_array = [loss.item() * batch_size for loss in loss_dict.values()]
                epoch_val_losses.append(losses_array)


        epoch_val_losses = np.sum(np.array(epoch_val_losses), axis=0)
        epoch_val_losses /= total_samples
        val_losses.append(epoch_val_losses)
        total_val_loss = np.sum(epoch_val_losses, axis=sumax)
        print(f'\n\nEpoch {epoch+1}/{num_epochs_full}, Validation total loss: {total_val_loss}')
        print(f'{list(loss_dict.keys())}: ')
        print(epoch_val_losses)
        print('\n*****')

        if args.stop_training:
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f'*******\nEarly stopping! Epochs = {epoch+1}')
                    stop_epoch = epoch
                    torch.save(model, f'{model_dir}/model.pth')
                    print('MODEL SAVED\n******')
                    break
        # epoch_val_losses = np.sum(epoch_val_losses, axis=sumax)
        #val_losses.append(epoch_val_losses)

        # val_loss /= len(val_loader)
        # epoch_loss = [loss.item() for loss in loss_dict.values]
        # val_losses.append(epoch_loss)

        if epoch == num_epochs_full-1:
            torch.save(model, f'{model_dir}/model.pth')
            print('MODEL SAVED\n******')

        stop_epoch = epoch

    print('\n\n\n*****')
    print('Training and Validation Complete')

    with open(f'{model_dir}/details.txt', 'a') as f:
        f.write(f'Final epoch: {stop_epoch}\n')
    
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    
    print('Train and val loss shapes')
    print(train_losses.shape)
    print(val_losses.shape)
    
    loss_names = list(loss_dict.keys())
    # loss_names = ['cls', 'box_reg', 'objectness', 'rpn']
    for i, nam in enumerate(loss_names):
        print(f'\ntrain loss: {nam}')
        print(train_losses[:,i])
        print(f'\nval loss: {nam}')
        print(val_losses[:,i])

    print('\nOverall train loss:')
    print(np.sum(train_losses,axis=-1))
    print('\nOverall val loss:')
    print(np.sum(val_losses,axis=-1))

    # save the train and val losses
    np.save(f'{model_dir}/loss_Train.npy', train_losses)
    np.save(f'{model_dir}/loss_Val.npy', val_losses)
    print('Losses saved')

    print('END')


if __name__ == '__main__':
    main()
