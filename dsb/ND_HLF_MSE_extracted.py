'''
This script will find the loss for different values of lambda
The train and val functions need to be modified in order to save arrays for the total, cls and reg losses
    - this must be done for each lambda value
Iterate through lambda: reinitialize the model and parameters each time
Be sure to use the same scheduler and optimizer for each lambda
'''


import numpy as np
import h5py
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import math
import pandas as pd
import csv
from matplotlib import pyplot as plt
from importlib import import_module
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.init as init
from torchmetrics.classification import BinaryHingeLoss
import sys

all_values = True
epochs = 100
home_dir = '/data/kgupta/registration_testing'
os.chdir(home_dir)
root_dir = home_dir + '/dsb/red_dsb'
ims_dir = home_dir + '/h5_files'
train_filename = 'RED_DSB_trainsplit.h5'
mname = f'test_mse_scheduler_lambda_epoch{epochs}_defanch_sgd_red_dsb_2patch_e-1'
model_filename = 'singlechannel_red'
model_dir = home_dir + f'/models/{mname}'
model_save_path = f'{model_dir}/epoch_models'
loss_save_path = f'{model_dir}/loss'
sys.path.append(root_dir)
sys.path.append(root_dir + '/training')
sys.path.append(root_dir + '/training/classifier')
sys.path.append(home_dir + '/dsb')
print('Added dirs to path')

if mname not in os.listdir(f'{home_dir}/models'):
    os.mkdir(model_dir)

if 'loss' not in os.listdir(model_dir):
    os.mkdir(loss_save_path)

if 'epoch_models' not in os.listdir(model_dir):
    os.mkdir(model_save_path)

import reg_functions as reg
import data_red_dsb as dsb
from layers import *
import net_detector_3 as nd
# contains the model
import trainval_detector as det
# contains the function that performs training
from config_training import config as config_training
from split_combine import SplitComb

print('DSB modules imported')

from utils import *

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model1', '-m1', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--model2', '-m2', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-e', '--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-b2', '--batch-size2', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='5', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test1', default=0, type=int, metavar='TEST',
                    help='do detection test')
parser.add_argument('--test2', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--test3', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--debug', default=0, type=int, metavar='TEST',
                    help='debug mode')
parser.add_argument('--freeze_batchnorm', default=0, type=int, metavar='TEST',
                    help='freeze the batchnorm when training')


'''
Gonna have 2 types of losses here, both of which will have the form L_cls + lambda*L_reg (these L values will both use MSE):
1. Finds MSE for all the output values
2. Finds MSE for output values that are selected based on the IOU thresholds (pos plus neg cases with hard mining)

sigmoid will be done for classification
'''

def hard_mining(neg_output, neg_labels, neg_idcs, num_hard):
    _, idcs = torch.topk(neg_output[:,0], min(num_hard, neg_output.size(0)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels, neg_idcs[idcs]

class MSELoss_adaptable(nn.Module):
    def __init__(self, num_hard = 0):
        super(MSELoss_adaptable, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.classify_loss = nn.BCELoss()
        # self.hinge_loss = BinaryHingeLoss().to('cuda:0')
        # self.regress_loss = nn.SmoothL1Loss()
        self.criterion = nn.MSELoss()
        # self.num_hard = num_hard

    def forward(self, output, labels, train = True, lambdafactor = 0.2):
        batch_size = labels.size(0)
        print('')
        print('all_vals: ', all_values)
        print('labels shape: ', labels.size())
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)
        print('labels shape after .view: ', labels.size())        
        print('labels shape: ', labels.size())
        
        # pos_idcs = labels[:, 0] > 0.5
        # pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        # pos_output = output[pos_idcs].view(-1, 5)
        # pos_labels = labels[pos_idcs].view(-1, 5)

        if all_values:
            l_cls = self.criterion(self.sigmoid(output[:,0]), self.sigmoid(labels[:,0]))
            loss_out = l_cls
            regress_losses = []
            for i in range(4):
                l_reg_current = self.criterion(output[:,i+1], labels[:,i+1])
                regress_losses.append(l_reg_current)
                loss_out += lambdafactor * l_reg_current

            return [loss_out, l_cls] + [regress_losses]

        
        else:
            pos_idcs = labels[:,0]>0.5
            pos_output = output[pos_idcs, :]
            pos_labels = labels[pos_idcs, :]
    
            neg_idcs = labels[:, 0] < -0.5
            # neg_output = output[:, 0][neg_idcs]
            # neg_labels = labels[:, 0][neg_idcs]
            neg_output = output[neg_idcs, :]
            neg_labels = labels[neg_idcs, :]
            # print(len(neg_labels))
            
            print('labels and pos_idcs types: ', type(labels), type(pos_idcs))
            print('pos_output and pos_labels types: ', type(pos_output), type(pos_labels))
            print('size of neg_idcs: ', neg_idcs.size())
            print('size of neg_output and neg_labels: ', neg_output.size(), neg_labels.size())
    
            print('COLUMNS OF LABELS AND OUTPUT: o^, dz, dy, dx, dr')
    
            nhard = pos_labels.size(dim=0)
            print('nhard: ', nhard)
            if self.num_hard > 0 and train:
                # neg_output, neg_labels = hard_mining(neg_output, neg_labels, nhard * batch_size)
                neg_output, neg_labels, hard_idcs = hard_mining(neg_output, neg_labels, neg_idcs, nhard * batch_size)
                # print('hard_idcs: ', hard_idcs)
                print('After hard mining')
                print('neg_output_cls: ', neg_output, neg_output.size())
                print('neg_labels+1: ', neg_labels+1, (neg_labels+1).size())
                #print('neg_output[]: ')
                #print(output[hard_idcs.cpu().numpy(),:])
                #print('neg_labels[]:')
                #print(labels[hard_idcs,:])
    
    
            neg_prob = self.sigmoid(neg_output[:,0])
    
            # print(neg_prob.shape)
            # print(len(neg_labels), neg_output.shape)
            
            # the below 3 lines concatenate pos and neg probs to find the loss together
            #classify_loss = self.classify_loss(
             #   torch.cat((pos_prob, neg_prob), 0),
              #  torch.cat((pos_labels[:, 0], neg_labels + 1), 0))
    
            if len(pos_output)>0:
                
                pos_prob = self.sigmoid(pos_output[:, 0])
                # print('pos_prob: ', pos_prob, pos_prob.size())
                print('Entered loss if statement')
                print('pos_prob: ', pos_prob, pos_prob.size(), pos_prob.device)
                print('pos_labels[:,0]: ', pos_labels[:,0], pos_labels[:,0].size(), pos_labels[:,0].device)
                # print('neg_prob: ', neg_prob, neg_prob.size())
                print('neg_output[:,0]: ', neg_output[:,0], neg_output[:,0].size(), neg_output[:,0].device)
                print('neg_labels[:,0]+1: ', neg_labels[:,0]+1, (neg_labels[:,0]+1).size(), (neg_labels[:,0]+1).device)
                
                print('')
                print('pos_output[:,:]: ')
                print(pos_output, pos_output.size(), pos_output.device)
                print('pos_labels[:,:]:')
                print(pos_labels, pos_labels.size(), pos_labels.device)
    #            print('neg_output[]: ')
    #            print(output[hard_idcs,:])
    #            print('neg_labels[]:')
    #            print(labels[hard_idcs,:])

                neg_labels[:,0]+=1  # the first column of neg_labels has 1 added to it to have the cls prob be 0
                combined_output = torch.cat((pos_out, neg_output), axis=0)
                combined_labels = torch.cat((pos_labels, neg_labels), axis=0)

                print('combined labels[:,0] before prob: ', combined_labels[:,0]) 
                
                combined_labels[:,0][combined_labels[:,0] != 1] = 0
                print('combined labels[:,0] after prob: ', combined_labels[:,0]) 
                
                l_cls = self.criterion(self.sigmoid(combined_output[:,0]), combined_labels[:,0])
                loss_out = l_cls
                regress_losses = []
                for i in range(4):
                    l_reg_current = self.criterion(combined_output[:,i+1], combined_labels[:,i+1])
                    regress_losses.append(l_reg_current)
                    loss_out += lambdafactor * l_reg_current

                return [loss_out, l_cls] + [regress_losses]
                
        #     else:
        #         # print('Entered the else statement')
        #         regress_losses = [0,0,0,0]
        #         # classify_loss =  0.5 * self.classify_loss(
        #         # neg_prob, neg_labels + 1)
        #         classify_loss =  0.5 * self.classify_loss(
        #                 neg_output[:,0], neg_labels[:,0]+1)
        #         pos_correct = 0
        #         pos_total = 0
        #         regress_losses_data = [0,0,0,0]

        #     loss = classify_loss
        #     for regress_loss in regress_losses:
        #         loss += regress_loss
    
        #     neg_correct = (neg_prob.data < 0.5).sum()
        #     neg_total = len(neg_prob)

        # return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]

def get_lr(epoch,args):
    assert epoch<=args.lr_stage[-1]
    if args.lr==None:
        lrstage = np.sum(epoch>args.lr_stage)
        lr = args.lr_preset[lrstage]
    else:
        lr = args.lr
    return lr


def train_nodulenet(data_loader, net, loss, epoch, loss_array, optimizer, args, epoch_save, model_filename, save_path, loss_path):
    criterion = nn.MSELoss()
    start_time = time.time()
    net.train()
    #if args.freeze_batchnorm:
    #    for m in net.modules():
    #        if isinstance(m, nn.BatchNorm3d):
    #            m.eval()

    #lr = get_lr(epoch,args)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr

    raw_metrics = []
   # for (i, batch) in enumerate(data_loader):
    # torch.cuda.empty_cache()
    # iterating through each patch
    # optimizer.zero_grad()
    for (i, batch) in enumerate(data_loader):
        
        print('loader iter: ', i)
        optimizer.zero_grad
        data = batch['patch'].to('cuda:0')
        target = batch['label'].to('cuda:0')
        # print('Data length: ', len(data))
        # print('Data shape: ', data.shape)
        # print('Target shape: ', target.shape)

        _, output = net(data)
        loss_output = loss(output, target)
        loss_array.append(loss_output)
        loss_output[0].backward()
        # print('loss_output dims: ', len(loss_output))
        
        #for t in range(len(loss_output)):
           # print(t, type(loss_output[t]))
           # if isinstance(loss_output[t], torch.Tensor):
            #    print('tensor size: ', loss_output[t].size())
            #print(loss_output[t])

        # print('cls term: ', type(loss_output[1]))
        # torch.tensor(loss_output[1], requires_grad=True).backward()
        
        printgrad = True
        large_names = []
        
        if printgrad:
            for name, param in net.named_parameters():
                if param.grad is not None:
                    print(f'{name}.grad: {param.grad.norm()}')
                    if (param.grad.norm()>0.05):
                        large_names.append(name)
                        print('True')

            print('')
            print(large_names)
            print('')

        optimizer.step()

    # scheduler.step()
        # loss_output[0] = loss_output[0].item()
        # print('loss output val: ', loss_output[0])
        # raw_metrics.append(loss_output)

    end_time = time.time()

    print('Training loss: ')
    print(loss_output)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr}")
    
    print('')

    # save the loss metrics
    # np.save(f'{loss_path}/Train_loss_metrics_{epoch}.npy', metrics)
    
    # save the parameters
    # if epoch_save:
    #     torch.save(net.state_dict(), f'{save_path}/{model_filename}_{epoch}_state_dict.pth')
    #     torch.save(net, f'{save_path}/{model_filename}_{epoch}.pth')


def validate_nodulenet(data_loader, net, loss, epoch, loss_array, args, save_path, loss_path):
    # criterion = nn.MSELoss()
    start_time = time.time()
    
    net.eval()

    raw_metrics = []
    for (i, batch) in enumerate(data_loader):
        data = batch['patch'].to('cuda:0')
        target = batch['label'].to('cuda:0')

        _,output = net(data)
        loss_output = loss(output, target, train = False)
        # loss_output = criterion(output, target)
        loss_array.append(loss_output)
        raw_metrics.append(loss_output)    
    end_time = time.time()

    print('Validation loss: ')
    print(loss_output)
    

def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # Call the model
    nodmodel = import_module('net_detector_3')
    config, nod_net, loss_no, get_pbb = nodmodel.get_model()
    del nod_net
    loss = MSELoss_adaptable()

    # optimizer = torch.optim.Adagrad(nod_net.parameters(), args.lr)
    # optimizer = torch.optim.Adam(nod_net.parameters(), args.lr)
    split_comber = SplitComb(192, config['max_stride'], config['stride'], 32, 0)
    # Call data object and dataloader from anchor_loss (dsb)
    train_dataset = dsb.HLFBoneMarrowCells(train_filename, ims_dir, config, split_comber, phase='Train')
    val_dataset = dsb.HLFBoneMarrowCells(train_filename, ims_dir, config, split_comber, phase='Val')
    print(f'batch size = {args.batch_size}')
    print(f'learning rate = {args.lr}')
    print(f'weight decay: {args.weight_decay}')
   # return
    
    train_loader_nod = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=0, collate_fn=dsb.custom_collate_dict_new)
    val_loader_nod = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=0, collate_fn=dsb.custom_collate_dict_new)
    print('Train and val loaders initialized')

    print('Freeze batchnorm? ', args.freeze_batchnorm)
    if args.freeze_batchnorm:
        print('Freeze is true, make it false to continue')
        return

    # args.lr_stage = config['lr_stage']

    print('all_values for loss? ', all_values)
    print('Entering training epochs')
    print('')
    print(f'Epochs = {epochs}')
    
    epochs_to_save = range(epochs)
    start_time = time.time()
    train_losses = []
    val_losses = []
    lrates = []

    # iterate through different values of lambda
    lambdas = np.linspace(0,1,11)
    for lambda_val in lambdas:
        print('*****')
        print(f'lambda = {lambda_val}')
        
        # initialize model, scheduler, optimizer
        nodmodel = import_module('net_detector_3')
        config, nod_net, loss_no, get_pbb = nodmodel.get_model()
        #loss = MSELoss_adaptable()
        #nod_net = torch.nn.parallel.DistributedDataParallel(nod_net)
        if torch.cuda.is_available():
            nod_net = nod_net.cuda()
        
       # nod_net = torch.nn.DataParallel(nod_net)
        
        print('Inits done')
        
        # config['anchors'] = [50.0,55.0,60.0,65.0]
        config['anchors'] = [10.0, 30.0, 60.0]
        config['lr_stage'] = [50,100,150,250]
        print('config: ')
        print(config)
    
        #split_comber = SplitComb(192, config['max_stride'], config['stride'], 32, 0)
        # Call data object and dataloader from anchor_loss (dsb)
        #train_dataset = dsb.HLFBoneMarrowCells(train_filename, ims_dir, config, split_comber, phase='Train')
        #val_dataset = dsb.HLFBoneMarrowCells(train_filename, ims_dir, config, split_comber, phase='Val')

        print('Model retrieved and parallelized')
        #return
        # Initialize the parameters / call them from a checkpoint
        # checkpoint = torch.load(args.resume) # try to see where args.resume points to. Do this if that pointed location exists
        
        #train_loader_nod = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
         #                         num_workers=0, collate_fn=dsb.custom_collate_dict_new)
        #val_loader_nod = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
        #                        num_workers=0, collate_fn=dsb.custom_collate_dict_new)
        #print('Train and val loaders initialized')


        # Call the optimizer (SGD)
        optimizer = torch.optim.SGD(nod_net.parameters(),args.lr,momentum = 0.9,weight_decay = args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        train_losses = []
        val_losses = []
        for i in range(epochs):
            if i in epochs_to_save:
                epoch_save = True
            else:
                epoch_save = False
            
           # lrates.append(optimizer.param_groups[0]['lr'])
            print('************************')
            print(f'Training epoch {i}')
            nod_net.train()
            train_nodulenet(train_loader_nod, nod_net, loss, i, train_losses, optimizer, 
                    args, epoch_save, model_filename, model_save_path, loss_save_path)
            print(f'Training for epoch {i} is complete')
            print('\n***\n')
            print('Validation:')
            validate_nodulenet(val_loader_nod, nod_net, loss, i, val_losses, args, model_save_path, loss_save_path)
            print(f'Validation for epoch {i} is complete')
            print('************************')
            
            current_lr = optimizer.param_groups[0]['lr']
            lrates.append(current_lr)
            print(f'Epoch {i} - Current Learning Rate: {current_lr}')
        
            scheduler.step()
            
            new_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {i} - New Learning Rate after Scheduler Step: {new_lr}')
            #lrates.append(scheduler.get_last_lr())
        
        print('\n\n')
        print('*   *   *   *   *')
        print(f'All {epochs}  epochs complete')
        print(f'Time taken = {time.time()-start_time}')

        # for np save see if the array structure will mess things up.
        np.save(f'{loss_save_path}/Train_loss_metrics_lambda_{lambdaval}_e_{epochs}.npy', train_losses)
        np.save(f'{loss_save_path}/Val_loss_metrics_lambda_{lambdaval}_e_{epochs}.npy', val_losses)
        
        print('Train losses by epoch')
        print(train_losses)
        print('Val losses by epoch')
        print(val_losses)
        print('Learning rates by epoch')
        print(lrates)

        print(f'COMPLETED lambda = {lambda_val}')
        print('\n')
        print('*****')

    print('******************')
    print('ALL TRAINING DONE')

if __name__ == '__main__':
    main()
