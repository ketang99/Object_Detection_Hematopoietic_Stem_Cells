import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class FasterRCNN_2D_Dataset(Dataset):
    def __init__(self, big_dir, phase='Train', transform=None, overfit = 0):

        self.phase_file = f'full_{phase}.txt' 
        
        self.big_dir = big_dir
        self.img_dir = f'{self.big_dir}/images'
        self.label_dir = f'{self.big_dir}/labels'
        self.phase = phase
        self.overfit = overfit
        
        with open(f'{self.big_dir}/{self.phase_file}', 'r') as file:
            self.img_paths = file.read().splitlines()

        if self.overfit:
            if phase == 'Train':
                self.img_paths = [self.img_paths[0]]
                self.img_paths = ['/home/kgupta/data/registration_testing/2d_bigdataset_new_Z_r3/images/img_569.jpg']
            elif phase == 'Val':
                self.img_paths = ['/home/kgupta/data/registration_testing/2d_bigdataset_new_Z_r3/images/img_542.jpg']
            print(self.img_paths[0])

        
        # to test this class
        # self.img_paths = self.img_paths[:8]
        # print(self.img_paths)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32)
        # normalize img to [0,1]
        img /= 255.0
        img_size = img.shape[1]
        if self.overfit:
            print('img_shape: ', img.shape)
            print('img_size: ', img_size)
        
        # Load label
        label_name = os.path.basename(img_path).replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        # print(label_path)
        boxes = []
        labels = []

        if self.phase != 'Test':
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    for line in file.readlines():
                        class_id, y_center, x_center, y_width, x_width = map(float, line.strip().split())
                        if class_id != 0:
                            labels.append(int(class_id))
    
                            # print('class_id, y_center, x_center, y_width, x_width')
                            # print(class_id, y_center, x_center, y_width, x_width)
                            # Convert to [xmin, ymin, xmax, ymax] format
                            xmin = (x_center - x_width / 2) * img_size
                            ymin = (y_center - y_width / 2) * img_size
                            xmax = (x_center + x_width / 2) * img_size
                            ymax = (y_center + y_width / 2) * img_size
                            boxes.append([xmin, ymin, xmax, ymax])
           
            #print('boxes: ', boxes)
            #print('')
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {'boxes': boxes, 'labels': labels}
            
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            
            if self.overfit: 
                print('Image tensor shape: ', img.shape)
    
            return img, target

        
        else:
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            
            if self.overfit: 
                print('Image tensor shape: ', img.shape)
            
            return img, label_name
