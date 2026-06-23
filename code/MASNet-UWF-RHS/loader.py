from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
import os
from PIL import Image
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage
from utils import get_logger, log_config_info, set_seed, get_optimizer, get_scheduler


# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


## Temporary
class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(isic_loader, self)
        self.train = train
        if train:
            print(os.path.join(path_Data, 'data_train_images.npy'))
            self.data = np.load(os.path.join(path_Data, 'data_train_images.npy'))
            self.mask = np.load(os.path.join(path_Data, 'data_train_masks.npy'))
        else:
          if Test:
            print(os.path.join(path_Data, 'data_test_images.npy'))
            self.data = np.load(os.path.join(path_Data, 'data_test_images.npy'))
            self.mask = np.load(os.path.join(path_Data, 'data_test_masks.npy'))
          else:
            # print(os.path.join(path_Data, '-------------------------------------------------------data_val_images.npy-------------------------------------------------------'))
            self.data = np.load(os.path.join(path_Data, 'data_val_images.npy'))
            self.mask = np.load(os.path.join(path_Data, 'data_val_masks.npy'))   
        # print(f"1__init__ Image min: {self.data.min()}, max: {self.data.max()}")
        # print(f"1__init__ Label min: {self.mask.min()}, max: {self.mask.max()}")
        self.data  = dataset_normalized(self.data)
        self.mask  = np.expand_dims(self.mask,axis=3)
        self.mask  = self.mask /255.
        # print(f"1min: {self.mask.min()}, max: {self.mask.max()}")

        # print(f"2__init__ Image min: {self.data.min()}, max: {self.data.max()}")
        # print(f"2__init__ Label min: {self.mask.min()}, max: {self.mask.max()}")


    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        # print(f"1Image min: {img.min()}, max: {img.max()}")
        # print(f"1Image min: {img.min()}, max: {img.max()}")
        # print(f"1Label min: {seg.min()}, max: {seg.max()}")
        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_flip(img, seg)
                # print(f"2Image min: {img.min()}, max: {img.max()}")
                # print(f"2seg min: {seg.min()}, max: {seg.max()}")
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)
                # print(f"3Image min: {img.min()}, max: {img.max()}")
                # print(f"3seg min: {seg.min()}, max: {seg.max()}")
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        # print(img.shape, seg.shape)
        seg = seg.squeeze(2)
        # img = img.unsqueeze(2)
        if img.shape[0] == 384:
            img = img.permute(1, 0, 2)
            seg = seg.permute(1, 0, 2)
        # print(f"__getitem__ Image min: {img.min()}, max: {img.max()}")
        # print(f"__getitem__ Label min: {seg.min()}, max: {seg.max()}")
        return img, seg
    
    def random_flip(self,image, label):
        #k = 2
        #image = np.rot90(image, k)
        #label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        # print(f" random_flip Image min: {image.min()}, max: {image.max()}")
        # print(f" random_flip Label min: {label.min()}, max: {label.max()}")
        return image, label
    
    def random_rotate(self,image, label):
        angle = np.random.randint(10, 30)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)    
        return image, label


               
    def __len__(self):
        return len(self.data)