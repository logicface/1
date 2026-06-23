from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms
import random

class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train=True, Test=False):
        super(isic_loader, self).__init__()
        self.train = train
        self.path_Data = path_Data
        self.image_paths = []
        self.mask_paths = []

        if train:
            image_dir = os.path.join(path_Data, 'train/images')
            mask_dir = os.path.join(path_Data, 'train/masks')
        else:
            if Test:
                image_dir = os.path.join(path_Data, 'test/images')
                mask_dir = os.path.join(path_Data, 'test/masks')
            else:
                image_dir = os.path.join(path_Data, 'val/images')
                mask_dir = os.path.join(path_Data, 'val/masks')

        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, indx):
        img_path = self.image_paths[indx]
        mask_path = self.mask_paths[indx]

        img = Image.open(img_path).convert('L')
        seg = Image.open(mask_path).convert('L')

        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_flip(img, seg)
            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)

        img = self.transform(img)
        seg = self.transform(seg)

        return img, seg

    def random_flip(self, image, label):
        axis = random.randint(0, 1)
        image = image.transpose(Image.FLIP_LEFT_RIGHT if axis == 0 else Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_LEFT_RIGHT if axis == 0 else Image.FLIP_TOP_BOTTOM)
        return image, label

    def random_rotate(self, image, label):
        angle = random.randint(10, 30)
        image = image.rotate(angle)
        label = label.rotate(angle)
        return image, label

    def __len__(self):
        return len(self.image_paths)