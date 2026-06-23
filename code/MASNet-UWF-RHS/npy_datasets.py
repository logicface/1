from PIL import Image
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import Dataset
from configs.config_setting import setting_config

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False):
        super(NPY_datasets, self).__init__()
        if train:
            images_list = os.listdir(path_Data + 'train/images/')
            masks_list = os.listdir(path_Data + 'train/masks/')
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        elif test:
            images_list = os.listdir(path_Data + 'test/images/')
            masks_list = os.listdir(path_Data + 'test/masks/')
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'test/images/' + images_list[i]
                mask_path = path_Data + 'test/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        else:
            images_list = os.listdir(path_Data + 'val/images/')
            masks_list = os.listdir(path_Data + 'val/masks/')
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.val_transformer

    def __getitem__(self, index):
        img_path, msk_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        img, msk = self.transformer((img, msk))
        
        # 打印图像和标签的最小值和最大值
        print(f"Image min: {np.array(img).min()}, max: {np.array(img).max()}")
        print(f"Label min: {np.array(msk).min()}, max: {np.array(msk).max()}")
        
        return img, msk

    def __len__(self):
        return len(self.data)

class JointTransform:
    def __init__(self, image_transform, mask_transform):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, img_mask_tuple):
        img, mask = img_mask_tuple
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        return img, mask
    
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False):
        super(NPY_datasets, self).__init__()
        self.data = []
        
        # 根据 train/test/val 划分路径
        if train:
            images_list = os.listdir(path_Data + 'train/images/')
            masks_list = os.listdir(path_Data + 'train/masks/')
            self.transformer = config.train_transformer
        elif test:
            images_list = os.listdir(path_Data + 'test/images/')
            masks_list = os.listdir(path_Data + 'test/masks/')
            self.transformer = config.test_transformer
        else:
            images_list = os.listdir(path_Data + 'val/images/')
            masks_list = os.listdir(path_Data + 'val/masks/')
            self.transformer = config.val_transformer
        
        # 将图像和掩码路径存入列表
        for i in range(len(images_list)):
            img_path = path_Data + 'train/images/' + images_list[i]
            mask_path = path_Data + 'train/masks/' + masks_list[i]
            self.data.append([img_path, mask_path])

    def __getitem__(self, index):
        # 获取图像和掩码路径
        img_path, msk_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')
        
        # 应用变换
        img, msk = self.transformer((img, msk))
        
        # 打印图像和标签的最小值和最大值
        print(f"Image min: {np.array(img).min()}, max: {np.array(img).max()}")
        print(f"Label min: {np.array(msk).min()}, max: {np.array(msk).max()}")
        
        return img, msk

    def __len__(self):
        return len(self.data)

    def save_as_npy(self, output_image_file, output_mask_file):
        """
        将图像和掩码保存为 `.npy` 文件。
        :param output_image_file: 输出图像的 `.npy` 文件路径
        :param output_mask_file: 输出掩码的 `.npy` 文件路径
        """
        images = []
        masks = []
        img_path = 'UWF-RHS Dataset\\images\\'
        msk_path = 'UWF-RHS Dataset\\masks\\'
        # 遍历数据集，加载图像和掩码
        for img_path, msk_path in self.data:
            img = Image.open(img_path).convert('RGB')
            msk = Image.open(msk_path).convert('L')

            # 应用变换（如果有的话）
            img, msk = self.transformer((img, msk))

            # 转换为 NumPy 数组
            images.append(np.array(img))
            masks.append(np.array(msk))

        # 转换为 NumPy 数组并保存为 .npy 文件
        np.save(output_image_file, np.array(images))
        np.save(output_mask_file, np.array(masks))

        print(f"Images and masks have been saved as {output_image_file} and {output_mask_file}.")

# 用法示例：
# config 是你之前传入的配置对象
config = setting_config()
dataset = NPY_datasets(path_Data='d:/graduate/UWF-RHS Dataset/', config=config, train=True)
dataset.save_as_npy('train_images.npy', 'train_masks.npy')