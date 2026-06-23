import os
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset.npy_datasets import NPY_datasets # 导入 NPY_datasets 类和 JointTransform 类
from utils import myNormalize, myToTensor, myRandomHorizontalFlip, myRandomVerticalFlip, myRandomRotation, myResize
from configs.config_setting import setting_config


if __name__ == '__main__':
    # 设置图像和掩码文件夹路径以及输出文件路径
    image_folder = './data/UWF-RHS Dataset/'
    mask_folder = './data/UWF-RHS Dataset/'
    output_train_image_file = './data/data_train_images.npy'
    output_val_image_file = './data/data_val_images.npy'
    output_test_image_file = './data/data_test_images.npy'
    output_train_mask_file = './data/data_train_masks.npy'
    output_val_mask_file = './data/data_val_masks.npy'
    output_test_mask_file = './data/data_test_masks.npy'
def preprocess_and_save(image_folder, mask_folder, output_train_image_file, output_val_image_file, output_test_image_file, output_train_mask_file, output_val_mask_file, output_test_mask_file, config):
    # 使用 NPY_datasets 类加载和预处理图像和掩码
    dataset = NPY_datasets(path_Data=image_folder, config=config, train=True)

    # 创建 DataLoader
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # 获取所有数据
    images, masks = next(iter(loader))

    # 将数据转换为 NumPy 数组
    images = images.numpy()

    masks = masks.numpy()

    # 按 7:1:2 比例划分数据集
    train_images, temp_images, train_masks, temp_masks = train_test_split(images, masks, test_size=0.3, random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(temp_images, temp_masks, test_size=2/3, random_state=42)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_train_image_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_image_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_image_file), exist_ok=True)

    # 将 NumPy 数组保存为 .npy 文件
    np.save(output_train_image_file, train_images)
    np.save(output_val_image_file, val_images)
    np.save(output_test_image_file, test_images)
    np.save(output_train_mask_file, train_masks)
    np.save(output_val_mask_file, val_masks)
    np.save(output_test_mask_file, test_masks)

    print(f"Saved {len(train_images)} training images to {output_train_image_file} with shape {train_images.shape}")
    print(f"Saved {len(val_images)} validation images to {output_val_image_file} with shape {val_images.shape}")
    print(f"Saved {len(test_images)} testing images to {output_test_image_file} with shape {test_images.shape}")
    print(f"Saved {len(train_masks)} training masks to {output_train_mask_file} with shape {train_masks.shape}")
    print(f"Saved {len(val_masks)} validation masks to {output_val_mask_file} with shape {val_masks.shape}")
    print(f"Saved {len(test_masks)} testing masks to {output_test_mask_file} with shape {test_masks.shape}")

# 创建一个简单的配置对象
class Configs:
        # 假设 input_size_h 和 input_size_w 是定义好的图像高度和宽度
        input_size_h = setting_config.input_size_h
        input_size_w = setting_config.input_size_w
        datasets = setting_config.datasets

        # 定义训练数据集的变换
        train_transformer = transforms.Compose([
            myNormalize(datasets, train=True),            # 自定义归一化变换
            # myToTensor(),                               # 自定义 ToTensor 变换
            myRandomHorizontalFlip(p=0.5),                # 随机水平翻转
            myRandomVerticalFlip(p=0.5),                  # 随机垂直翻转
            myRandomRotation(p=0.5, degree=[0, 360]),     # 随机旋转
            myResize(input_size_h,input_size_w)  # 自定义调整大小变换
        ])

        # 定义验证数据集的变换
        val_transformer = transforms.Compose([
            myNormalize(datasets, train=False),           # 自定义归一化变换（不进行训练时的归一化）
            # myToTensor(),                                     # 自定义 ToTensor 变换
            myResize(input_size_h, input_size_w)    # 自定义调整大小变换
        ])

    # 创建一个配置对象
        test_transformer = transforms.Compose([
            myNormalize(datasets, train=False),           # 自定义归一化变换（不进行训练时的归一化）
            # myToTensor(),                                     # 自定义 ToTensor 变换
            myResize(input_size_h, input_size_w)    # 自定义调整大小变换
        ]) 
config = Configs()

preprocess_and_save(image_folder, mask_folder, output_train_image_file, output_val_image_file, output_test_image_file, output_train_mask_file, output_val_mask_file, output_test_mask_file, config)
