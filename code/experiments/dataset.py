import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class NumpyDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images = np.load(images_path)
        self.masks = np.load(masks_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = NumpyDataset('man/data/test_images.npy', 'man/data/train_masks.npy', transform=transform)
val_dataset = NumpyDataset('man/data/val_images.npy', 'man/data/val_masks.npy', transform=transform)
test_dataset = NumpyDataset('man/data/test_images.npy', 'man/data/test_masks.npy', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print('train_loader:', train_loader.size, len(train_loader.dataset))
print('val_loader:', len(val_loader), len(val_loader.dataset))