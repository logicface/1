import numpy as np
import os

# 载入npy文件

# 获取文件夹下所有npy文件的最大值和最小值
folder_path = './data'  # Replace with the folder containing local .npy files.
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        max_value = np.max(data)
        min_value = np.min(data)
        print(f"文件:{filename} - 最大值: {max_value}, 最小值: {min_value}")
