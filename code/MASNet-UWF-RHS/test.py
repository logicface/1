import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import isic_loader

from models.model import MASNet,LossNet
# from dataset.npy_datasets import NPY_datasets
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import get_logger, log_config_info, set_seed, get_optimizer, get_scheduler
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

# def preprocess_data(data):
#     # 示例数据预处理步骤
#     # 确保数据归一化到 [0, 1] 范围内
#     data = data / 255.0
#     return data

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(r'/mnt/d/graduate/VM-UNet-main/checkpoints/latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    model = MASNet()
    if os.path.isfile(resume_model):
        checkpoint = torch.load(resume_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_model))
    else:
        print("=> no checkpoint found at '{}'".format(resume_model))
    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    
    print('#----------Preparing dataset----------#')
    data_path     =  '/mnt/d/graduate/lsat/datanpy'
    test_dataset  = isic_loader(path_Data = data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                batch_size= 1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    
    for data in test_loader:
        img, msk = data
        # img = preprocess_data(img)

    print('#----------Prepareing Models----------#')
    # model_cfg = config.model_config
    model = MASNet()
    net = LossNet()
    model.train(True)
    net.eval()

    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    net = torch.nn.DataParallel(net.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight,strict=False)
    loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )



if __name__ == '__main__':
    config = setting_config
    main(config)