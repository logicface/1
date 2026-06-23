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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1, 2, 3"

from utils import get_logger, log_config_info, set_seed, get_optimizer, get_scheduler
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = isic_loader(path_Data = config.data_path, train = True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    print(config.batch_size)
    val_dataset = isic_loader(path_Data = config.data_path, train = False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Models----------#')
    #model_cfg = config.model_config
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
    scaler = GradScaler()  # 使用新的 GradScaler





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    print('#----------Training----------#')
    global_step = 0
    start_time = time.time()  # 记录训练开始时间
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()  # 记录每个 epoch 开始时间
        # print('Epoch:', epoch)
        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            net,
            global_step,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )
        print("------------")
        loss = val_one_epoch(
            val_loader,
            net,
            model,
            criterion,
            epoch,
            logger,
            config
             )


        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))


        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 
        
        epoch_end_time = time.time()  # 记录每个 epoch 结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算每个 epoch 的持续时间
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

    total_training_time = time.time() - start_time  # 计算总训练时间
    print(f"Total training time: {total_training_time:.2f} seconds")

if __name__ == '__main__':
    config = setting_config
    main(config)