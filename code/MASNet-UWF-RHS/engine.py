import numpy as np
from tqdm import tqdm
import torch
from torch.cuda import amp
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from sklearn.preprocessing import normalize

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)

def train_one_epoch(train_loader,
                    net,
                    global_step,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        with amp.autocast(enabled=use_fp16):
            # print(images.shape)
            output = model(images)
            loss2u = net(F.sigmoid(output), targets)
            # print("1",loss2u.shape,output.shape, targets.shape)
            loss1u = structure_loss(output, targets)
            loss = loss1u + 0.1 * loss2u
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter %10 == 0:
            # print('%s | step:%d/%d/%d | lr=%.6f | loss1u=%.6f | loss2u=%.6f '%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss1u.item(), loss2u.item()))
            log_info = f'train: epoch {epoch}, iter:{iter}, loss1u: {loss1u.item():.6f}, loss2u: {loss2u.item():.6f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()

def val_one_epoch(test_loader,
                  net,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    loss1u_list = []
    loss2u_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            # print(f" Model output min: {out.min().item()}, max: {out.max().item()}")
        
            loss2u = net(F.sigmoid(out), msk)
            loss1u = structure_loss(out, msk)
            loss = loss1u + 0.1 * loss2u

            #out = model(img)
            #loss = criterion(out, msk)
            loss_list.append(loss.item())
            loss1u_list.append(loss1u.item())
            loss2u_list.append(loss2u.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
    # print(np.max(preds), np.max(gts),np.min(preds), np.min(gts))
    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        

        preds = np.array(preds)
        preds = preds / np.max(preds)  # 最大值归一化
        preds = np.where(preds>=config.threshold, 1, 0)
        y_pre = np.where(preds>=0.5 , 1, 0)
        # y_pre = np.where(preds >= (np.max(preds)+np.min(preds))/2, 1, 0)
        # y_pre = np.where(preds/np.max(preds) >= config.threshold, 1, 0)
        # print("y_pre", y_pre)
        y_true = np.where(gts >= 0.5, 1, 0)
        # print(np.max(y_pre), np.max(gts), np.min(y_pre), np.min(gts),(np.max(preds)+np.min(preds))/2)
        print(f"y_pre counts - 0: {np.count_nonzero(y_pre == 0)}, 1: {np.count_nonzero(y_pre == 1)}")
        print(f"y_true counts - 0: {np.count_nonzero(y_true == 0)}, 1: {np.count_nonzero(y_true == 1)}")


        confusion = confusion_matrix(y_true, y_pre)
        
        # 打印混淆矩阵和其形状
        # print("Confusion Matrix:")
        # print(confusion)
        # print("Shape:", confusion.shape)
        # 确保混淆矩阵的大小是 2x2
        if confusion.shape == (2, 2):
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
            print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
        else:
            print("Error: Confusion matrix is not 2x2.")
            pass
        # TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.6f}, loss1u: {np.mean(loss1u_list):.6f}, loss2u: {np.mean(loss2u_list):.6f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            # print(torch.max(img),torch.min(img))
            out = model(img)
            # print(out)
            msk = msk.squeeze(1).cpu().detach().numpy()
            # print(f"Model output min: {msk.min().item()}, max: {msk.max().item()}")
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            # print(out)
            preds.append(out)
            save_imgs(img, msk, out, i, config.work_dir , config.datasets, config.threshold,
                        test_data_name=test_data_name)

        # preds = np.array(preds).reshape(-1)
        # gts = np.array(gts).reshape(-1)

        # y_pre = np.where(preds >= config.threshold, 1, 0)
        # y_true = np.where(gts >= 0.5, 1, 0)
        # confusion = confusion_matrix(y_true, y_pre)
        # print("Confusion matrix shape:", confusion.shape,y_true.shape,y_pre.shape)
        # print("Confusion matrix:", confusion)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        preds = np.array(preds)
        preds = preds / np.max(preds)  # 最大值归一化
        # print(np.max(preds), np.max(gts),np.min(preds), np.min(gts))
        # y_pre = np.where(preds/np.max(preds) >= config.threshold, 1, 0)
        y_pre = np.where(preds >= config.threshold, 1, 0)       # y_pre[-1] = 1
        # y_pre[-1] = 1
        y_true = np.where(gts > 0.5, 1, 0)
        # print(np.max(y_pre), np.max(gts), np.min(y_pre), np.min(gts),(np.max(preds)+np.min(preds))/2)
        print(f"y_pre counts - 0: {np.count_nonzero(y_pre == 0)}, 1: {np.count_nonzero(y_pre == 1)}")
        print(f"y_true counts - 0: {np.count_nonzero(y_true == 0)}, 1: {np.count_nonzero(y_true == 1)}")
        # 确保 y_true 和 y_pre 的形状一致
        if y_true.shape != y_pre.shape:
            raise ValueError(f"Inconsistent shapes: y_true shape {y_true.shape}, y_pre shape {y_pre.shape}")

        confusion = confusion_matrix(y_true, y_pre)
        # print("Confusion matrix shape:", confusion.shape)
        # print("Confusion matrix:", confusion)

        # 确保 confusion 矩阵的大小是 2x2
        if confusion.shape == (2, 2):
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
        else:
            raise ValueError("Confusion matrix shape is not 2x2")

        # 打印混淆矩阵
        print("Confusion Matrix:", confusion)
       
        # print("Confusion Matrix:")
        # TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model,miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return f1_or_dsc