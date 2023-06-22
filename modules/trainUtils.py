

# from utils.utils import *
# from utils.distributed_utils import reduce_value, is_main_process

import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not np.isnan(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

def cal_metrics(labels, predicts):
     # 计算每个batch的混淆矩阵
    cm = np.array(confusion_matrix(labels, predicts, labels=[0, 1]))
    # 计算1类的召回率
    if 1 in labels:
        if cm.shape[0] > 1:
            recall_rate = round(cm[1,1] / (cm[1,0] + cm[1,1]), 5)
        else:
            recall_rate = 1.00
    else:
        recall_rate = np.nan

    # 计算 kappa 值
    kappa = cohen_kappa_score(labels, predicts)
    # 计算 F1 值
    f1 = f1_score(labels, predicts)
    return recall_rate, kappa, f1, cm

def train_one_epoch_1m(model, criterion, optimizer, dataloader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    accs = AverageMeter('acc', ':.4e')
    recalls = AverageMeter('recall_rate', ':.4e')
    f1s = AverageMeter('f1', ':.4e')
    kappas = AverageMeter('kappa', ':.4e')

    # 确保模型在训练前是处于training的模式
    model.train()

    # 这些列表用来记录训练中的信息
    end = time.time()
    for features, labels in tqdm(dataloader, desc="Train"):
        # measure data loading time
        data_time.update(time.time() - end)

        # 前向传播数据（确保数据和模型都在同一个设备上）
        logits = model(features.to(device))
        # 计算交叉熵损失.
        # 计算交叉熵损失前不需要使用softmax，因为它是自动完成的
        loss = criterion(logits, labels.to(device))

        # 更新各项指标
        accs.update((logits.argmax(dim=-1) == labels.to(device)).float().mean().cpu().item()) 
        losses.update(loss.cpu().item())
        recall_rate, kappa, f1, cm = cal_metrics(labels.cpu(), logits.argmax(dim=-1).cpu())
        recalls.update(recall_rate)
        f1s.update(f1)
        kappas.update(kappa)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 计算整个epoch的训练集平均loss和accuracy
    metrics = {
        losses.name: losses.avg,
        accs.name: accs.avg,
        recalls.name: recalls.avg,
        f1s.name: f1s.avg,
        kappas.name: kappas.avg

    }
    print(f"Train batch time: {batch_time.avg:.4f}, Train data time: {data_time.avg:.4f}")
    print(metrics)

    return metrics


def validate_1m(model, criterion, dataloader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    accs = AverageMeter('acc', ':.4e')
    recalls = AverageMeter('recall_rate', ':.4e')
    f1s = AverageMeter('f1', ':.4e')
    kappas = AverageMeter('kappa', ':.4e')
    # 确保模型现在处于验证的模式，这样模型中的一些模块比如dropout会被禁用
    model.eval()
    # model.cuda()
    # 这些列表用来记录训练中的信息

    # 验证中不需要计算梯度
    # 使用 torch.no_grad() 加速前向传播
    end = time.time()
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Val"):
            # measure data loading time
            data_time.update(time.time() - end)

            logits = model(features.to(device))
            # loss还是可以算的（但是不算梯度）
            loss = criterion(logits, labels.to(device))    

            # 更新各项指标
            accs.update((logits.argmax(dim=-1) == labels.to(device)).float().mean().cpu().item()) 
            losses.update(loss.cpu().item())
            recall_rate, kappa, f1, cm = cal_metrics(labels.cpu(), logits.argmax(dim=-1).cpu())
            recalls.update(recall_rate)
            f1s.update(f1)
            kappas.update(kappa)
            print(cm)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # 计算整个epoch的训练集平均loss和accuracy
    metrics = {
        losses.name: losses.avg,
        accs.name: accs.avg,
        recalls.name: recalls.avg,
        f1s.name: f1s.avg,
        kappas.name: kappas.avg

    }

    print(f"Val batch time: {batch_time.avg:.4f}, Val data time: {data_time.avg:.4f}")
    print(metrics)

    return metrics

def train_one_epoch_3m(model, criterion, optimizer, dataloader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    accs = AverageMeter('acc', ':.4e')
    recalls = AverageMeter('recall_rate', ':.4e')
    f1s = AverageMeter('f1', ':.4e')
    kappas = AverageMeter('kappa', ':.4e')

    # 确保模型在训练前是处于training的模式
    model.train()

    # 这些列表用来记录训练中的信息
    end = time.time()
    for images, sv_attrs, taxi_attrs, labels in tqdm(dataloader, desc="Train"):
        # measure data loading time
        data_time.update(time.time() - end)
        
        logits = model(images.to(device), sv_attrs.to(device), taxi_attrs.to(device))
 
        # 计算交叉熵损失.
        # 计算交叉熵损失前不需要使用softmax，因为它是自动完成的
        loss = criterion(logits, labels.to(device))

        # 更新各项指标
        accs.update((logits.argmax(dim=-1) == labels.to(device)).float().mean().cpu().item()) 
        losses.update(loss.cpu().item())
        recall_rate, kappa, f1, cm = cal_metrics(labels.cpu(), logits.argmax(dim=-1).cpu())
        recalls.update(recall_rate)
        f1s.update(f1)
        kappas.update(kappa)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 计算整个epoch的训练集平均loss和accuracy
    metrics = {
        losses.name: losses.avg,
        accs.name: accs.avg,
        recalls.name: recalls.avg,
        f1s.name: f1s.avg,
        kappas.name: kappas.avg

    }
    print(f"Train batch time: {batch_time.avg:.4f}, Train data time: {data_time.avg:.4f}")
    print(metrics)

    return metrics


def validate_3m(model, criterion, dataloader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    accs = AverageMeter('acc', ':.4e')
    recalls = AverageMeter('recall_rate', ':.4e')
    f1s = AverageMeter('f1', ':.4e')
    kappas = AverageMeter('kappa', ':.4e')
    # 确保模型现在处于验证的模式，这样模型中的一些模块比如dropout会被禁用
    model.eval()
    # model.cuda()
    # 这些列表用来记录训练中的信息

    # 验证中不需要计算梯度
    # 使用 torch.no_grad() 加速前向传播
    end = time.time()
    with torch.no_grad():
        for images, sv_attrs, taxi_attrs, labels in tqdm(dataloader, desc="Val"):
            # measure data loading time
            data_time.update(time.time() - end)

            logits = model(images.to(device), sv_attrs.to(device), taxi_attrs.to(device))
            # loss还是可以算的（但是不算梯度）
            loss = criterion(logits, labels.to(device))    

            # 更新各项指标
            accs.update((logits.argmax(dim=-1) == labels.to(device)).float().mean().cpu().item()) 
            losses.update(loss.cpu().item())
            recall_rate, kappa, f1, cm = cal_metrics(labels.cpu(), logits.argmax(dim=-1).cpu())
            recalls.update(recall_rate)
            f1s.update(f1)
            kappas.update(kappa)
            print(cm)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # 计算整个epoch的训练集平均loss和accuracy
    metrics = {
        losses.name: losses.avg,
        accs.name: accs.avg,
        recalls.name: recalls.avg,
        f1s.name: f1s.avg,
        kappas.name: kappas.avg

    }

    print(f"Val batch time: {batch_time.avg:.4f}, Val data time: {data_time.avg:.4f}")
    print(metrics)

    return metrics

def train_one_epoch_2m(model, criterion, optimizer, dataloader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    accs = AverageMeter('acc', ':.4e')
    recalls = AverageMeter('recall_rate', ':.4e')
    f1s = AverageMeter('f1', ':.4e')
    kappas = AverageMeter('kappa', ':.4e')

    # 确保模型在训练前是处于training的模式
    model.train()

    # 这些列表用来记录训练中的信息
    end = time.time()
    for f1, f2, labels in tqdm(dataloader, desc="Train"):
        # measure data loading time
        data_time.update(time.time() - end)
        
        logits = model(f1.to(device), f2.to(device))
 
        # 计算交叉熵损失.
        # 计算交叉熵损失前不需要使用softmax，因为它是自动完成的
        loss = criterion(logits, labels.to(device))

        # 更新各项指标
        accs.update((logits.argmax(dim=-1) == labels.to(device)).float().mean().cpu().item()) 
        losses.update(loss.cpu().item())
        recall_rate, kappa, f1, cm = cal_metrics(labels.cpu(), logits.argmax(dim=-1).cpu())
        recalls.update(recall_rate)
        f1s.update(f1)
        kappas.update(kappa)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # 计算整个epoch的训练集平均loss和accuracy
    metrics = {
        losses.name: losses.avg,
        accs.name: accs.avg,
        recalls.name: recalls.avg,
        f1s.name: f1s.avg,
        kappas.name: kappas.avg

    }
    print(f"Train batch time: {batch_time.avg:.4f}, Train data time: {data_time.avg:.4f}")
    print(metrics)

    return metrics


def validate_2m(model, criterion, dataloader, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    accs = AverageMeter('acc', ':.4e')
    recalls = AverageMeter('recall_rate', ':.4e')
    f1s = AverageMeter('f1', ':.4e')
    kappas = AverageMeter('kappa', ':.4e')
    # 确保模型现在处于验证的模式，这样模型中的一些模块比如dropout会被禁用
    model.eval()
    # model.cuda()
    # 这些列表用来记录训练中的信息

    # 验证中不需要计算梯度
    # 使用 torch.no_grad() 加速前向传播
    end = time.time()
    with torch.no_grad():
        for f1, f2, labels in tqdm(dataloader, desc="Val"):
            # measure data loading time
            data_time.update(time.time() - end)

            logits = model(f1.to(device), f2.to(device))
            # loss还是可以算的（但是不算梯度）
            loss = criterion(logits, labels.to(device))    

            # 更新各项指标
            accs.update((logits.argmax(dim=-1) == labels.to(device)).float().mean().cpu().item()) 
            losses.update(loss.cpu().item())
            recall_rate, kappa, f1, cm = cal_metrics(labels.cpu(), logits.argmax(dim=-1).cpu())
            recalls.update(recall_rate)
            f1s.update(f1)
            kappas.update(kappa)
            print(cm)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # 计算整个epoch的训练集平均loss和accuracy
    metrics = {
        losses.name: losses.avg,
        accs.name: accs.avg,
        recalls.name: recalls.avg,
        f1s.name: f1s.avg,
        kappas.name: kappas.avg

    }

    print(f"Val batch time: {batch_time.avg:.4f}, Val data time: {data_time.avg:.4f}")
    print(metrics)

    return metrics