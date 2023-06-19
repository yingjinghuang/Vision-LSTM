import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os, shutil

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

def plot_confusion_matrix(cm, classes, normalize=False, title='State transition matrix', cmap=plt.cm.Blues):
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
        

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    
    plt.ylabel('Self patt')
    plt.xlabel('Transition patt')
    
    plt.tight_layout()
    # plt.savefig('res/method_2.png', transparent=True, dpi=800) 
    
    plt.show()

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

def check_folder(folder, mode="create"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(folder, "removed!")
    if mode != "del":
        os.makedirs(folder)

class Logger():
    def __init__(self, path):
        self.path = path
    
    def log(self, *messages):
        with open(self.path, "a") as f:
            for message in messages:
                f.write(str(message))
                f.write(" ")
            f.write("\n")

def collate_fn_sv(batch):  
    # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    sv_data_list, label_list = zip(*batch)

    # 每一个sample [n, 3, 224, 224] 做padding都转成 [m, 3, 224, 224]，其中m是这个batch最长的序列
    # 最后得到的是 [batch_size, m, 3, 224, 224] (transpose 之后
    sv_data_tensor = pad_sequence(sv_data_list, batch_first=True) 
    # sv_data_tensor = sv_data_tensor.transpose(0,1)

    label_tensor = torch.LongTensor(label_list)
    data_copy = (sv_data_tensor, label_tensor)
    return data_copy

def collate_fn_end2end(batch):  
    # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    remote_data_list, sv_data_list, taxi_data_list, label_list = zip(*batch)

    # 每一个sample [n, 3, 224, 224] 做padding都转成 [m, 3, 224, 224]，其中m是这个batch最长的序列
    # 最后得到的是 [batch_size, m, 3, 224, 224] (transpose 之后
    sv_data_tensor = pad_sequence(sv_data_list) 
    sv_data_tensor = sv_data_tensor.transpose(0,1)

    remote_data_tensor = torch.stack(remote_data_list)
    taxi_data_tensor = torch.stack(taxi_data_list)
    label_tensor = torch.LongTensor(label_list)
    return remote_data_tensor, sv_data_tensor, taxi_data_tensor, label_tensor

def collate_fn_end2end2(batch):  
    # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    f1_list, f2_list, label_list = zip(*batch)
    
    if f1_list[0].shape[0]==3:
        f1_tensor = torch.stack(f1_list)
        
        if f2_list[0].shape[0] > 3:
        
            # 每一个sample [n, 3, 224, 224] 做padding都转成 [m, 3, 224, 224]，其中m是这个batch最长的序列
            # 最后得到的是 [batch_size, m, 3, 224, 224] (transpose 之后
            sv_data_tensor = pad_sequence(f2_list) 
            f2_tensor = sv_data_tensor.transpose(0,1)
        else:
            f2_tensor = torch.stack(f2_list)
    else:
        sv_data_tensor = pad_sequence(f1_list) 
        f1_tensor = sv_data_tensor.transpose(0,1)
        
        f2_tensor = torch.stack(f2_list)

    label_tensor = torch.LongTensor(label_list)
    return f1_tensor, f2_tensor, label_tensor

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