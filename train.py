import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from collections import Counter
import os, time
from tqdm import tqdm
import argparse
from torchinfo import summary

# from config import *
from allgrid_config import *
import sys
sys.path.append(project_folder)

from utils.utils import *
from utils.train_val_utils import *
from utils.datasets import MultiDataset
from utils.model import MultiFeature, MultiFeaturev2
import imblearn

# python grid_multi_feature.py multi_sum_weighted_concat --gpu 1 --mode sum --imbalance weighted --fusion concat

def main(args):
    device = torch.device("cuda")

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    models_path = os.path.join(weights_folder, args.foldername)
    tensorboard_path = os.path.join(tensorboard_folder, args.foldername)

    check_folder(models_path)
    check_folder(tensorboard_path)

    logger = Logger(os.path.join(models_path, "log.log"))
    logger.log(args)

    writer = tensorboard.SummaryWriter(tensorboard_path + "/")
    
    data_df = pd.read_pickle(model_data_path)
    # sv_df = pd.read_csv(sv_feature_csv_path, dtype = {'GID': str}, index_col=0)
    # taxi_df = pd.read_csv(taxi_final_csv_path, dtype = {'GID': str}, index_col=0)
    taxi_features_col = list(data_df.columns[2:-3])
    # labels= pd.read_csv(labels_csv_path, dtype = {'GID': str, 'label': int}, index_col=0)
    # data_df = sv_df.join(taxi_df.set_index("GID"), on="GID")
    # data_df = data_df.join(labels.set_index("GID"), on="GID")

    # Columns: GID,features,[taxi_features],ratio,label

    print("data loaded!")

    # 训练集测试集划分
    # val_index = data_df.sample(frac=args.valid_ratio).index.tolist()
    # data_df["mode"] = "train"
    # data_df.loc[val_index, "mode"] = "val"
    logger.log(data_df[["label", "mode"]].value_counts())

    # 数据集构建
    data_df_train = data_df[data_df["mode"] == "train"]
    data_df_val = data_df[data_df["mode"] == "val"]

    x_train = data_df_train.iloc[:, :-2]
    y_train = data_df_train["label"].tolist()
    x_val = data_df_val.iloc[:, :-2]
    y_val = data_df_val["label"].tolist()

    logger.log("data prepared done.")

    
    counter = Counter(data_df['label'].to_list())
    # 数据重采样
    if args.imbalance == "resample":
        oversampler = imblearn.over_sampling.RandomOverSampler()
        x_train, y_train = oversampler.fit_resample(x_train, y_train)

    train_dataset = MultiDataset(x_train["GID"].to_numpy(), x_train["features"].to_numpy(), x_train[taxi_features_col].to_numpy(), y_train, image_dir)
    val_dataset = MultiDataset(x_val["GID"].to_numpy(), x_val["features"].to_numpy(), x_val[taxi_features_col].to_numpy(), y_val, image_dir, mode='valid')
    logger.log("dataset prepared done.")
    logger.log("Train Dataset size: ", len(train_dataset), "Validation Dataset size: ", len(val_dataset))

    # 定义data loader
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn_end2end
        )

    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn_end2end
        )

    if args.fusion == "concat":
        model = MultiFeature(mode=args.mode)
    else:
        model = MultiFeaturev2(mode=args.mode)
    # logger.log(summary(model, ((2,3,224,224), (2, 50, 512), (2, 340))))
    
    model = model.to(device)
    
    # 分类问题使用交叉熵损失
    if args.imbalance == "weighted":
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1/counter[0], 1/counter[1]]).cuda()) # 加权交叉熵
    else:
        criterion = nn.CrossEntropyLoss()
    # 初始化一个优化器，我们可以自行调节一些超参数进行微调，比如说学习率
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    # 学习率衰减
    if not args.warmup:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor, verbose=True)
    else:
        t = 10 # warmup
        T = args.epochs # epochs - 10 为 cosine rate
        lr_rate = 0.0035
        n_t = 0.5
        # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) /2 ) * (1 - args.lr_factor) + args.lr_factor # cosine
        lf = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (1 + math.cos(math.pi * (epoch - t) / (T - t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_loss = 500.0

    for epoch in range(args.epochs):
         # ---------- Training ----------
        train_metrics = train_one_epoch_multi_singleGPU(model, criterion, optimizer, train_loader, device)
        
        # ---------- Validation ----------
        val_metrics = validate_multi_singleGPU(model, criterion, val_loader, device)
        
        if args.warmup:
            scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])

        # ============= tensorboard =============
        writer.add_scalar('Loss/Train',train_metrics["loss"], epoch)
        writer.add_scalar('Accuracy/Train',train_metrics["acc"], epoch)
        writer.add_scalar('Recall_Rate/Train',train_metrics["recall_rate"], epoch)
        writer.add_scalar('Kappa/Train',train_metrics["kappa"], epoch)
        writer.add_scalar('F1/Train',train_metrics["f1"], epoch)
        writer.add_scalar('Loss/Val',val_metrics["loss"], epoch)
        writer.add_scalar('Accuracy/Val',val_metrics["acc"], epoch)
        writer.add_scalar('Recall_Rate/Val',val_metrics["recall_rate"], epoch)
        writer.add_scalar('Kappa/Val',val_metrics["kappa"], epoch)
        writer.add_scalar('F1/Val',val_metrics["f1"], epoch)

        train_metrics_str = ", ".join([f"{k} = {v:.5f}" for k, v in train_metrics.items()])
        val_metrics_str = ", ".join([f"{k} = {v:.5f}" for k, v in val_metrics.items()])
        logger.log(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] {train_metrics_str}\n[ Val | {epoch + 1:03d}/{args.epochs:03d} ] {val_metrics_str}")
        print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] {train_metrics_str}\n[ Val | {epoch + 1:03d}/{args.epochs:03d} ] {val_metrics_str}")

        # save model
        # if val_metrics["loss"] < best_loss:
            # best_loss = val_metrics["loss"]
        savepath = os.path.join(models_path, f'model_epoch{epoch+1:03d}_{val_metrics["acc"]:.3f}.pth.tar')
        torch.save(model.state_dict(), savepath)
            # logger.log(f'\n\t*** Saved checkpoint in {savepath} ***\n')
        
    writer.close()

def get_parameters():
    parser = argparse.ArgumentParser(description='PyTorch Multi Feature Classifier')

    parser.add_argument('foldername', metavar='DIR',
                        help='folder to save weights and tensorboard')

    parser.add_argument('--gpu', default= '0,1,2,3', type=str,
                        help='GPU id(s) to use.'
                            'Format: a comma-delimited list')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--epochs', default=100, type=int,
                        help='num of epochs')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', '--weight-decy', default=1e-4, type=float,
                        metavar='WD', help='initial weight decay', dest='wd')                    
    parser.add_argument('--lr-patience', default=10, type=int, help='patience of learning rate')
    parser.add_argument('--lr-factor', default=0.1, type=float, help='factor of learning rate')
    parser.add_argument('--valid-ratio', default=0.2, type=float, help='the ratio of validation dataset')
    parser.add_argument('--workers', default=6, type=int, help='number of workers')

    parser.add_argument('--mode', default= 'mean', type=str,
                    help='the mode to decomposite street view'
                         'Format: mean or pca')
    parser.add_argument('--imbalance', default= 'weighted', type=str,
                    help='weighted, resample')
    parser.add_argument('--fusion', default= 'concat', type=str,
                    help='concat, attention')
    parser.add_argument('--warmup', default= False, type=bool,
                    help='Learning rate warmup setting')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parameters()
    main(args)