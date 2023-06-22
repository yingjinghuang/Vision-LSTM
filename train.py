import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
import pandas as pd
from collections import Counter
import imblearn

from modules.utils import *
from modules.datasets import *
from modules.models import *
from modules.trainUtils import *
from configs import *


# python grid_multi_feature.py multi_sum_weighted_concat --gpu 1 --mode sum --imbalance weighted --fusion concat

def main(configs):
    device = torch.device("cuda")

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]= configs.gpu

    models_path = os.path.join(configs.weights_folder, configs.model_name)
    tensorboard_path = os.path.join(configs.tensorboard_folder, configs.model_name)
    logs_path = os.path.join(configs.log_folder, configs.model_name + ".log")

    check_folder(models_path)
    check_folder(tensorboard_path)
    check_folder(configs.log_folder)

    logger = Logger(logs_path)
    logger.log(configs)

    writer = tensorboard.SummaryWriter(tensorboard_path + "/")
    
    data_df = pd.read_pickle(configs.model_data_path)
    taxi_features_col = list(data_df.columns[2:-2])
    # Columns: GID,features,[taxi_features],label,mode
    logger.log("data loaded!")
    logger.log(data_df[["label", "mode"]].value_counts())

    # 数据集构建
    data_df_train = data_df[data_df["mode"] == "train"]
    data_df_val = data_df[data_df["mode"] == "val"]

    x_train = data_df_train.iloc[:, :-2]
    y_train = data_df_train["label"].tolist()
    x_val = data_df_val.iloc[:, :-2]
    y_val = data_df_val["label"].tolist()

    logger.log("data prepared done.")

    
    counter = Counter(data_df_train['label'].to_list())
    # 数据重采样
    oversampler = imblearn.over_sampling.RandomOverSampler()
    x_train, y_train = oversampler.fit_resample(x_train, y_train)

    if configs.modality_count == 3:
        if configs.mode == "single":
            train_dataset = MultiDataset1SV(x_train["GID"].to_numpy(), x_train["features"].to_numpy(), x_train[taxi_features_col].to_numpy(), y_train, configs.rs_path)
            val_dataset = MultiDataset1SV(x_val["GID"].to_numpy(), x_val["features"].to_numpy(), x_val[taxi_features_col].to_numpy(), y_val, configs.rs_path, mode='valid')
            model = MultiFeature1SV()
        else:
            train_dataset = MultiDataset(x_train["GID"].to_numpy(), x_train["features"].to_numpy(), x_train[taxi_features_col].to_numpy(), y_train, configs.rs_path)
            val_dataset = MultiDataset(x_val["GID"].to_numpy(), x_val["features"].to_numpy(), x_val[taxi_features_col].to_numpy(), y_val, configs.rs_path, mode='valid')
            model = MultiFeature(mode=configs.mode)
        collate_fn = collate_fn_end2end
        
    elif configs.modality_count == 2:
        train_dataset = TwoDataset(x_train["GID"].to_numpy(), x_train["features"].to_numpy(), x_train[taxi_features_col].to_numpy(), y_train, configs.rs_path, configs.modalities)
        val_dataset = TwoDataset(x_val["GID"].to_numpy(), x_val["features"].to_numpy(), x_val[taxi_features_col].to_numpy(), y_val, configs.rs_path, configs.modalities, mode='valid')
        if "sv" in configs.modalities:
            collate_fn = collate_fn_end2end2
        else:
            collate_fn = None
        model = TwoFeature(mode=configs.mode, modal=configs.modalities)
    else:
        if "remote" in configs.modalities:
            train_dataset = RemoteData(x_train["GID"].to_numpy(), y_train, configs.rs_path)
            val_dataset = RemoteData(x_val["GID"].to_numpy(), y_val, configs.rs_path, mode='valid')
            collate_fn = None
            model = RemoteNet()
        elif "sv" in configs.modalities:
            train_dataset = SVFeatureDataset(x_train["features"].to_numpy(), y_train)
            val_dataset = SVFeatureDataset(x_val["features"].to_numpy(), y_val, mode='valid')
            collate_fn = collate_fn_sv
            model = SVFeature(mode=configs.mode)
        else:
            train_dataset = TaxiDataset(x_train[taxi_features_col].to_numpy(), y_train)
            val_dataset = TaxiDataset(x_val[taxi_features_col].to_numpy(), y_val, mode='valid')
            collate_fn = None
            model = TaxiNet()

    logger.log("dataset prepared done.")
    logger.log("Train Dataset size: ", len(train_dataset), "Validation Dataset size: ", len(val_dataset))

    # 定义data loader
    if collate_fn is None:
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=configs.batch_size, 
                shuffle=True,
                num_workers=configs.workers,
                pin_memory=True
            )

        val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=configs.batch_size, 
                shuffle=False,
                num_workers=configs.workers,
                pin_memory=True
            )
    else:
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=configs.batch_size, 
                shuffle=True,
                num_workers=configs.workers,
                pin_memory=True,
                collate_fn=collate_fn
            )

        val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=configs.batch_size, 
                shuffle=False,
                num_workers=configs.workers,
                pin_memory=True,
                collate_fn=collate_fn
            )

    
    model = model.to(device)
    
    # 使用加权交叉熵
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1/counter[0], 1/counter[1]]).cuda()) # 加权交叉熵
    
    # 初始化一个优化器，我们可以自行调节一些超参数进行微调，比如说学习率
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=configs.lr, weight_decay=configs.wd, momentum=0.9)
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=configs.lr_p, factor=configs.lr_f, verbose=True)

    best_loss = 500.0

    for epoch in range(configs.epochs):
         # ---------- Training ----------
        if configs.modality_count == 3:
            train_metrics = train_one_epoch_3m(model, criterion, optimizer, train_loader, device)
        elif configs.modality_count == 2:
            train_metrics = train_one_epoch_2m(model, criterion, optimizer, train_loader, device)
        else:
            train_metrics = train_one_epoch_1m(model, criterion, optimizer, train_loader, device)
        
        # ---------- Validation ----------
        if configs.modality_count == 3:
            val_metrics = train_one_epoch_3m(model, criterion, val_loader, device)
        elif configs.modality_count == 2:
            val_metrics = train_one_epoch_2m(model, criterion, val_loader, device)
        else:
            val_metrics = train_one_epoch_1m(model, criterion, val_loader, device)
        
        
        scheduler.step()
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
        logger.log(f"[ Train | {epoch + 1:03d}/{configs.epochs:03d} ] {train_metrics_str}\n[ Val | {epoch + 1:03d}/{configs.epochs:03d} ] {val_metrics_str}")
        print(f"[ Train | {epoch + 1:03d}/{configs.epochs:03d} ] {train_metrics_str}\n[ Val | {epoch + 1:03d}/{configs.epochs:03d} ] {val_metrics_str}")

        # save model
        # if val_metrics["loss"] < best_loss:
            # best_loss = val_metrics["loss"]
        savepath = os.path.join(models_path, f'model_epoch{epoch+1:03d}_{val_metrics["acc"]:.3f}.pth.tar')
        torch.save(model.state_dict(), savepath)
            # logger.log(f'\n\t*** Saved checkpoint in {savepath} ***\n')
        
    writer.close()

if __name__ == "__main__":
    main(configs)