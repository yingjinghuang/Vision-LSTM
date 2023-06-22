from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

################################################################################
################################ TAXI Part #####################################
################################################################################

class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_layers, lstm_hs, dropout=0.8, attention=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # input is of the form (batch_size, num_layers, time_steps), e.g. (128, 1, 512)
        x = torch.transpose(x, 0, 1)
        # lstm layer is of the form (num_layers, batch_size, time_steps)
        x, (h_n, c_n) = self.lstm(x)
        # dropout layer input shape (Sequence Length, Batch Size, Hidden Size * Num Directions)
        y = self.dropout(x)
        # output shape is same as Dropout intput
        return y

class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()
    def forward(self, x):
        # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = self.conv(x)
        # input (batch_size, out_channel, L_out)
        x = self.batch_norm(x)
        # same shape as input
        y = self.relu(x)
        return y

class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 128, 256, 256], kernels=[8, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        return y

class LSTMFCNBlock(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=256, channels=[1, 128, 256, 256]):
        super(LSTMFCNBlock, self).__init__()
        # self.lstm_block = BlockLSTM(time_steps, 1, lstm_hs)
        self.lstm_block = BlockLSTM(int(time_steps/2), 2, int(lstm_hs/2))
        self.fcn_block = BlockFCN(time_steps, channels=channels)

    def forward(self, x):
        # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
        x1 = x
        x = x.unsqueeze(1)
        # pass input through LSTM block
        # x1 = x
        x1_1, x1_2 = x1.split(170, dim=-1)
        x1 = torch.stack([x1_1, x1_2], dim=1) # (batch_size, 2, 170)
        
        x1 = self.lstm_block(x1)
        # x1 = self.lstm_block(x)
        # x1 = torch.squeeze(x1)
        x1 = torch.transpose(x1, 0, 1)
        x1 = torch.flatten(x1, start_dim=1, end_dim=2)
        # pass input through FCN block
        x2 = self.fcn_block(x)
        x2 = torch.squeeze(x2, dim=-1)
        # concatenate blocks output
        x = torch.cat([x1, x2], 1)
        return x

class TaxiNet(nn.Module):
    def __init__(self):
        super(TaxiNet, self).__init__()
        self.backbone = LSTMFCNBlock()

        self.fc = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        
        return x
################################################################################
################################ SV Part #####################################
################################################################################

class DensenetEncoder(nn.Module):
    def __init__(self):
        super(DensenetEncoder, self).__init__()
        densnet = models.densenet121(pretrained=True)
        self.feature = densnet.features
        self.classifier = nn.Sequential(*list(densnet.classifier.children())[:-1])
        pretrained_dict = densnet.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.avg = nn.AvgPool2d(7, stride=1)
 
    def forward(self, x):
        output = self.feature(x)
        output = self.avg(output)
        return output

class VSFCnet(nn.Module):
    def __init__(self):
        super(VSFCnet, self).__init__()
        self.hidden1 = nn.Sequential(
                nn.Linear(in_features=512, out_features=256, bias=True),
                nn.ReLU())
        self.hidden2 = nn.Sequential(
                nn.Linear(in_features=256, out_features=64, bias=True),
                nn.ReLU())
        self.hidden3 = nn.Linear(64,2)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return x

class SVEnd2endBlock(nn.Module):
    def __init__(self, mode='mean'):
        super(SVEnd2endBlock, self).__init__()
        self.mode = mode

        model = models.__dict__["resnet18"](num_classes=365)
        checkpoint = torch.load('/home/huangyj/urban_village/utils/resnet18_places365.pth.tar')  # map_location=lambda storage, loc:storage
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)

        # layer = model._modules.get("avgpool")
        #保存feature部分  去掉fc层
        model = torch.nn.Sequential(*(list(model.children())[:-1])) # 提取512维特征
        self.feature_extrator = model

        if mode == 'lstm':
            self.lstm = nn.LSTM(512, 512, 1, batch_first=True)
        
    def forward(self, x):
        b, n, c, h, w = x.size()  # (batch, num, channal, height, width)
        
        x_list = []
        for sample in range(b): # 拆解 batch 这个维度
            # b,n,c,h,w -> n,c,h,w
            x_tmp = x[sample]
            # n,c,h,w -> n_valid,c,h,w
            x_tmp = x_tmp[x_tmp.nonzero(as_tuple=True)].view(-1, x_tmp.shape[1], x_tmp.shape[2], x_tmp.shape[3]) # 去掉0的padding
            x_tmp = self.feature_extrator(x_tmp) # 得到 (num, n, 1, 1)
            x_tmp = x_tmp.view(-1, 512)
            
            if self.mode == "mean":
                x_tmp = torch.mean(x_tmp, dim=0) # 按列求均值，得到(512)
            elif self.mode == "pca":
                x_tmp, s, v = torch.pca_lowrank(x_tmp.T, 1)  # 按 PCA 降维，得到(512,1)
                x_tmp = torch.squeeze(x_tmp) # 得到 (512)
            elif self.mode == "max":
                x_tmp = torch.max(x_tmp, dim=0).values # 按列求max
            elif self.mode == "lstm":
                x_tmp, (h_n, c_n) = self.lstm(x_tmp.view(1, -1, 512)) # 输入 lstm 需要加上 batch 这个维度
                x_tmp = x_tmp[:,-1,:] # 只取最后一层的输出 # （1,1,512)
                x_tmp = torch.squeeze(x_tmp) # 得到 (512)
            elif self.mode == "vit":
                x_tmp = self.vit(x_tmp.view(1, -1, 512)) # 输入 lstm 需要加上 batch 这个维度
                x_tmp = torch.squeeze(x_tmp) # 得到 (512)
            else:
                pass
        
            x_list.append(x_tmp)
        
        x = torch.stack(x_list) # 拼接,(batch,512)
        return x

class SVFeatureBlock(nn.Module):
    def __init__(self, mode='mean'):
        super(SVFeatureBlock, self).__init__()
        self.mode = mode

        if mode == 'lstm':
            self.lstm = nn.LSTM(512, 512, 1, batch_first=True)
    
    def forward(self, x):
        b, c, f = x.size() # batch, channnel, feature

        x_list = []
        for sample in range(b): # 拆解 batch 这个维度
            # b,c,f -> c,f
            x_tmp = x[sample]
            # c,f -> c_valid, f
            x_tmp = x_tmp[x_tmp.nonzero(as_tuple=True)].view(-1, 512) # 去掉0的padding
            
            if self.mode == "mean":
                x_tmp = torch.mean(x_tmp, dim=0) # 按列求均值，得到(512)
            elif self.mode == "pca":
                x_tmp, s, v = torch.pca_lowrank(x_tmp.T, 1)  # 按 PCA 降维，得到(512,1)
                x_tmp = torch.squeeze(x_tmp) # 得到 (512)
            elif self.mode == "sum":
                x_tmp = torch.sum(x_tmp, dim=0) # 按列求和
            elif self.mode == "max":
                x_tmp = torch.max(x_tmp, dim=0).values # 按列求max
            elif self.mode == "lstm":
                x_tmp, (h_n, c_n) = self.lstm(x_tmp.view(1, -1, 512)) # 输入 lstm 需要加上 batch 这个维度
                x_tmp = x_tmp[:,-1,:] # 只取最后一层的输出 # （1,1,512)
                x_tmp = torch.squeeze(x_tmp) # 得到 (512)
            else:
                pass
        
            x_list.append(x_tmp)
        
        x = torch.stack(x_list) # 拼接,(batch,512)
        return x

class SVEnd2end(nn.Module):
    def __init__(self, mode='mean'):
        super(SVEnd2end, self).__init__()
        self.backbone = SVEnd2endBlock(mode)

        self.fc = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        
        return x

class SVFeature(nn.Module):
    def __init__(self, mode='mean'):
        super(SVFeature, self).__init__()
        self.backbone = SVFeatureBlock(mode)

        self.fc = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        
        return x

################################################################################
############################### Remote Part ####################################
################################################################################

class RemoteNet(nn.Module):
    def __init__(self, ratio=0.5, model_name='densenet'):
        super(RemoteNet, self).__init__()
        self.model_name = model_name
        if model_name == "densenet":
            self.backbone = DensenetEncoder()
            self.fc = nn.Sequential(
                nn.Linear(1024,512),
                nn.ReLU(True),
                nn.Dropout(ratio),
                nn.Linear(512, 2)
            )
        else:
            self.backbone = models.resnet18(pretrained=False)
            #保存feature部分  去掉fc层
            self.backbone = nn.Sequential(*(list(self.backbone.children())[:-1]))
            self.fc = nn.Sequential(
                nn.Linear(512,256),
                nn.ReLU(True),
                nn.Dropout(ratio),
                nn.Linear(256, 2)
            )
        
    def forward(self,x):
        x = self.backbone(x)
        x = torch.squeeze(x)
        if self.model_name == "densenet":
            x = x.view(-1, 1024)
        else:
            x = x.view(-1, 512)
        x = self.fc(x)
        return x

################################################################################
############################### Remote Part ####################################
################################################################################

################################################################################
############################### Multi Part  ####################################
################################################################################

class MultiFeature1SV(nn.Module):
    def __init__(self):
        super(MultiFeature1SV, self).__init__()
        # Remote branch
        model1 = models.resnet18(num_classes=2)
        self.remote_backbone = torch.nn.Sequential(*(list(model1.children())[:-1])) # 提取512维特征

        # SV branch
        model2 = models.resnet18(num_classes=2)
        self.sv_backbone =  torch.nn.Sequential(*(list(model2.children())[:-1])) # 提取512维特征

        # Taxi branch
        self.taxi_backbone = LSTMFCNBlock(time_steps=340, num_variables=2)
        
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512 + 512 + 512, 2)
        )

    def forward(self, image, sv, taxi):
        # Remote branch
        x1 = self.remote_backbone(image) # 得到 (batch, 512)
        x1 = x1.view(-1, 512)

        # SV branch
        x2 = self.sv_backbone(sv)
        x2 = x2.view(-1, 512)

        # taxi branch
        x3 = self.taxi_backbone(taxi)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x

class MultiFeature(nn.Module):
    def __init__(self, mode='mean'):
        super(MultiFeature, self).__init__()
        # Remote branch
        model1 = models.resnet18(num_classes=2)
        self.remote_backbone = torch.nn.Sequential(*(list(model1.children())[:-1])) # 提取512维特征

        # SV branch
        self.sv_backbone = SVFeatureBlock(mode)

        # Taxi branch
        self.taxi_backbone = LSTMFCNBlock(time_steps=340, num_variables=2)
        
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512 + 512 + 512, 2)
        )

    def forward(self, image, sv, taxi):
        # Remote branch
        x1 = self.remote_backbone(image) # 得到 (batch, 512)
        x1 = x1.view(-1, 512)

        # SV branch
        x2 = self.sv_backbone(sv)

        # taxi branch
        x3 = self.taxi_backbone(taxi)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x


################################################################################
############################### Multi Part  ####################################
################################################################################

################################################################################
########################## two modal Part  #####################################
################################################################################

class TwoFeature(nn.Module):
    def __init__(self, mode='mean', modal=["remote", "sv"]):
        super(TwoFeature, self).__init__()
        
        self.modal = modal
        
        # Remote branch
        model1 = models.resnet18(num_classes=2)
        self.remote_backbone = torch.nn.Sequential(*(list(model1.children())[:-1])) # 提取512维特征

        # SV branch
        self.sv_backbone = SVFeatureBlock(mode)

        # Taxi branch
        self.taxi_backbone = LSTMFCNBlock(time_steps=340, num_variables=2)
        
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512 + 512, 2)
        )

    def forward(self, f1, f2):
        if "remote" in self.modal:
            # Remote branch
            x1 = self.remote_backbone(f1) # 得到 (batch, 512)
            x1 = x1.view(-1, 512)
            
            if "sv" in self.modal:
                # SV branch
                x2 = self.sv_backbone(f2)
            else:
                # taxi branch
                x2 = self.taxi_backbone(f2)
        else:
            # SV branch
            x1 = self.sv_backbone(f1)

            # taxi branch
            x2 = self.taxi_backbone(f2)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x