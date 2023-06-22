import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os, sys
import numpy as np

Image.MAX_IMAGE_PIXELS = 2300000000

# 继承pytorch的dataset，创建自己的
class RemoteData(Dataset):
    def __init__(self, img_list, label_list, file_path, resize_size=(224,224), mode='train'):
        self.img_list = img_list
        self.label_list = label_list
        self.file_path = file_path
        self.resize_size = resize_size
        self.mode = mode
                
    def __getitem__(self, index):
        # 从image_list中得到索引对应的文件路径
        img_path = self.img_list[index][0]
        
        # 读取图像文件
        image = Image.open(os.path.join(self.file_path, img_path + ".tif"))
        
        # 设置好需要转换的变量，还可以包括一系列的 normalize 等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，选择一个概率
                transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转，选择一个概率
                transforms.RandomRotation(degrees=180), # 随机旋转 
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
            ])
        else:
            # 测试和验证不做数据增强
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
            ])
        
        image = transform(image)
        
        label = self.label_list[index]
        sample = (image, label)
        return sample
    
    def __len__(self):
        return len(self.img_list)

class TaxiDataset(Dataset):
    def __init__(self, taxi_list, label_list, mode='train'):
        self.taxi_list = taxi_list
        self.label_list = label_list
        self.mode = mode

    def __getitem__(self, index):
        ## 处理 Taxi
        taxi_attrs = self.taxi_list[index]
        taxi_attrs = torch.FloatTensor(taxi_attrs)

        label = self.label_list[index]
        sample = (taxi_attrs, label)
        return sample
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.label_list)

class SVFeatureDataset(Dataset):
    def __init__(self, feature_list, label_list, mode='train'):
        self.feature_list = feature_list
        self.label_list = label_list
        self.mode = mode

    def __getitem__(self, index):
        attrs = self.feature_list[index][0]
        attrs = [float(x) if float(x)!=0 else 0.0000001 for x in attrs.split(",")]
        attrs = torch.FloatTensor(attrs)
        attrs = attrs.view(-1, 512)

        label = self.label_list[index]
        sample = (attrs, label)
        return sample
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.label_list)

class SVDataset(Dataset):
    def __init__(self, img_list, label_list, resize_size=(224,224), mode='train'):
        self.img_list = img_list
        self.label_list = label_list
        self.resize_size = resize_size
        self.mode = mode
        
    def __getitem__(self, index):
        # 从image_list中得到索引对应的文件路径
        img_path = self.img_list[index]
              
        # 读取图像文件
        image = Image.open(img_path)
        
        # 设置好需要转换的变量，还可以包括一系列的 normalize 等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.Resize(self.resize_size),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # 测试和验证不做数据增强
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        image = transform(image)
        image = torch.where(image==0, 0.0000001, image)
        
        label = self.label_list[index]
        sample = (image, label)
        return sample
    
    def __len__(self):
        return len(self.img_list)

class MultiDataset(Dataset):
    def __init__(self, img_list, sv_list, taxi_list, label_list, image_dir_path, resize_size=(224, 224), mode='train'):
        self.img_list = img_list
        self.sv_list = sv_list
        self.taxi_list = taxi_list
        self.label_list = label_list
        self.image_dir_path = image_dir_path
        self.resize_size = resize_size
        self.mode = mode
    
    def __getitem__(self, index):
        ## 处理 Remote
        img_path = self.img_list[index]
        # 读取图像文件
        image = Image.open(os.path.join(self.image_dir_path, img_path + ".tif"))
        # 数据增强
        if self.mode == 'train':
            transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，选择一个概率
                    transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转，选择一个概率
                    transforms.RandomRotation(degrees=180), # 随机旋转 
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
            ])
        image = transform(image)

        ## 处理 Taxi
        taxi_attrs = np.array(self.taxi_list[index])
        taxi_attrs = torch.FloatTensor(taxi_attrs.astype(np.float32))

        ## 处理 sv
        sv_attrs = self.sv_list[index]
        sv_attrs = [float(x) if float(x)!=0 else 0.0000001 for x in sv_attrs.split(",")]
        sv_attrs = torch.FloatTensor(sv_attrs)
        sv_attrs = sv_attrs.view(-1, 512)

        # label = self.label_list[index].astype(np.int64)
        label = self.label_list[index]

        sample = (image, sv_attrs, taxi_attrs, label)
        return sample
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_list)

class MultiDataset1SV(Dataset):
    def __init__(self, img_list, sv_list, taxi_list, label_list, image_dir_path, resize_size=(224, 224), mode='train'):
        self.img_list = img_list
        self.sv_list = sv_list
        self.taxi_list = taxi_list
        self.label_list = label_list
        self.image_dir_path = image_dir_path
        self.resize_size = resize_size
        self.mode = mode
    
    def __getitem__(self, index):
        ## 处理 Remote
        img_path = self.img_list[index]
        # 读取图像文件
        image = Image.open(os.path.join(self.image_dir_path, img_path + ".tif"))
        # 数据增强
        if self.mode == 'train':
            transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，选择一个概率
                    transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转，选择一个概率
                    transforms.RandomRotation(degrees=180), # 随机旋转 
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
            ])
        image = transform(image)

        ## 处理 Taxi
        taxi_attrs = np.array(self.taxi_list[index])
        taxi_attrs = torch.FloatTensor(taxi_attrs.astype(np.float32))

        ## 处理 sv
        sv_path = self.sv_list[index]
        # 读取图像文件
        sv = Image.open(sv_path)
        # 数据增强
        sv = transform(sv)

        # label = self.label_list[index].astype(np.int64)
        label = self.label_list[index]

        sample = (image, sv, taxi_attrs, label)
        return sample
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_list)

class TwoDataset(Dataset):
    def __init__(self, img_list, sv_list, taxi_list, label_list, image_dir_path, modal, resize_size=(224, 224), mode='train'):
        self.modal = modal
        self.img_list = img_list
        self.sv_list = sv_list
        self.taxi_list = taxi_list
        self.label_list = label_list
        self.image_dir_path = image_dir_path
        self.resize_size = resize_size
        self.mode = mode
    
    def __getitem__(self, index):
        if "remote" in self.modal:
            ## 处理 Remote
            img_path = self.img_list[index]
            # 读取图像文件
            image = Image.open(os.path.join(self.image_dir_path, img_path + ".tif"))
            # 数据增强
            if self.mode == 'train':
                transform = transforms.Compose([
                        transforms.Resize(self.resize_size),
                        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，选择一个概率
                        transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转，选择一个概率
                        transforms.RandomRotation(degrees=180), # 随机旋转 
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
                    ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
                ])
            f1 = transform(image)
            
            if "sv" in self.modal:
                ## 处理 sv
                sv_attrs = self.sv_list[index]
                sv_attrs = [float(x) if float(x)!=0 else 0.0000001 for x in sv_attrs.split(",")]
                sv_attrs = torch.FloatTensor(sv_attrs)
                f2 = sv_attrs.view(-1, 512)
            else:
                ## 处理 Taxi
                taxi_attrs = np.array(self.taxi_list[index])
                f2 = torch.FloatTensor(taxi_attrs.astype(np.float32))
        else:
            ## 处理 sv
            sv_attrs = self.sv_list[index]
            sv_attrs = [float(x) if float(x)!=0 else 0.0000001 for x in sv_attrs.split(",")]
            sv_attrs = torch.FloatTensor(sv_attrs)
            f1 = sv_attrs.view(-1, 512)
            
            ## 处理 Taxi
            taxi_attrs = np.array(self.taxi_list[index])
            f2 = torch.FloatTensor(taxi_attrs.astype(np.float32))
        
        # label = self.label_list[index].astype(np.int64)
        label = self.label_list[index]

        sample = (f1, f2, label)
        return sample

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_list)

class SVEnd2EndDataset(Dataset):
    def __init__(self, sv_list, label_list, dir, resize_size=(224, 224), mode='train'):
        self.sv_list = sv_list
        self.label_list = label_list
        self.resize_size = resize_size
        self.dir = dir
        self.mode = mode
    
    def __getitem__(self, index):
        ## 处理 sv
        sv_paths = self.sv_list[index][0]
        # print(sv_paths)
        try:
            sv_paths = sv_paths.split(",")
        except:
            print(sv_paths)
            sys.exit(1)
        sv_list = []
        for sv_path in sv_paths:
            # 读取图像文件
            sv_img = Image.open(os.path.join(self.dir, sv_path))
            # 设置好需要转换的变量，还可以包括一系列的 normalize 等等操作
            if self.mode == 'train':
                transform = transforms.Compose([
                # transforms.Resize(self.resize_size),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=(0.6,1.4), contrast=(0.6,1.4), saturation=(0.6,1.4), hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            else:
                # 测试和验证不做数据增强
                transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            sv_img = transform(sv_img)
            sv_list.append(sv_img)
        sv_attrs = torch.stack(sv_list, 0)
        sv_attrs[sv_attrs == 0] = 0.0000001

        label = self.label_list[index]

        sample = (sv_attrs, label)
        return sample
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.label_list)

class End2EndDataset(Dataset):
    def __init__(self, img_list, sv_list, taxi_list, label_list, image_dir_path, resize_size=(224, 224), mode='train'):
        self.img_list = img_list
        self.sv_list = sv_list
        self.taxi_list = taxi_list
        self.label_list = label_list
        self.image_dir_path = image_dir_path
        self.resize_size = resize_size
        self.mode = mode
    
    def __getitem__(self, index):
        ## 处理 Remote
        img_path = self.img_list[index]
        # 读取图像文件
        image = Image.open(os.path.join(self.image_dir_path, img_path + ".tif"))
        # 数据增强
        if self.mode == 'train':
            transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，选择一个概率
                    transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转，选择一个概率
                    transforms.RandomRotation(degrees=180), # 随机旋转 
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010]) # 按imagenet权重的归一化标准归一化
            ])
        image = transform(image)

        ## 处理 Taxi
        taxi_attrs = np.array(self.taxi_list[index])
        taxi_attrs = torch.FloatTensor(taxi_attrs.astype(np.float32))

        ## 处理 sv
        sv_paths = self.sv_list[index]
        sv_paths = sv_paths.split(",")
        sv_list = []
        for sv_path in sv_paths:
            # 读取图像文件
            sv_img = Image.open(sv_path)
            # 设置好需要转换的变量，还可以包括一系列的 normalize 等等操作
            if self.mode == 'train':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=(0.6,1.4), contrast=(0.6,1.4), saturation=(0.6,1.4), hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                # 测试和验证不做数据增强
                transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            sv_img = transform(sv_img)
            sv_list.append(sv_img)
        sv_attrs = torch.stack(sv_list, 0)
        # print("sv attrs shape: ", sv_attrs.shape)

        label = self.label_list[index].astype(np.int64)

        sample = (image, sv_attrs, taxi_attrs, label)
        return sample
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.img_list)