import os
import pdb
import cv2
import torch
import numpy as np
from torchvision import transforms, utils
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
def default_loader(path):
    return Image.open(path).convert('RGB')
    # pdb.set_trace()
    # if grey==0:
    # img = cv2.imread(path)
    #     img = cv2.resize(img, (384, 384))
    #     img = img / 255.0
    #     img = torch.Tensor(img)
    # else:
    #      img = cv2.imread(path, 0)
    #      img = cv2.resize(img, (384, 384))
    #      img = torch.Tensor(img)

    # return img


class Busi(Dataset):
    def __init__(self, imagefilepath, txtfile, transform, loader=default_loader):
        imgs = []
        labels = []
        data = []
        with open(txtfile) as f:
            for line in f.readlines():
                temp = line.strip().split(",")
                data.append(temp)
        # print(data)
        for i in data:
            imgs.append(i[0])
            labels.append(int(i[1]))
        labels = torch.tensor(labels, dtype=None, device=None)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        # self.transform_test = transform_test
        self.loader = loader
        self.imagefilepath = imagefilepath
        #self.imagefilepath_label = imagefilepath_label

    def __getitem__(self, index):
        imageSize = len(self.imgs)
        # fn, label = self.imgs[index]
        img_name = self.imgs[index]
        label_name = self.labels[index]
        img = self.loader(self.imagefilepath + img_name)
        # label = self.loader(self.imagefilepath_label + label_name)
        # label = self.loader(self.imagefilepath_label + label_name,grey=1)
        img = self.transform(img)
        #label = self.transform_test(label)
        label = self.labels[index]
        # print('22222222222',img.size)
        # pdb.set_trace()
        # print(img.shape)
        return img, label

    def name(self):
        return self.imgs[0]

    def __len__(self):
        return len(self.imgs)

class STL10Dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            self.data_path = os.path.join(data_dir, 'train_X.bin')
            self.labels_path = os.path.join(data_dir, 'train_y.bin')
        elif split == 'test':
            self.data_path = os.path.join(data_dir, 'test_X.bin')
            self.labels_path = os.path.join(data_dir, 'test_y.bin')

        with open(self.data_path, 'rb') as f:
            self.data = np.fromfile(f, dtype=np.uint8)
            self.data = self.data.reshape((-1, 3, 96, 96))

        with open(self.labels_path, 'rb') as f:
            self.labels = np.fromfile(f, dtype=np.uint8)
            self.labels -= 1  
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index].transpose(1, 2, 0))
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        # print(label.shape)
        return image, label

class SVHNDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            self.data_path = os.path.join(data_dir, 'train_digitStruct.mat')
        elif split == 'test':
            self.data_path = os.path.join(data_dir, 'test_digitStruct.mat')

        f = h5py.File(self.data_path, 'r')
        self.data = f['digitStruct']['']
        self.labels = f['digitStruct'][1]
        # 将标签数据中的10替换为0（SVHN数据集中标签10表示数字0）
        self.labels[self.labels == 10] = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[:, :, :, index].transpose(1, 2, 0))
        label = int(self.labels[index, 0])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class ISIC2018dataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_labels = pd.read_csv(labels)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) + ".jpg"
        image = Image.open(img_path).convert('RGB')
        label = list(self.img_labels.iloc[idx])[1:]
        label = torch.tensor(label.index(1.0))
        if self.transform:
            image = self.transform(image)
        return image, label