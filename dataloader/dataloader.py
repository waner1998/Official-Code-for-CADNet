import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import os
from PIL import Image

# import medmnist
# from medmnist import INFO, Evaluator
# from .cutout import Cutout
# from .preprocessing import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, RandomErasing
from .sampler import RASampler

from dataloader.RandAugment import RandAugment

data_infos = {
    'norm': {'mean': [0.4802, 0.4481, 0.3975], 'std': [0.2770, 0.2691, 0.2821]},
    "cifar10": {"img_size": 32, "num_classes": 10, 'norm': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]]},
    "cifar100": {"img_size": 32, "num_classes": 100, 'norm': [[0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]]},
    "tinyimagenet": {"img_size": 64, "num_classes": 200, 'norm': [[0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]]},
    "imagenet": {"img_size": 224, "num_classes": 1000, 'norm': [[0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]]},
    "isic2018": {"img_size": 224, "num_classes": 7},
    "STL-10": {"img_size": 96, "num_classes": 10},
    "SVHN": {"img_size": 32, "num_classes": 10, 'norm': [[0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]]},
    "oxford102": {"img_size": 96, "num_classes": 102},
    "Intel Image Classification": {"img_size": 96, "num_classes": 6},
    "fer2013": {"img_size": 32, "num_classes": 7},
    "mnist": {"img_size": 28, "num_classes": 10, 'norm': [[.5], [.5]]},
    "fashionmnist": {"img_size": 28, "num_classes": 10, 'norm': [[.5], [.5]]},
    "fruits100": {"img_size": 64, "num_classes": 100},
    "pathmnist": {"img_size": 28, "num_classes": 9},
    "dermamnist": {"img_size": 28, "num_classes": 7},
    "octmnist": {"img_size": 28, "num_classes": 4},
    "retinamnist": {"img_size": 28, "num_classes": 5},
    "bloodmnist": {"img_size": 28, "num_classes": 8},
    "tissuemnist": {"img_size": 28, "num_classes": 8},
    "organamnist": {"img_size": 28, "num_classes": 11},
    "organcmnist": {"img_size": 28, "num_classes": 11},
    "organsmnist": {"img_size": 28, "num_classes": 11},
    "emnist": {"img_size": 28, "num_classes": 10, 'norm': [[.5], [.5]]},
}

class One2threechannel(object):
    def __call__(self, img):
        # print(img.size(0))
        num_channels = img.size(0)
        # print(time.time())
        if num_channels == 1:
            img = torch.cat((img, img, img), dim=0)
        # print(time.time())
        return img

def dataloader(batch_size, data_type, num_workers=0
               ):
    data_info = data_infos[f'{data_type}']
    if 'norm' in data_info:
        norm_mean = data_info['norm'][0]
        norm_std = data_info['norm'][1]
    else:
        norm_mean = data_infos['norm']['mean']
        norm_std = data_infos['norm']['std']
    '''
    if data_type == 'cifar10' or data_type == 'cifar100':
        policy = CIFAR10Policy()
    elif data_type == 'SVHN':
        policy = SVHNPolicy()
    else:
        policy = ImageNetPolicy()
    '''
    transform_train = transforms.Compose([
        RandAugment(2, 15),
        transforms.Resize((data_info['img_size'],data_info['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4),
        # policy,
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
        # RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=norm_mean),
        One2threechannel(),

    ])

    transform_val = transforms.Compose([
        transforms.Resize((data_info['img_size'],data_info['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
        One2threechannel(),
    ])

    if data_type == 'cifar10':  # 32
        train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [3000, 7000])

        train_loader = DataLoader(train_dataset, pin_memory=True,
                                                   batch_sampler=RASampler(len(train_dataset), 
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)

        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, pin_memory=True, batch_size=batch_size,shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, pin_memory=True, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    elif data_type == 'cifar100':  # 32
        train_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(valset, [3000, 7000])

        train_loader = DataLoader(train_dataset, pin_memory=True,
                                                  batch_sampler=RASampler(len(train_dataset), batch_size, 1, 3, shuffle=True, drop_last=True),
                                                     num_workers=num_workers)

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_type == 'tinyimagenet':  # 64
        from .dataset import Busi
        filepath = os.getcwd() + "/data/tinyimagenet/"
        trainfilepath = filepath + 'train/'
        valfilepath = filepath + 'val/'
        train_dataset = Busi(trainfilepath, filepath + "train.txt", transform_train)
        val_dataset = Busi(valfilepath, filepath + "val.txt", transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [3000, 7000])
        train_loader = DataLoader(train_dataset, pin_memory=True,
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_type == 'isic2018':
        from .dataset import ISIC2018dataset
        train_data_path = "./data/ISIC2018/ISIC2018_Task3_Training_Input"
        train_label_path = "./data/ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv"
        val_data_path = "./data/ISIC2018/ISIC2018_Task3_Validation_Input"
        val_label_path = "./data/ISIC2018/ISIC2018_Task3_Validation_GroundTruth.csv"
        test_data_path = "./data/ISIC2018/ISIC2018_Task3_Test_Input"
        test_label_path = "./data/ISIC2018/ISIC2018_Task3_Test_GroundTruth.csv"

        # Creating the datasets
        train_dataset = ISIC2018dataset(train_data_path, train_label_path, transform_train)
        # Resize the validation images
        val_dataset = ISIC2018dataset(val_data_path, val_label_path, transform_val)

        test_dataset = ISIC2018dataset(test_data_path, test_label_path, transform_val)

        train_loader = DataLoader(train_dataset, pin_memory=True,
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_type == 'STL-10':  # https://cs.stanford.edu/~acoates/stl10/ # Binary files
        from .dataset import  STL10Dataset
        data_dir = os.getcwd() + "/data/STL-10/"
        train_dataset = STL10Dataset(data_dir, split='train', transform=transform_train)
        test_dataset = STL10Dataset(data_dir, split='test', transform=transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [2000, 6000])
        train_loader = DataLoader(train_dataset, pin_memory=True,
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_type == 'SVHN':
        data_dir = os.getcwd() + "/data/SVHN/"
        train_dataset = torchvision.datasets.SVHN(data_dir, split='train', transform=transform_train, download=True)
        test_dataset = torchvision.datasets.SVHN(data_dir, split='test', transform=transform_val, download=True)
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [9000, 17032])
        train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=True, 
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = DataLoader(val_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif data_type == 'oxford102': # https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset?select=dataset
        from .dataset import Busi
        filepath = os.getcwd() + "/data/Oxford102/dataset/"
        trainfilepath = filepath + 'train/'
        valfilepath = filepath + 'val/'
        train_dataset = Busi(trainfilepath, filepath + "train.txt", transform_train)
        val_dataset = Busi(valfilepath, filepath + "val.txt", transform_val)
        # val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [3000, 7000])

        # train_dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(train_dataset.classes)}
        # val_dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(val_dataset.classes)}
        val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [300, 518])

        train_loader = DataLoader(train_dataset, pin_memory=True,
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = DataLoader(dataset=val_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_type == 'Intel Image Classification':  # https://www.kaggle.com/datasets/puneet6060/intel-image-classification
        train_folder = os.getcwd() + '/data/Intel Image Classification/seg_train'
        val_folder = os.getcwd() + '/data/Intel Image Classification/seg_test'

        train_dataset = torchvision.datasets.ImageFolder(train_folder, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(val_folder, transform=transform_val)

        train_dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(train_dataset.classes)}
        val_dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(val_dataset.classes)}
        val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [1000, 2000])

        train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=True, 
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_type == 'fruits100':  # https://www.kaggle.com/datasets/marquis03/fruits-100/
        train_dataset = torchvision.datasets.ImageFolder(root=os.getcwd()+'/data/fruits100/train', transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(root=os.getcwd()+'/data/fruits100/val', transform=transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [1600, 3400])
        # test_dataset = torchvision.datasets.ImageFolder(root=os.getcwd()+'/data/fruits100/test', transform=transform_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=True, 
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_type == 'mnist':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(os.getcwd() + "/data/", train=True, download=True, transform=transform_train), batch_size=batch_size, shuffle=True)
        sets = torchvision.datasets.MNIST(os.getcwd() + "/data/", train=False, download=True, transform=transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    elif data_type == 'fashionmnist':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(os.getcwd() + "/data/", train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True)
        sets = torchvision.datasets.FashionMNIST(os.getcwd() + "/data/", train=False, download=True, transform=transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    elif data_type == 'emnist':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST(os.getcwd() + "/data/", split = 'mnist', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True)
        sets = torchvision.datasets.EMNIST(os.getcwd() + "/data/",split = 'mnist', train=False, download=True, transform=transform_val)
        val_dataset, test_dataset = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    elif data_type == 'fer2013': # https://www.kaggle.com/datasets/msambare/fer2013?select=train
        from .dataset import Busi
        filepath = os.getcwd() + "/data/fer2013/"
        trainfilepath = filepath + 'train/'
        valfilepath = filepath + 'test/'
        train_dataset = Busi(trainfilepath, filepath + "train.txt", transform_train)
        test_dataset = Busi(valfilepath, filepath + "test.txt", transform_val)
        # print(test_dataset.shape)
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [3000, 4178])
        train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=True,
                                                   batch_sampler=RASampler(len(train_dataset),
                                                    batch_size, 1, 3, shuffle=True, drop_last=True), num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        info = INFO[data_type]
        task = info['task']
        DataClass = getattr(medmnist, info['python_class'])
        data_transform_train = torchvision.transforms.Compose([
        transforms.ToTensor(),
        One2threechannel(),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize(mean=[.5], std=[.5])
        ])
        data_transform_val = torchvision.transforms.Compose([
        transforms.ToTensor(),
        One2threechannel(),
        torchvision.transforms.Normalize(mean=[.5], std=[.5])
        ])
        train_loader = torch.utils.data.DataLoader(
        DataClass(split='train', transform=data_transform_train, download=True),
        batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
        DataClass(split='val', transform=data_transform_val, download=True),
        batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
        DataClass(split='test', transform=data_transform_val, download=True),
        batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
