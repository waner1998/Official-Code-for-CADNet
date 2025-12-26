import torch
import torchvision.models as models
from torch import nn, fx, optim
from model.vit import vit
from model.pit import pit
from model.cait import cait
from model.swin import swin
from model.t2t import t2t
# from model.torchvision.models import resnet50, resnext50_32x4d, wide_resnet50_2

from model.coatnet import coatnet_2
from model.crossformer import make_crossformer
# from local_net.EfficientNet.model import EfficientNet
from model.cad import cadnet_0,cadnet_1,cadnet_2,cadnet_3,cadnet_4
from model.astroformermain.astroformer import astroformer
from .LSCE import LabelSmoothingCrossEntropy
from model.HiFuse_main.main_model import HiFuse_Base
from model.efficentv2 import effnetv2_m, effnetv2_s
from model.mobilenetv2 import mobilenetv2
from model.efficientnet.efficientnet import *
# from model.TwoDMamba_main.TwoVMamba.classification.models.vmamba import VSSM
from model.TransXNet_main.models.transxnet import TransXNet
# from model.BTG_Netpp_main.models.BTG_Net_pp import BTG_Net_pp
# from model.MAMH_DFCNN import Mymodel as MAMH_DFCNN
model_infos = {
'resnet50':{'batch_size':256, 'lr':1e-3},
'resnext':{'batch_size':256, 'lr':1e-3},
'wideresnet50':{'batch_size':256, 'lr':1e-3},
'coatnet':{'batch_size':256, 'lr':1e-3},
'cadnet2':{'batch_size':256, 'lr':1e-3},
'vit':{'batch_size':128, 'lr':1e-3},
'cait':{'batch_size':128, 'lr':1e-3},
'pit':{'batch_size':128, 'lr':1e-3},
'swin':{'batch_size':128, 'lr':1e-3},
't2t':{'batch_size':128, 'lr':1e-3},

'astroformer':{'batch_size':256, 'lr':1e-3},
'crossformer':{'batch_size':128, 'lr':1e-3},
'hifuse':{'batch_size':256, 'lr':1e-3},
'efficientnet_v2_m':{'batch_size':512, 'lr':1e-3},
'efficientnet_v2_s':{'batch_size':128, 'lr':1e-3},
'mobilenetv2':{'batch_size':512, 'lr':1e-3},
'2DMamba':{'batch_size':128, 'lr':5e-4},
'btgnetpp':{'batch_size':128, 'lr':5e-4},
'MAMH_DFCNN':{'batch_size':128, 'lr':1e-3},
'TransXNet':{'batch_size':128, 'lr':1e-3},
}

def _resnet50(num_classes):
    net = models.resnet50(num_classes=num_classes, pretrained=False)
    return net

def _resnext(num_classes):
    net = models.resnext50_32x4d(num_classes=num_classes, pretrained=False)
    return net

def _wideresnet50(num_classes):
    net = models.wide_resnet50_2(num_classes=num_classes, pretrained=False)
    return net

def _vit(num_classes, data_size):
    net = vit(data_size, num_classes)
    return net

def _cait(num_classes, data_size):
    net = cait(data_size, num_classes)
    return net

def _pit(num_classes, data_size):
    net = pit(data_size, num_classes)
    return net

def _swin(num_classes, data_size):
    net = swin(data_size, num_classes)
    return net

def _t2t(num_classes, data_size):
    net = t2t(data_size, num_classes)
    return net

def _coatnet(num_classes, data_szie):
    net = coatnet_2(data_szie, num_classes)
    return net

def _crossformer(num_classes, data_size):
    net = make_crossformer(data_size, num_classes)
    return net

def _astroformer(num_classes, data_size):
    net = astroformer(data_size, num_classes)
    return net

def _cadnet2(M, k, num_classes, data_size):
    net = cadnet_2(M, k, num_classes, data_size)
    return net

def _hifuse(num_classes, data_size):
    net = HiFuse_Base(num_classes, data_size)
    return net

def _efficientnet_v2_m(num_classes):
    net = effnetv2_m(num_classes=num_classes)
    return net

def _efficientnet_v2_s(num_classes):
    net = effnetv2_s(num_classes=num_classes)
    return net

def _mobilenetv2(num_classes):
    net = mobilenetv2(num_classes=num_classes)
    return net
def _TransXNet(num_classes, data_size):
    net = TransXNet(arch='b', num_classes=num_classes, data_size=data_size)
    return net
# def _btgnetpp(num_classes):
#     net = BTG_Net_pp(num_classes=num_classes)
#     return net
#
# def _MAMH_DFCNN(num_classes):
#     net = MAMH_DFCNN(num_classes=num_classes)
#     return net

def _2DMamba(num_classes, data_size):
    net = VSSM(
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        depths=[2, 2, 9, 2],
        dims=96,
        # ===================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank=("auto"),
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        # ===================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # ===================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="ln",
        downsample_version="v2",
        patchembed_version="v2",
        gmlp=False,
        use_checkpoint=False,
        # ===================
        posembed=False,
        imgsize=data_size,
        # v2d mamba params
        use_v2d=False,
    )
    return net
'''
def _EfficientNet(num_classes):
    net = EfficientNet.from_name('efficientnet-b0')
    net._fc = nn.Linear(net._fc.in_features, num_classes)

    return net

def _vgg16(num_classes):
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(net.classifier[6].in_features, num_classes)

    return net

def _wideresnet101(num_classes):
    net = models.wide_resnet101_2(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    return net

def _googlenet(num_classes):
    net = models.googlenet(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    return net

def _densenet(num_classes):
    net = models.densenet121(pretrained=True)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)

    return net
'''
def init_net(net_type, num_classes, data_size, M, k):
    if net_type == 'resnet50':
        return _resnet50(num_classes)
    elif net_type == 'resnext':
        return _resnext(num_classes)
    elif net_type == 'wideresnet50':
        return _wideresnet50(num_classes)
    elif net_type == 'coatnet':
        return _coatnet(num_classes, data_size)
    elif net_type == 'vit':
        return _vit(num_classes, data_size)
    elif net_type == 'cait':
        return _cait(num_classes, data_size)
    elif net_type == 'pit':
        return _pit(num_classes, data_size)
    elif net_type == 'swin':
        return _swin(num_classes, data_size)
    elif net_type == 't2t':
        return _t2t(num_classes, data_size)
    elif net_type == 'crossformer':
        return _crossformer(num_classes, data_size)
    elif net_type == 'astroformer':
        return _astroformer(num_classes, data_size)
    elif net_type == 'cadnet2':
        return _cadnet2(M, k, num_classes, data_size)
    elif net_type == 'hifuse':
        return _hifuse(num_classes, data_size)
    elif net_type == 'efficientnet_v2_m':
        # return EfficientNet(num_class=num_classes)
        # return models.efficientnet_v2_m(num_classes=num_classes,pre_trained=False)
        return _efficientnet_v2_m(num_classes)
    elif net_type == 'efficientnet_v2_s':
        # return EfficientNet(num_class=num_classes)
        # return models.efficientnet_v2_m(num_classes=num_classes,pre_trained=False)
        return _efficientnet_v2_s(num_classes)
    elif net_type == 'mobilenetv2':
        return _mobilenetv2(num_classes)
    elif net_type == 'efficientnet':
        # return timm.create_model('efficientnetv2_rw_m', num_classes=num_classes, pretrained=False)
        # return EfficientNet(num_class=num_classes)
        # return models.efficientnet_v2_m(num_classes=num_classes, pretrained=False)
        return efficientnet_b5(num_classes=num_classes)
    elif net_type == '2DMamba':
        return _2DMamba(num_classes, data_size)
    elif net_type == 'TransXNet':
        return _TransXNet(num_classes, data_size)
    # elif net_type == 'btgnetpp':
    #     return _btgnetpp(num_classes, data_size)
    # elif net_type == 'MAMH_DFCNN':
    #     return _MAMH_DFCNN(num_classes)
    else:
        raise ValueError("please check the input of net_type")
'''
    elif net_type == 'googlenet':
        return _googlenet(num_classes)
    elif net_type == 'densenet':
        return _densenet(num_classes)
    elif net_type == 'EfficientNet':
        return _EfficientNet(num_classes)
    elif net_type == 'vgg16':
        return _vgg16(num_classes)
    elif net_type == 'wideresnet101':
        return _wideresnet101(num_classes)
'''


def criterion_choose(criterion_type):
    if criterion_type == 'CE':
        return nn.CrossEntropyLoss()
    elif criterion_type == 'LSCE':
        return LabelSmoothingCrossEntropy()
    else:
        raise ValueError("please check the input of criterion_type")

def optimizer_choose(optimizer_type, net, lr):
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-2)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("please check the input of optimizer_type")

    return optimizer



