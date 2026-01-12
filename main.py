import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
from untils.log_mk import mk_log_dir, update_experiment_results
from untils.copy_file import copy_file
from dataloader.dataloader import dataloader
from train import train_model
from val import val_model
from test import test_model

from untils.model_init import init_net, criterion_choose, optimizer_choose, model_infos
# from draw.draw_image import draw_image

from copy import deepcopy
from untils.scheduler import build_scheduler
import time

from dataloader.dataloader import data_infos
import os
from thop import profile
from torchsummary import summary



parser = argparse.ArgumentParser()
parser.add_argument('--layer_num', dest='layer_num', type=int, default=5,
                    help='the number of layers')
parser.add_argument('--gpu_choose', dest='gpu_choose', type=int, default=0,
                    help='the number of gpu')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=None, 
                    help='batch_size')
parser.add_argument('--lr', dest='lr', type=float, default=None, 
                    help='lr')
parser.add_argument('--data_type', dest='data_type', type=int, default=0,
                    help='the type of data')
parser.add_argument('--net_type', dest='net_type', type=int, default=12,
                    help='the type of net')
parser.add_argument('--optimizer_type', dest='optimizer_type', type=str, default='AdamW',  # 'Adam', 'SGD'
                    help='the type of optimizer')
parser.add_argument('--remark', dest='remark', type=str, default=None,
                    help='Remarks for this experiment')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100,
                    help='num of epochs')
parser.add_argument('--examtype', dest='examtype', type=str, default='outer',
                    help='examtype')
parser.add_argument('--log_name', dest='log_name', type=str, default=None,
                    help='log_name')
parser.add_argument('--mix_prob', default=0.5, type=float,
                    help='mixup probability')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.8,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--run_times', dest='run_times', type=int, default=5,
                    help='run_times')

alldatalist = ['cifar10', 'cifar100', 'tinyimagenet', 'fruits100', 'oxford102',
               'sports100', 'fer2013', 'SVHN', 'isic2018', 'STL-10',
               'Intel Image Classification', 'imagenet',
               'fashionmnist', 'pathmnist', 'dermamnist', 'octmnist', 'retinamnist',
                'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist',
                 'emnist', 'mnist']

allnetlist = ['resnet50', 'resnext', 'wideresnet50', 'coatnet', 'cadnet2',
              'astroformer', 'vit', 'cait', 'pit', 'swin',
               't2t', 'hifuse', 'efficientnet_v2_s', 'mobilenetv2', 'dmahpc',
              'efficientnet', '2DMamba', 'btgnetpp', 'MAMH_DFCNN', 'TransXNet',
               'crossformer','cadnet1', '', 'cadnet3', 'cadnet4']# , 'wideresnet101',  'googlenet', 'EfficientNet'
criterion_str = 'LSCE'  # 'CE', 'LSCE'

args = parser.parse_args()

data_type = alldatalist[args.data_type]
log_name = data_type if args.log_name is None else args.log_name
net_type = allnetlist[args.net_type]
times = args.run_times
M = args.layer_num if 'cadnet' in net_type else 0
k = 1
device = args.gpu_choose
optimizer_str = args.optimizer_type
Remark = args.remark
device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
# device = 'cpu'
num_epochs = args.num_epochs
examtype = args.examtype
batch_size = model_infos[net_type]['batch_size'] if args.batch_size == None else args.batch_size
lr = model_infos[net_type]['lr'] if args.lr == None else args.lr
if __name__ == '__main__':
    for time_n in range(times):
        test_times_records = [10, 20, 30, 40, 50, 60, 70, 80]
        test_acc_records = []
        train_time_spend = 0
        for _ in test_times_records:
            test_acc_records.append('')
        num_classes, data_size = data_infos[f'{data_type}']['num_classes'], data_infos[f'{data_type}']['img_size']
        # train_iter, val_iter, test_iter = dataloader(batch_size, data_type)
        net = init_net(net_type, num_classes, data_size, M, k=k).to(device)
        # print(net)
        print('parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))
        print('*' * 10, 'configs', '*' * 10)
        print('No.time = {}/{},\nlog_name = {}\n'.format(time_n + 1, times, log_name),
              'net_type = {}/{},\nlr = {}, data_num = {}/{},\n batchsize ={} , M={}\n'.format(args.net_type, net_type, lr, args.data_type, data_type, batch_size, M),
              'optimizer = {},\nRemark = {}'.format(optimizer_str, Remark),
              )
        from calflops import calculate_flops
        input_shape = (1, 3, data_size, data_size)
        # time1 = time.time()
        flops, macs, params = calculate_flops(model=net,
                                              input_shape=input_shape,
                                              output_as_string=True,
                                              output_precision=4)
        print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
        print('parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

        criterion = criterion_choose(criterion_str)
        optimizer = optimizer_choose(optimizer_str, net, lr)
        scheduler = build_scheduler(lr, num_epochs, optimizer, len(train_iter))
        dir_path, train_log_name, val_log_name, test_log_name, best_log_name, config_log_name, dir_name \
                                                    = mk_log_dir(M, data_type, net_type, examtype)
        copy_file(dir_path)
        checkpoint_best_val_acc = None
        current_best_val_acc = 0
        for epoch in range(num_epochs):
            time1 = time.time()
            net, train_acc, train_loss = train_model(epoch, num_epochs, train_log_name,
                                                        net, train_iter, device,
                                                        num_classes, criterion, optimizer, scheduler, args)
            time2 = time.time()
            train_time_spend += time2 - time1

            current_best_val_acc, val_acc, val_loss, checkpoint_best_val_acc = val_model(epoch, num_epochs, val_log_name,
                                                                                            best_log_name,
                                                                                            net, val_iter, device,
                                                                                            num_classes, criterion, optimizer,
                                                                                            current_best_val_acc,
                                                                                            checkpoint_best_val_acc, dir_path)
            if epoch + 1 in test_times_records:
                checkpoint_short_time = {
                    "model_static_dict": deepcopy(net.state_dict()),
                    "epoch": epoch,
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                    'scheduler': deepcopy(scheduler.state_dict()),
                }
                net.load_state_dict(checkpoint_best_val_acc['model_static_dict'])
                epo_test, _, _, _, _ = test_model(net, test_iter, device, num_classes, criterion)
                test_acc_records[test_times_records.index(epoch + 1)] = '{:.2f}/{:.2f}'.format(current_best_val_acc, epo_test)
                net.load_state_dict(checkpoint_short_time['model_static_dict'])
                optimizer.load_state_dict(checkpoint_short_time['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint_short_time['scheduler'])

            time3 = time.time()
            print('epoch time spend {} s'.format(time3 - time1))

        net.load_state_dict(checkpoint_best_val_acc['model_static_dict'])
        test_acc, precision, recall, f1, test_loss = test_model(net, test_iter, device, num_classes, criterion)

        per_train_time_spent = train_time_spend / (num_epochs+1)

        update_experiment_results(dir_path, log_name, Remark, examtype, data_type, net_type,
                                    k, M, current_best_val_acc, test_acc, test_times_records, test_acc_records, batch_size,
                                    criterion_str, optimizer_str, lr, epoch + 1, precision, recall, f1, per_train_time_spent)
