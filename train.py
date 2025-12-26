import torch.nn.functional as F
import torch
from untils.log_mk import write_log
import numpy as np
from dataloader.mix_up import mixup_data, mixup_criterion
def train_model(epoch, num_epochs, log_name,
                net, train_iter, device, num_classes,
                criterion, optimizer, scheduler, args):
    total, correct, loss_train, = 0, 0, 0
    net.train()
    for i, (images, labels) in enumerate(train_iter):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).to(torch.int64)
        if labels.dim() >= 2:
            labels = labels.squeeze(1)

        r = np.random.rand(1)
        if r < args.mix_prob:
            r = np.random.rand(1)
            if r < args.mix_prob:
                images, y_a, y_b, lam = mixup_data(images, labels, args)
                predict = net(images)
                loss = mixup_criterion(criterion, predict, y_a, y_b, lam)
            else:
                predict = net(images)
                loss = criterion(predict, labels)
        else:
            predict = net(images)
            loss = criterion(predict, labels)
        # print(predict.shape)
        _, predicted = torch.max(predict, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss_train += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print("[epoch/epoches:{}/{}], --------------train_acc,{}%,loss:{}"
          .format(epoch + 1, num_epochs, 100 * correct / total, loss_train / (i + 1)))
    write_log(log_name, "epoch,{},acc,{}%,loss,{}\n"
              .format(epoch + 1, 100 * correct / total, loss_train / (i + 1)))
    return net, 100 * correct / total, (loss_train.detach() / (i + 1)).item()

