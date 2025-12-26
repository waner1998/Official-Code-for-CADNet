import torch.nn.functional as F
import torch
from untils.log_mk import write_log
from copy import deepcopy
def val_model(epoch, num_epochs, log_name, best_log_name,
              net, val_iter, device, num_classes,
              criterion, optimizer,
              current_best_val_acc, checkpoint_best_val_acc, dir_path):
    net.eval()
    total, correct, loss_val, = 0, 0, 0
    # print(log_name)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_iter):
            images = images.to(device)
            labels = labels.to(device).to(torch.int64)
            if labels.dim() >= 2:
                labels = labels.squeeze(1)
            outputs = net(images)
            _, predicted = torch.max(outputs.detach(), 1)
            # labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=num_classes).float()
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_val += loss

        print("[epoch/epoches:{}/{}], --------------val_acc,{}%,loss:{}"
              .format(epoch + 1, num_epochs, 100 * correct / total, loss_val / (i + 1)))

        write_log(log_name, 'epoch,{},val,{}%,loss,{}\n'
                  .format(epoch + 1, 100 * correct / total, loss_val / (i+1)))
        if current_best_val_acc <= 100 * correct / total:
            current_best_val_acc = 100 * correct / total
            checkpoint_best_val_acc = {
                "model_static_dict": deepcopy(net.state_dict()),
                "epoch": epoch,
                "optimizer_state_dict": deepcopy(optimizer.state_dict())
            }
            if 'cadnet' in log_name:
                torch.save(checkpoint_best_val_acc, dir_path + '/parameter.pkl')
            write_log(best_log_name,
                      'while epoch {} current best val {}%\n'
                      .format(epoch + 1, current_best_val_acc))

    return current_best_val_acc, 100 * correct / total, (loss_val.detach() / (i + 1)).item(), checkpoint_best_val_acc
