from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F

def test_model(net, test_iter, device, num_classes, criterion):
    net.eval()
    loss_test = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_iter):
            images = images.to(device)
            labels = labels.to(device).to(torch.int64)

            if labels.dim() >= 2:
                labels = labels.squeeze(1)

            outputs = net(images)
            _, predicted = torch.max(outputs.detach(), 1)

            labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=num_classes).float()

            loss = criterion(outputs, labels)
            loss_test += loss

            y_true_list.extend(labels_onehot.argmax(dim=1).cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print("test_acc,{}%,loss:{}"
          .format(accuracy, loss_test / (i + 1)))

    # print(f'Accuracy: {accuracy:.4f}')
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'F1 Score: {f1:.4f}')


    return accuracy, precision, recall, f1, (loss_test / (i + 1)).item()
