from datetime import datetime
import os
import csv
import shutil
def mk_log_dir(M, data_type, net_type, examtype):
    log_name = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")).replace(" ", "-").replace(":", "-") + '_{}_{}_{}'.format(M, data_type, net_type)
    dir_name = log_name
    dir_path = os.getcwd() + '/result/{}/'.format(examtype) + net_type + '/' + data_type + '/' + dir_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_log_name = dir_path + '/train' + log_name + ".txt"
    val_log_name = dir_path + '/val' + log_name + ".txt"
    test_log_name = dir_path + '/test' + log_name + ".txt"
    best_log_name = dir_path + '/best' + log_name + ".txt"
    config_log_name = dir_path + '/config' + log_name + ".txt"
    return dir_path, train_log_name, val_log_name, test_log_name, best_log_name, config_log_name, dir_path

def write_log(file_name,text):
    logfile = open(file_name, "a")
    logfile.write(text)

def update_experiment_results(log_dir, all_log_name, Remark, examtype, data, net, k, M, best_val_acc, test_acc, test_times_records, test_acc_records, batch_size,
                              criterion, optimizer, lr, run_times, precision, recall, f1, per_train_time_spent):
    filename = os.getcwd() + '/{}.csv'.format(all_log_name)

    parameters = ['log_dir', 'Remark', 'examtype', 'data', 'net', 'k', 'M', 'best_val_acc', 'test_acc'] + \
                 ['test' + str(x) for x in test_times_records] + \
                 ['batch_size', 'criterion', 'optimizer', 'lr', 'run_times', 'precision', 'recall', 'f1', 'per_train_time_spent']

    file_exists = os.path.exists(filename)
    if not file_exists:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(parameters)


    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([log_dir, Remark, examtype, data, net,
                          k, M, best_val_acc, test_acc] + [x for x in test_acc_records] +
                         [batch_size, criterion, optimizer, lr, run_times, precision, recall, f1, per_train_time_spent])
    return
