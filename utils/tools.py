import os
import os.path as osp
import numpy as np
import torch
from .config import Config
import random
import torch.backends.cudnn as cudnn
import json
import pickle


def get_gt_labels(dataset, root):
    if dataset == 'cifar-10':
        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]
        base_folder = 'cifar-10-batches-py'
    elif dataset == 'cifar-100':
        train_list = [
            ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ]
        base_folder = 'cifar-100-python'
    targets = []
    for file_name, _ in train_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])
    return targets


def load_config(filename: str = None, _print: bool = True):
    '''
    load and print config
    '''
    print('loading config from ' + filename + ' ...')
    configfile = Config(filename=filename)
    config = configfile._cfg_dict

    if _print == True:
        print_config(config)

    return config


def print_config(config):
    print('---------- params info: ----------')
    for k, v in config.items():
        print(k, ' : ', v)
    print('---------------------------------')

def plot_results(epochs, test_acc, plotfile):
    # Optional dependency: only required if this helper is used.
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.plot(np.arange(1, epochs), test_acc, label='scratch - acc')
    plt.xticks(np.arange(0, epochs + 1, max(1, epochs // 20)))  # train epochs
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0, 101, 10))  # Acc range: [0, 100]
    plt.ylabel('Acc divergence')
    plt.savefig(plotfile)

def get_log_name0(config, path='./log2'):
    log_name =  config['dataset'] + '_' + config['algorithm'] + '_' + config['noise_type'] + '_' + \
                str(config['percent']) + '_seed' + str(config['seed'])+'.json'

    if not osp.exists(path):
        os.makedirs(path)
    data_root = path + '/' + config['dataset'] + '/' + config['noise_type']
    if not osp.exists(data_root):
        os.makedirs(data_root)
    log_name = osp.join(data_root, log_name)

    return log_name

def get_result_name(config, path='./results'):
    dataset_name = config['dataset']

    # Real-world noisy datasets don't have synthetic noise
    is_real_noise = any(x in dataset_name.lower() for x in ['webvision', 'clothing', 'food101'])

    if is_real_noise:
        result_dir = osp.join(path, dataset_name)
        log_name = config['dataset'] + '_seed:' + str(config['seed']) + '.json'
    else:
        noise_type = config['noise_type']
        noise_ratio = str(config['percent'])
        result_dir = osp.join(path, dataset_name, noise_type, noise_ratio)
        log_name = config['dataset'] + '_' + config['noise_type'] + '_' + str(config['percent']) + '_seed:' + str(config['seed']) + '.json'

    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    log_name = osp.join(result_dir, log_name)

    return log_name

def get_log_name(config, path='./results'):
    dataset_name = config['dataset']

    # Real-world noisy datasets don't have synthetic noise
    is_real_noise = any(x in dataset_name.lower() for x in ['webvision', 'clothing', 'food101'])

    if is_real_noise:
        result_dir = osp.join(path, dataset_name)
        log_name = config['dataset'] + '_seed:' + str(config['seed'])
    else:
        noise_type = config['noise_type']
        noise_ratio = str(config['percent'])
        result_dir = osp.join(path, dataset_name, noise_type, noise_ratio)
        log_name = config['dataset'] + '_' + config['noise_type'] + '_' + str(config['percent']) + '_seed:' + str(config['seed'])

    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    log_name = osp.join(result_dir, log_name)

    return log_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner   
    cudnn.deterministic = True


def save_results(config, last_ten, best_acc, best_epoch, jsonfile, num_list=None, acc_list=None, test_acc=None,
                 val_top1_mean=None, val_top5_mean=None, test_top1_mean=None, test_top5_mean=None):
    result_dict = config.copy()
    result_dict['last10_acc_mean'] = float(last_ten.mean())
    result_dict['last10_acc_std'] = float(last_ten.std())
    result_dict['best_acc'] = float(best_acc)
    result_dict['best_epoch'] = int(best_epoch)

    if num_list is not None:
        result_dict['num_list'] = [int(x) if isinstance(x, (np.integer, np.int64)) else x for x in num_list]
    if acc_list is not None:
        result_dict['acc_list'] = [float(x) if isinstance(x, (np.floating, np.float64)) else x for x in acc_list]
    if test_acc is not None:
        result_dict['test_acc'] = test_acc.tolist() if isinstance(test_acc, np.ndarray) else test_acc

    if val_top1_mean is not None:
        result_dict['webvision_val_top1_last5'] = val_top1_mean
    if val_top5_mean is not None:
        result_dict['webvision_val_top5_last5'] = val_top5_mean
    if test_top1_mean is not None:
        result_dict['imagenet_test_top1_last5'] = test_top1_mean
    if test_top5_mean is not None:
        result_dict['imagenet_test_top5_last5'] = test_top5_mean

    with open(jsonfile, 'w') as out:
        json.dump(result_dict, out, sort_keys=False, indent=4)





def get_test_acc(acc):
    #return (acc[0] + acc[1]) / 2. if isinstance(acc, tuple) else acc
    return acc