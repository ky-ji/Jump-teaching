import argparse
from utils import load_config, get_log_name, get_result_name, set_seed, save_results, \
                  get_test_acc, print_config
from datasets import cifar_dataloader, clothing_dataloader, webvision_dataloader, food101N_dataloader, animal10N_dataloader, tiny_imagenet_dataloader
from distutils.util import strtobool
import algorithms
import numpy as np
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    '-c',
                    type=str,
                    default='./configs/jumpteaching_cifar.py',
                    help='The path of config file.')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--root', type=str, default=None)
parser.add_argument('--imgnet_root', type=str, default=None)
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--noise_type', type=str, default=None)
parser.add_argument('--percent', type=float, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--num_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)

parser.add_argument('--prestart', type=int, default=0)
parser.add_argument('--save',type=int,default=1)

parser.add_argument("--save_result", type=lambda x: bool(strtobool(str(x))), default=True)
parser.add_argument("--save_log", type=lambda x: bool(strtobool(str(x))), default=True)

args = parser.parse_args()



def main():
    prestart = bool(args.prestart)
    save = bool(args.save)

    config = load_config(args.config, _print=False)
    
    # Only override config values if explicitly provided via command line
    if args.dataset is not None:
        config['dataset'] = args.dataset
    if args.root is not None:
        config['root'] = args.root
    if args.imgnet_root is not None:
        config['imgnet_root'] = args.imgnet_root
    if args.gpu is not None:
        config['gpu'] = args.gpu
    if args.noise_type is not None:
        config['noise_type'] = args.noise_type
    if args.percent is not None:
        config['percent'] = args.percent
    if args.seed is not None:
        config['seed'] = args.seed
    if args.num_classes is not None:
        config['num_classes'] = args.num_classes
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size



    if args.save_log:
        writer_base_path = './results'
        config['writer_name'] = get_log_name(config, writer_base_path)

    config['save_path'] = './results'
    


    print_config(config)
    set_seed(config['seed'])
    
    
    if config['algorithm'] == 'jumpteaching':
        model = algorithms.jumpteaching(config,
                                input_channel=config['input_channel'],
                                num_classes=config['num_classes'])
        train_mode = 'train_index'
    elif config['algorithm'] == 'PENCIL':
        model = algorithms.PENCIL(config,
                                  input_channel=config['input_channel'],
                                  num_classes=config['num_classes'])
        train_mode = 'train_index'
    elif config['algorithm'] == 'StandardCE':
        model = algorithms.StandardCE(config,
                                  input_channel=config['input_channel'],
                                  num_classes=config['num_classes'])
        train_mode = 'train_index'
    else:
        model = algorithms.__dict__[config['algorithm']](
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_index'
        if config['algorithm'] == 'StandardCETest':
            train_mode = 'train_index'

    if 'cifar' in config['dataset']:
        dataloaders = cifar_dataloader(cifar_type=config['dataset'],
                                       root=config['root'],
                                       batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       noise_type=config['noise_type'],
                                       percent=config['percent'])
        trainloader, testloader = dataloaders.run(
            mode=train_mode), dataloaders.run(mode='test')
        
    elif 'clothing' in config['dataset']:
        clothing_dataloaders = clothing_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'])
        trainloader, evalloader, testloader = clothing_dataloaders.run()
        
    elif 'webvision' in config['dataset']:
        webvision_dataloaders = webvision_dataloader(
            root_dir_web=config['root'],
            root_dir_imgnet=config['imgnet_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            num_class=config['num_classes'])
        trainloader, evalloader, testloader = webvision_dataloaders.run()
        
    elif 'food101N' in config['dataset']:
        food101N_dataloaders = food101N_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'])
        trainloader, testloader = food101N_dataloaders.run(mode=train_mode)
    
    elif 'tiny_imagenet' in config['dataset']:
        tiny_imagenet_dataloaders = tiny_imagenet_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            noise_type=config['noise_type'],
            percent=config['percent'])
        trainloader = tiny_imagenet_dataloaders.run(mode=train_mode)
        testloader = tiny_imagenet_dataloaders.run(mode='test')
    
    elif 'animal' in config['dataset'].lower():
        animal10N_dataloaders = animal10N_dataloader(
            root_dir=config['root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'])
        trainloader, testloader = animal10N_dataloaders.run(mode=train_mode)
        
    
    num_test_images = len(testloader.dataset)
    start_epoch = 0
    epoch = 0


    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0
    val_acc_list, test_acc_list = [], []

    
    # loading training labels
    if config['algorithm']in ['jumpteaching']:
        if 'cifar' in config['dataset']:
            model.get_labels(trainloader)
        elif 'tiny_imagenet' in config['dataset']:
            model.get_labels(trainloader)
        elif 'clothing' in config['dataset']:
            model.get_clothing_labels(config['root'])
            start_epoch = 0
        elif 'webvision' in config['dataset']:
            model.get_webvision_labels(config['root'])
        elif 'food101N' in config['dataset']:
            model.get_food101N_labels(config['root'])
        print('The labels are loaded!!!')

    since = time.time()
    if prestart:
        model.load_checkpoints()
        start_epoch = model.start_epoch

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    print(f"Peak memory usage before training: {peak_memory:.2f} MB")

    for epoch in range(start_epoch, config['epochs']):
        # train
        if(hasattr(model,"start_epoch") and epoch ==model.start_epoch and save ):
            model.save_checkpoints()
        torch.cuda.reset_peak_memory_stats()
        model.train(trainloader, epoch)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        
        if 'webvision' in config['dataset']: # webvision needs to validate on webvision's val set and ImageNet's test set
            val_acc = model.evaluate(evalloader)
            print(
                'Epoch [%d/%d] Val Accuracy on the %s val images: top1: %.4f top5: %.4f %%'
                % (epoch + 1, config['epochs'], num_test_images, val_acc[0],
                   val_acc[1]))

            test_acc = model.evaluate(testloader)
            if best_acc < test_acc[0]:
                best_acc, best_epoch = test_acc[0], epoch

            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: top1: %.4f, top5: %.4f. %%'
                % (epoch + 1, config['epochs'], num_test_images, test_acc[0],
                   test_acc[1]))

            if epoch >= config['epochs'] - 5:
                val_acc_list.append(val_acc)
                test_acc_list.append(test_acc)
            if epoch >= config['epochs'] - 10:
                acc_list.extend([test_acc[0]])
            acc_all_list.extend([test_acc[0]])

        else:
            # evaluate
            if 'clothing' in config['dataset']:
                count_num=5
            else:
                count_num=10
            test_acc = get_test_acc(model.evaluate(testloader))
            if(isinstance(test_acc,tuple)):
                print(
                    'Epoch [%d/%d] Test Accuracy on the %s test images: %.4f,%4f %%' %
                    (epoch + 1, config['epochs'], num_test_images, test_acc[0],test_acc[1]))
                if(hasattr(model,"writer")):
                    model.writer.add_scalar("accuracy",test_acc[1],epoch)
                if epoch > config['epochs'] - count_num:
                    acc_list.extend([test_acc[1]])
                acc_all_list.extend([test_acc[1]])
            else:
                if best_acc < test_acc:
                    best_acc, best_epoch = test_acc, epoch

                print(
                    'Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%' %
                    (epoch + 1, config['epochs'], num_test_images, test_acc))
                if(hasattr(model,"writer")):
                    model.writer.add_scalar("accuracy",test_acc,epoch)
                if epoch > config['epochs'] - count_num:
                    acc_list.extend([test_acc])
                if 'clothing' in config['dataset']:
                    trainloader, evalloader, testloader = clothing_dataloaders.run(
                    )

                acc_all_list.extend([test_acc])
        


    time_elapsed = time.time() - since
    total_min = time_elapsed // 60
    hour = total_min // 60
    min = total_min % 60
    sec = time_elapsed % 60

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        hour, min, sec))
    print_config(config)

    if args.save_result:
        acc_np = np.array(acc_list)

        jsonfile = get_result_name(config, path=config['save_path'])
        np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))

        if 'webvision' in config['dataset'] and len(val_acc_list) > 0:
            val_arr = np.array(val_acc_list)
            test_arr = np.array(test_acc_list)
            webvision_results = {
                'val_top1_mean': float(val_arr[:, 0].mean()),
                'val_top5_mean': float(val_arr[:, 1].mean()),
                'test_top1_mean': float(test_arr[:, 0].mean()),
                'test_top5_mean': float(test_arr[:, 1].mean())
            }
            save_results(config=config,
                         last_ten=acc_np,
                         best_acc=best_acc,
                         best_epoch=best_epoch,
                         jsonfile=jsonfile,
                         **webvision_results)
        else:
            save_results(config=config,
                         last_ten=acc_np,
                         best_acc=best_acc,
                         best_epoch=best_epoch,
                         jsonfile=jsonfile)


if __name__ == '__main__':
    main()
