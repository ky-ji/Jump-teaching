import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet50, vgg19_bn

import numpy as np
from utils import get_model
from losses import GCELoss, Mixup
from tqdm import tqdm
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import gc
from torch.distributions.beta import Beta
from models.resnet import Resnet50SH
import sys



class jumpteaching:
    def __init__(
            self,
            config: dict = None,
            input_channel: int = 3,
            num_classes: int = 10,
        ):
        self.num_classes = num_classes
        self.hashbits = config["hashbits"]
        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.epochs = config['epochs']
        self.tmp_epoch = 0
        self.alpha = torch.tensor(5.0).to(self.device)
        self.dataset = config['dataset']
        self.labelhashcodes = torch.load(config['labelhashcodes_path']+str(config['hashbits'])+'_'+str(config['dataset'])+'_'+str(config['num_classes'])+'_class.pkl').to(self.device)
        self.noise_type = config['noise_type']+'_'+str(config['percent'])


        is_real_noise = any(x in self.dataset.lower() for x in ['webvision', 'clothing', 'food101'])
        self.display_name = self.dataset if is_real_noise else f"{self.dataset}_{self.noise_type}"
        if config.get('writer_name'):  # Default value is True
            self.writer = SummaryWriter(config['writer_name'])
        else:
            self.writer = SummaryWriter(log_dir=None)  # Do not write to file


        self.Step = config['Step']
        self.T = config['T']

        if 'cifar' in self.dataset or 'tiny_imagenet' in self.dataset:
            self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, self.hashbits, device)
        elif 'clothing' in self.dataset or 'food' in self.dataset or 'webvision' in self.dataset:
            self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, self.hashbits, device)


        
        self.adjust_lr = 1
        self.lamd_ce = 1
        self.lamd_h = 0
        self.start_epoch = 30
        self.lr = 0.2
        self.th = config['tau']

        self.weight_decay = 1e-3
        
        if 'cifar-100' in config['dataset']:
            if config['percent'] == 0.8 or config['percent'] == 0.5:
                self.weight_decay=5e-4
                
        if 'webvision' in config['dataset']:
            self.weight_decay=1e-4
        
        if 'food' in config['dataset']:
            self.lr=0.01
            self.weight_decay=5e-4

        self.percent = config['percent']
        self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        config['optimizer'] = 'sgd' 
        self.optim_type = 'sgd'

        self.scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'],eta_min=0.0005)
        config['weight_decay'] = self.weight_decay
      

          
        if 'clothing' in config['dataset']:
            self.lr = 0.01
            self.start_epoch = 5
            self.weight_decay = 1e-3
            self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            config['optimizer'] = 'sgd' 
            self.optim_type = 'sgd'
            self.scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'],eta_min=0.0002)


        config['lr'] = self.lr
        config['start_epoch'] = self.start_epoch


        if 'cifar' in config['dataset']:
            N = 50000
        elif 'clothing1M' in config['dataset']:
            N = 1037498
        elif 'tiny_imagenet' in config['dataset']:
            N = 100000
        elif 'food' in config['dataset']:
            N = 75750
        elif 'animal' in config['dataset']:
            N = 50000
        elif 'webvision' in config['dataset']:
            N = 65944
        self.N = N
        self.labels = torch.ones(N).long().to(self.device)
        if 'cifar' in config['dataset']:
            self.gt_labels = torch.tensor(self.get_gt_labels(config['dataset'], config['root'])).to(self.device)
        self.clean_flags = torch.zeros(N).bool().to(self.device)
        
        self.accs = np.zeros(self.epochs)
        self.GCE_loss = GCELoss(num_classes=num_classes, gpu=config['gpu'])
        self.mixup_loss = Mixup(gpu=config['gpu'], num_classes=num_classes, alpha=config['alpha'])
        self.criterion_u = nn.BCELoss(reduction="none")
        self.criterion = nn.BCELoss()
        self.criterion2 = nn.L1Loss()
        self.CELoss = nn.CrossEntropyLoss()

    def train(self, train_loader, epoch):
        print('Training ...')
        epoch_loss = 0.0
        epoch_celoss = 0.0
        epoch_bceloss = 0.0
        T=self.T
        self.model_scratch.train()
        self.tmp_epoch = epoch

        initial_memory = torch.cuda.memory_allocated()/ (1024 ** 2)
        print("Initial memory usage: {} MB".format(initial_memory))
        pbar = tqdm(train_loader)
        
        ## Warm-up
        if epoch < self.start_epoch:
            for (images, targets, indexes) in pbar:
                w_imgs= Variable(images[0]).to(self.device, non_blocking=True)
                
                targets = Variable(targets).to(self.device)
                targetcodes = self.labelhashcodes[targets]    
                         
                outputs_c,w_logits = self.model_scratch(w_imgs)

                #Initialize training loss
                loss_sup = torch.tensor(0).float().to(self.device)
                              
                outputs_c = outputs_c.to(self.device)
                w_logits = w_logits.to(self.device)
                outputs_c /= T
                w_logits = torch.clamp(w_logits,-1. +1e-4, 1.- 1e-4)
                w_losses = self.criterion_u(0.5*(w_logits+1), 0.5*(targetcodes+1))
                
                # Dual heads
                loss_sup += torch.mean(w_losses)
                epoch_bceloss += torch.mean(w_losses)


                loss_sup += self.CELoss(outputs_c, targets)
                epoch_celoss+=self.CELoss(outputs_c, targets)
                self.optimizer.zero_grad()

                loss_sup.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    w_vars = torch.var(w_losses,dim=1)
                    output_prob_max,output_prob_max_index = torch.max(outputs_c,dim=1)
                    labels= self.labels[indexes]
                    w_mask = torch.logical_or(output_prob_max_index==labels,w_vars<self.th)
                    self.clean_flags[indexes] = w_mask
                    
                pbar.set_description(
                        '%s:Epoch [%d/%d], loss_sup: %.4f'
                        % (self.display_name, epoch + 1, self.epochs, loss_sup.data.item()))
                epoch_loss += loss_sup.item()
                
                        
        else:
            for (images, targets, indexes) in pbar:
                targets = Variable(targets).to(self.device)
                targetcodes = self.labelhashcodes[targets]
                w_imgs = Variable(images[0]).to(self.device, non_blocking=True)
                s_imgs= Variable(images[1]).to(self.device, non_blocking=True)
                lam = Beta(self.alpha, self.alpha).sample() if self.alpha > 0 else 1
                lam = max(lam,1-lam)
                w_imgs = lam*w_imgs +(1-lam)*s_imgs

                #############CE+GCE############

                outputs_c,w_logits = self.model_scratch(w_imgs)
                
                outputs_c /= T

                w_logits = torch.clamp(w_logits,-1. +1e-4, 1.- 1e-4)
                
                w_losses = self.criterion_u(0.5*(w_logits+1), 0.5*(targetcodes+1)) #N*Hashbits

                loss_sup = torch.tensor(0).float().to(self.device)
                
                b_clean_flags = self.clean_flags[indexes]
                
                clean_num = b_clean_flags.sum() 
                batch_size = len(w_imgs)
                
                if clean_num:
                    clean_loss_sup = self.CELoss(outputs_c[b_clean_flags], targets[b_clean_flags])
                    epoch_celoss += clean_loss_sup
                    clean_loss_sup += self.criterion(0.5*(w_logits[b_clean_flags]+1), 0.5*(targetcodes[b_clean_flags]+1)) 
                    epoch_bceloss += self.criterion(0.5*(w_logits[b_clean_flags]+1), 0.5*(targetcodes[b_clean_flags]+1))  
                    loss_sup += clean_loss_sup * self.lamd_ce * (clean_num/batch_size)
                    epoch_loss += loss_sup.item()

                   
                with torch.no_grad():
                    w_vars = torch.var(w_losses,dim=1)
                    output_prob_max,output_prob_max_index = torch.max(outputs_c,dim=1)
                    labels = self.labels[indexes]
                    w_mask = torch.logical_or(output_prob_max_index==labels,w_vars<self.th)
                    self.clean_flags[indexes] = w_mask


                if loss_sup :
                    self.optimizer.zero_grad()
                    loss_sup.backward()
                    self.optimizer.step()

                pbar.set_description(
                        '%s:Epoch [%d/%d], loss_sup: %.4f'
                        % (self.display_name, epoch + 1, self.epochs, loss_sup.data.item()))
                
                
        
        with torch.no_grad():
          
            clean_num = self.clean_flags.sum()
            print("Clean num is %d"%clean_num)


        if self.adjust_lr:
            if self.optim_type == 'sgd':
                self.scheduler.step()
            elif self.optim_type == 'adam':
                self.adjust_learning_rate(self.optimizer, epoch)
                print("lr is %.8f." % (self.alpha_plan[epoch]))
        
        if 'cifar' in self.dataset:
            self.writer.add_scalar('Clean_Number', clean_num, epoch)
            self.writer.add_scalar('Loss/Training', epoch_loss / len(train_loader), epoch)
            self.writer.add_scalar('CELoss/Training', epoch_celoss/ len(train_loader), epoch)
            self.writer.add_scalar('BCELoss/Training', epoch_bceloss / len(train_loader), epoch)
                
    def evaluate(self, test_loader):
        print('Evaluating ...')

        self.model_scratch.eval()  # Change model to 'eval' mode

        correct = 0
        correct2 = 0 
        correct_top5 = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            logits_c,logits = self.model_scratch(images)
            outputs = F.softmax(logits_c, dim=1)
            _, pred = torch.max(outputs.data, 1)
            correct += (pred.cpu() == labels).sum()
            
            semcodes = torch.sign(logits)
            scores = semcodes@self.labelhashcodes.t()
            preds,predlabels = torch.max(scores,dim=1)
            total += labels.size(0)
            if 'webvision' in self.dataset:
                _, index = outputs.topk(5, dim=1)
                index = index.cpu()
                correct_tmp = index.eq(labels.view(-1, 1).expand_as(index))
                correct_top5 += correct_tmp.sum().cpu()
            correct2 += (predlabels.cpu() == labels).sum()

        
        acc = 100 * float(correct) / float(total)
        acc2 = 100 * float(correct2) / float(total)
        self.accs[self.tmp_epoch] = acc
        if 'webvision' in self.dataset:
            acc_top5 =100 * float(correct_top5)/float(total)
            return acc, acc_top5

        if 'clothing1M' in self.dataset and self.tmp_epoch>1:
            
            checkpoint_root = './Experiments/checkpoints/%s/'%self.dataset
            if not os.path.exists(checkpoint_root):
                os.makedirs(checkpoint_root)
            if acc > max(self.accs[:self.tmp_epoch]):
                print('| Saving Best Net%d ...'%self.tmp_epoch)
                save_point = checkpoint_root+'DSCO_prevthresh.pth.tar'
                torch.save(self.model_scratch.state_dict(), save_point)

        return acc,acc2

   

    def save_checkpoints(self):
        checkpoint_root = './Experiments/checkpoints/%s/jumpteaching_'%self.dataset
        filename = checkpoint_root + 'save_epoch%d_%s'%(self.start_epoch,self.noise_type)
        
        if not os.path.exists(checkpoint_root):
            os.makedirs(checkpoint_root)

        state = {'epoch': self.start_epoch,'model': self.model_scratch.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename+'.pth')
        print("The model has been saved !!!!!")


    def load_checkpoints(self):
        filename = './Experiments/checkpoints/%s/jumpteaching_save_epoch%d_%s.pth' % (
            self.dataset, self.start_epoch, self.noise_type
        )
        model_parameters = torch.load(filename)
        self.model_scratch.load_state_dict(model_parameters['model'])
        print("The model has been loaded !!!!!")

    def get_clothing_labels(self, root):
        # import ipdb; ipdb.set_trace()
        with open('%s/label/noisy_label_kv.txt'%root,'r') as f:
            lines = f.read().splitlines()
            train_labels = {}
            for l in lines:
                entry = l.split(' ')
                img_path = entry[0]
                if os.path.exists(root+'/'+img_path):                        
                    # self.train_imgs.append(img_path)                                               
                    train_labels[img_path] = int(entry[1])

        with open('%s/label/noisy_train_key_list.txt'%root,'r') as f:
            lines = f.read().splitlines()
            for i, l in enumerate(lines):
                img_path = l.strip()
                self.labels[i]=torch.tensor(train_labels[img_path]).long()

    def get_food101N_labels(self, root):
        # import ipdb; ipdb.set_trace()
        class2idx = {}
        imgs_root = root + '/images/'
        train_file = root + '/meta/train.txt'
        imgs_dirs = os.listdir(imgs_root)
        imgs_dirs.sort()
        
        for idx, food in enumerate(imgs_dirs):
            class2idx[food] = idx
            
        with open(train_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                train_img = line.strip()
                target = class2idx[train_img.split('/')[0]]
                self.labels[i]=torch.tensor(int(target)).long()

    def get_animal10N_labels(self, root):
        imgs_root = root + '/training/'
        imgs_dirs = os.listdir(imgs_root)
        
        for i, img in enumerate(imgs_dirs):
            self.labels[i] = torch.tensor(int(img[0])).long()

    def get_webvision_labels(self, root):
        with open(root+'info/train_filelist_google.txt') as f:
            lines=f.readlines()    
            i=0
            for line in lines:
                _, target = line.split()
                target = int(target)
                if target<50:
                    self.labels[i]=torch.tensor(target).long()
                    i+=1

    def get_labels(self, train_loader):
        print("Loading labels......")
        pbar = tqdm(train_loader)
        for (_, targets, indexes) in pbar:
            targets = targets.to(self.device)
            self.labels[indexes] = targets
        print("The labels are loaded!")

    def get_gt_labels(self, dataset, root):
        if dataset=='cifar-10':
            train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
            base_folder = 'cifar-10-batches-py'
        elif dataset=='cifar-100':
            train_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]
            base_folder = 'cifar-100-python'
        targets = []
        for file_name, checksum in train_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        return targets

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1

    


    


        

