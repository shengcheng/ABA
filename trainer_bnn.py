#!/usr/bin/env python
import gc 
import os
import sys
import json 
import time 
sys.path.append(os.path.abspath(''))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from datetime import datetime
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm, trange
import random 
from lib.networks import RandConvModule, BConvModule, Multi_BNN
from lib.utils.average_meter import AverageMeter
from lib.utils.metrics import accuracy
from lib.utils import TVLoss
from lib.datasets import get_dataset
from lib.datasets.transforms import GreyToColor


import matplotlib.pyplot as plt 
# pytorch pretrained model is saved at:
os.environ['TORCH_HOME'] = os.path.realpath('lib/networks/')  
torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)



def t2s(x):
    x_list = x.detach().cpu().reshape(-1).tolist() # make it flat and tolist
    x_str = '-'.join("{:.3f}".format(xx) for xx in x_list)
    return x_str

def asarray_and_reshape(imgs, labels, C=3, H=32, W=32):
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    imgs = np.reshape(imgs, (-1, C, H, W))
    labels = np.reshape(labels, (-1,)).tolist()
    return imgs, labels

def get_exp_name(args):
    exp_name = "".join([
        args.name, '-', args.net, '-', args.source, 
        '-SGD' if args.SGD else '-Adam',
        '-lr{}'.format(args.lr),
        '-bs{}'.format(args.batch_size),
        '-iter{}'.format(args.n_iter)
        ])
    if args.bal:
        bal_name = "".join([
            '-pre_epoch{}'.format(args.pre_epoch), 
            '-lr_adv{}'.format(args.lr_adv),
            '-nadv{}'.format(args.adv_steps),
            '-clw{}'.format(args.clw),
            '-mix' if args.mixing else '',
            '-clamp' if args.clamp_output else '',
            '-wr{}'.format(args.wr) if args.wr!=1.0 else '',
            '-toss{}'.format(args.toss) if args.toss!=0.0 else '',
            '-beta{}'.format(args.elbo_beta),
            '-glayers{}'.format(args.trans_depth),
            ])
        exp_name = exp_name + bal_name
    return exp_name


def get_BCNN_module(args, data_mean, data_std):
    return Multi_BNN(
        out_channels=args.channel_size,
        kernel_size=args.kernel_size,
        mixing=args.mixing,
        clamp=args.clamp_output,
        num_blocks=args.trans_depth,
        data_mean=data_mean,
        data_std=data_std
    )

def get_random_module(args, data_mean, data_std):
    return RandConvModule(
        net=None,
        in_channels=3,
        out_channels=args.channel_size,
        kernel_size=args.kernel_size,
        mixing=True,
        identity_prob=0.0,
        rand_bias=False,
        distribution='kaiming_normal',
        data_mean=data_mean,
        data_std=data_std,
        clamp_output=False
        )


class BAL:
    def __init__(self, args):
        hostname = os.uname()[1]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(gg) for gg in args.gpu_ids]) \
            if type(args.gpu_ids) is list else str(args.gpu_ids)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            cudnn.benchmark = True

        if args.n_iter:
            args.n_epoch = args.n_iter // args.val_iter

        self.args = args
        self.exp_name = get_exp_name(self.args)

        print("Benchmark: {}. Source: {}. \n Exp_Name: {} \n ----".format(
            self.args.data_name, self.args.source, self.exp_name)
        )
        self.set_path()
        with open(os.path.join(
                self.ckpoint_folder, 'seed{}_args.txt'.format(self.args.rand_seed)
                ), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.tv_loss = TVLoss()

        self.writer = SummaryWriter(
            os.path.join(
                self.ckpoint_folder, 'seed{}'.format(self.args.rand_seed)
                )
            )
        self.metric_name = 'acc'
        self.metric = accuracy
        # these will be used only if args.multi_aug is True
        self.jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
        self.gray = transforms.RandomGrayscale()


    def multi_augment(self, input):
        if self.args.multi_aug:
            input = self.dewhiten(input)
            input = self.jitter(input)
            input = self.gray(input)
            return self.whiten(input).contiguous()
        else:
            return input

    
    def whiten(self, input):
        return (input - self.data_mean) / self.data_std

    
    def dewhiten(self, input):
        return input * self.data_std + self.data_mean


    def init_weights(self, m):
        if type(m) == nn.Conv2d: #or type(m)==nn.Linear:
            if self.args.distribution in ['kaiming', 'kaiming_normal']:
                nn.init.kaiming_normal_(m.weight.data)
            elif self.args.distribution == 'dirac':
                nn.init.dirac_(m.weight.data)
            elif self.args.distribution == 'normal':
                nn.init.normal_(m.weight.data)
            elif self.args.distribution == 'xavier':
                nn.init.xavier_normal_(m.weight.data)
            elif self.args.distribution == 'orthogonal':
                nn.init.orthogonal_(m.weight.data)



    def set_path(self):
        self.ckpoint_folder = os.path.join(
            'checkpoints', self.args.data_name, self.exp_name)
        if not os.path.isdir(self.ckpoint_folder):
            os.makedirs(self.ckpoint_folder)
        self.best_ckpoint = os.path.join(
            self.ckpoint_folder, 
            'best{}.ckpt.pth'.format('_seed{}'.format(self.args.rand_seed))
            )
        self.best_target_ckpoint = os.path.join(
            self.ckpoint_folder, 
            'best_target{}.ckpt.pth'.format(
                '_seed{}'.format(self.args.rand_seed)
                )
            )
        self.last_ckpoint = os.path.join(
            self.ckpoint_folder, 
            'last{}.ckpt.pth'.format('_seed{}'.format(self.args.rand_seed))
            )
        log_file_name = 'log'
        log_file_name += '{}.txt'.format(
            "".join(
                ["_seed{}".format(self.args.rand_seed),
                '_target' if self.args.test_target else '']
                )
            )
        self.log_path = os.path.join(self.ckpoint_folder, log_file_name)
        print('log path', self.log_path)
    

    def set_optimizer_and_scheduler(self, paras, lr, SGD=False, \
        momentum=0.9, weight_decay=5e-4, nesterov=False, \
        scheduler_name='', step_size=20, gamma=0.1, milestones=(30,60), \
        n_epoch=30, power=2):
        if SGD:
            optimizer = optim.SGD(
                paras, lr=lr, momentum=momentum, 
                weight_decay=weight_decay #, nesterov=nesterov
                )
        else:
            optimizer = optim.Adam(paras, lr=lr)

        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=step_size, gamma=gamma
                )
        elif scheduler_name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=gamma
                )
        elif scheduler_name == 'CosLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_steps_per_epoch * n_epoch
                )
        elif not scheduler_name:
            scheduler = None
        else:
            raise NotImplementedError()

        return optimizer, scheduler


    def train(self, net, trainset, trainloaders, validloaders, \
        testloaders=None, data_mean=None, data_std=None, net_paras=None):
        """
        Training a classfication CNN with ALT

        :net: base model
        :trainset:  
        :trainloaders:
        :validloaders:
        :testloaders:
        :data_mean: mean of dataset (a vector of 3 for color images)
        :data_std: std of dataset (a vector of 3 for color images)
        :net_paras: optional, the paprameter of base model to be optimized. 
                          Use it when customized training needed
        :return:
        """

        self.data_mean = torch.tensor(data_mean).reshape(3,1,1).to(self.device)
        self.data_std = torch.tensor(data_std).reshape(3,1,1).to(self.device)
        self.best_metric = 0  # best valid accuracy
        self.best_target_metric = 0  # best valid accuracy on target domain
        self.current_metric = 0
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch


        # r() rand_module
        self.rand_module = get_random_module(
           self.args, data_mean, data_std).to(self.device)

        self.bcnn_module = get_BCNN_module(self.args, data_mean, data_std).to(self.device)

        # g() trans module
        self.paras = []
        net = net.to(self.device)
        print("CLASSIFIER Summary")
        # summary(net, (3, self.args.image_size, self.args.image_size))
        if net_paras is not None:
            self.paras += list(net_paras)
        else:
            self.paras += [{'params': net.parameters()}]
        self.trainset = trainset


        if self.args.n_iter:
            self.n_steps_per_epoch = self.args.val_iter
        else:
            self.n_steps_per_epoch = max([len(loader) for loader in trainloaders])

        self.total_epoch = self.args.n_epoch 

                

        if not self.args.test:
            optimizer, scheduler = self.set_optimizer_and_scheduler(
                self.paras, 
                lr=self.args.lr, 
                SGD=self.args.SGD, 
                momentum=self.args.momentum, 
                weight_decay=self.args.weight_decay,
                nesterov=self.args.nesterov, 
                scheduler_name=self.args.scheduler, 
                step_size=self.args.step_size,
                gamma=self.args.gamma, 
                milestones=(30,60), 
                n_epoch=self.total_epoch, power=self.args.power
                )
            aug_optimizer, _ = self.set_optimizer_and_scheduler(
                self.bcnn_module.parameters(),
                lr=self.args.lr_adv
                )
            
            lr_factor = 1
            plot_cls_loss = []
            plot_kld_loss = []
            plot_adv_loss = []
            for epoch in trange(start_epoch, self.total_epoch, leave=True): 
                trainloaders, plots = self.train_one_epoch(
                    epoch, net, trainloaders, optimizer, scheduler, 
                    aug_optimizer
                    )
                plot_cls_loss = plot_cls_loss + plots[0]
                plot_kld_loss = plot_kld_loss + plots[1]
                plot_adv_loss = plot_adv_loss + plots[2]


                #  validation
                if epoch%self.args.val_freq==0 or epoch==self.total_epoch-1 or epoch==self.args.pre_epoch:
                    source_metric, target_metric = self.validate(
                        epoch, net, self.device, validloaders, 
                        self.args.n_val if self.args.val_with_rand else 1,
                        optimizer)

                    print(
                        "Source {}: {}, \nBest Source {}: {}, \ntarget {}: {},\nBest target {}: {}".format(
                            self.metric_name, source_metric, 
                            self.metric_name, self.best_metric, 
                            self.metric_name, target_metric,
                            self.metric_name, self.best_target_metric
                            )
                         )


                    x = np.linspace(0, epoch, len(plot_cls_loss))
                    fig, ax = plt.subplots()
                    ax.plot(x, plot_cls_loss, 'r', label='Loss$_{CLS}$')
                    ax.plot(x, plot_kld_loss, 'b', label='Loss$_{KLD}$')

                    ax.set_ylabel('Loss')
                    ax.set_xlabel('Epoch')
                    plt.legend(loc='best')
                    plt.savefig(
                        os.path.join(
                            self.ckpoint_folder,
                            'loss_seed{}.png'.format(self.args.rand_seed)
                            )
                        )

                    if epoch >= self.args.pre_epoch:
                        x = np.linspace(
                            self.args.pre_epoch, epoch, len(plot_adv_loss)
                            )
                        print(len(plot_adv_loss))
                        fig, ax = plt.subplots() 
                        ax.plot(x, plot_adv_loss, 'g', label='Loss$_{ADV}$')
                        ax.set_ylabel('Loss')
                        ax.set_xlabel('Epoch')
                        plt.savefig(
                            os.path.join(
                                self.ckpoint_folder,
                                'loss_adv_seed{}.png'.format(
                                    self.args.rand_seed)
                                )
                            )


                # if self.args.tiny and epoch > self.args.pre_epoch + 4:
                #     break
                if scheduler is not None and self.args.scheduler != 'CosLR':
                    scheduler.step()

        self.print_now()
        self.run_testing(net, testloaders)


    def run_testing(self, net, testloaders):
        print("\n========Testing Latest========")
        self.resume_model(net, test_latest=True)
        self.test(net, testloaders)
        self.print_now()

    def save_images(self, inputs, epoch, batch_idx, name='inputs'):
        samples_in = vutils.make_grid(inputs, nrow=8)
        vutils.save_image(
            samples_in, os.path.join(
                self.ckpoint_folder, 
                'seed{}'.format(self.args.rand_seed),
                '{}_{}_{}.jpg'.format(name, epoch, batch_idx)
                )
            ) 


    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    # Training one epoch
    # @staticmethod
    def train_one_epoch(self, epoch, net, trainloaders, optimizer, \
        scheduler=None, aug_optimizer=None):
        train_loss = 0

        if not isinstance(trainloaders, dict) and not isinstance(
            trainloaders, list):
            trainloaders = [trainloaders]

        dataiters = [iter(dataloader) for dataloader in trainloaders]
        metric_meter = AverageMeter(self.metric_name, ':6.2f')
        new_trainloaders = []
        counter_aug_epoch = 0

        ######## TRAIN PHASE ########
        net.train()
        self.bcnn_module.eval()

        plot_cls_loss = []
        plot_kld_loss = []
        plot_adv_loss = []

        with trange(self.n_steps_per_epoch, desc='Epoch[{}/{}]'.format(
            epoch, self.total_epoch)) as t:
            for batch_idx in t:
                loss = 0.0 
                loss_aug = 0.0
                loss_cl = 0.0
                # self.trans_module.eval()

                try:
                    inputs, targets = next(dataiters[0])
                except StopIteration:
                    # inputs, targets = tra
                    dataiters = [iter(dataloader) for dataloader in trainloaders]
                    inputs, targets = next(dataiters[0])
                inputs = inputs.to(self.device, non_blocking=True) 
                targets = targets.to(self.device, non_blocking=True)

                # visualize inputs from this dataloader       
                if batch_idx==0 and epoch%self.args.val_freq==0:
                    self.save_images(
                        inputs, epoch, batch_idx, 'inputs')                    

                outputs = net(inputs)
                loss = self.criterion(outputs, targets)
                metric = self.metric(outputs, targets)
                train_loss = loss.item()

                if batch_idx % int(0.05*len(t)) == 0:
                    plot_cls_loss.append(train_loss)


                if epoch >= self.args.pre_epoch and random.random() > self.args.toss:
                    #### g() 
                    net.eval()
                    self.bcnn_module.randomize()
                    self.bcnn_module.train()


                    # # g() adversarial maximization
                    x = inputs.clone()
                    x_g_old = torch.zeros_like(x)
                    for n in range(self.args.adv_steps):
                        net.zero_grad()
                        aug_optimizer.zero_grad()
                        self.bcnn_module.zero_grad()

                        x_g, bnn_kl = self.bcnn_module(inputs)
                        y_g = net(x_g)  

                        if torch.isnan(x_g).any():
                            print("x_g problem, sum: {}".format(
                                torch.isnan(x_g).sum()))
                        if torch.isnan(inputs).any():
                            print("inputs problem")
                        if torch.isnan(y_g).any():
                            print(torch.isnan(x_g).any())
                            print("y_g problem")


                        loss_aug = -self.criterion(y_g, targets) - self.args.elbo_beta * bnn_kl
                        if torch.isnan(loss_aug).any():
                            print("loss problem")
                        
                        plot_adv_loss.append(-loss_aug.item())

                        loss_aug.backward()
                        aug_optimizer.step()

                    # get final x_g output 
                    x_g, bnn_kl = self.bcnn_module(inputs)

                    if self.args.with_randconv:
                        #### r() randconv component
                        self.rand_module.randomize()
                        x_r = self.rand_module(inputs)
                    else:
                        x_r, bnn_kl_r = self.bcnn_module(inputs)
                    
                    # visualize a few augmented images
                    with torch.no_grad():
                        if batch_idx % int(0.2*len(t)) ==0:  
                            viz_g = x_g.detach().cpu()
                            viz_r = x_r.detach().cpu()
                            viz_x = inputs.detach().cpu()
                            self.save_images(viz_g, epoch, batch_idx, 'g')
                            self.save_images(viz_r, epoch, batch_idx, 'r')
                            self.save_images(viz_x, epoch, batch_idx, 'x')


                    #### consistency loss
                    outs_r = net(x_r) 
                    outs_g = net(x_g)

                    p_clean = F.softmax(outputs, dim=1)
                    p_r = F.softmax(outs_r, dim=1)
                    p_g = F.softmax(outs_g/self.args.temp, dim=1)

                    p_mix = (p_clean+ self.args.wr*p_r+ (2-self.args.wr)*p_g)/3
                    log_mix = torch.clamp(p_mix, 1e-4, 1).log() 
                    kl_clean = self.kl_div(log_mix, p_clean)
                    kl_r = self.kl_div(log_mix, p_r)
                    kl_g = self.kl_div(log_mix, p_g)

                    loss_cl = (kl_clean + self.args.wr*kl_r + (2-self.args.wr)*kl_g)/3




                    loss = (1 - self.args.clw) * loss + self.args.clw * loss_cl
      
                    train_loss = loss.item()
                    loss_cl = loss_cl.item()

                if batch_idx % int(0.05*len(t)) == 0:
                    plot_kld_loss.append(loss_cl)
                net.zero_grad()
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 

                if scheduler is not None and self.args.scheduler == 'CosLR':
                    scheduler.step()
                global_step = epoch*self.n_steps_per_epoch + batch_idx
                if global_step % 100 == 0:
                    self.writer.add_scalar(
                        'Loss/train_cls', 
                        train_loss / (batch_idx + 1), 
                        global_step)
                    self.writer.add_scalar(
                        '{}/train'.format(self.metric_name), 
                        metric_meter.avg, 
                        global_step)
                t.set_postfix_str(
                    'loss: {:.3f}'.format(train_loss / (batch_idx + 1))
                    )

                if self.args.tiny and batch_idx> int(0.05*len(t)):
                    sys.stdout.flush()
                    break
            
            plots = [plot_cls_loss, plot_kld_loss, plot_adv_loss]
        return trainloaders, plots

    def infer(self, net, device, testloader, criterion=None, with_rand=False, n_eval=1, name='', label_collection=None, epoch=None, save_viz=False):
        """
        base function for validation/testing with options of repeat multiple runs
        """
        net.eval()
        test_loss = 0
        metric_meter = AverageMeter(self.metric_name, ':6.2f')

        if self.bcnn_module is None:
            n_eval = 1

        loader_iter = iter(testloader)
        with torch.no_grad():
            for batch_idx in range(len(testloader)):
                inputs, targets = next(loader_iter)
                if type(label_collection) is list:
                    label_collection.append(targets)

                inputs, targets = inputs.to(device), targets.to(device)

                pred_batchs = []
                for i in range(n_eval):
                    outputs = net(inputs)                         
                    if save_viz and batch_idx%25==0 and epoch is not None and name==self.args.source:
                        aug_inputs, _ = self.bcnn_module(inputs)
                        rand_inputs = self.rand_module(inputs)
                        self.save_images(
                            inputs, epoch, batch_idx, "val_inputs"
                            )
                        self.save_images(
                            aug_inputs, epoch, batch_idx, "val_trans"
                            )
                        self.save_images(
                            rand_inputs, epoch, batch_idx, "val_rand"
                            )


                    _, predicted = outputs.max(1)
                    pred_batchs.append(predicted)

                    loss = criterion(outputs, targets)/n_eval
                    test_loss = test_loss + loss.item()

                # get the avg validation metric of n_eval runs
                metric = 0
                for pred_ in pred_batchs:
                    metric = metric + self.metric(pred_, targets)
                metric = metric/len(pred_batchs)
                metric_meter.update(metric, inputs.size(0))

        return metric_meter.avg*100, test_loss / (batch_idx + 1)

    def validate(self, epoch, net, device, validloaders, n_eval=1, optimizer=None):
        """"""
        if not isinstance(validloaders, dict) and not isinstance(validloaders, list):
            validloaders = [validloaders]

        net.eval()
        source_metrics, target_metrics = [], []
        global_step = (epoch+1) * self.n_steps_per_epoch

        with torch.no_grad():
            for i, loader in enumerate(validloaders):
                if type(validloaders) is dict:
                    name = loader
                    loader = validloaders[name]
                else:
                    name = None

                save_viz = name==self.args.source
                print("Saving Val Vizualization") 

                metric_temp, loss_temp = self.infer(
                    net, device, loader, self.criterion, 
                    with_rand=self.args.val_with_rand, 
                    n_eval=n_eval, name=name, epoch=epoch, save_viz=save_viz)

                if name is not None and name in self.args.source:
                    source_metrics.append(metric_temp)
                if name is not None and name not in self.args.source:
                    target_metrics.append(metric_temp)
                    print(name, metric_temp)
                if name is None:
                    source_metrics.append(metric_temp)
                self.writer.add_scalar(
                    '{}/seed{}/{}_valid_{}'.format(
                        self.ckpoint_folder, 
                        self.args.rand_seed, 
                        self.metric_name, 
                        name if name is not None else i
                        ), 
                    metric_temp, global_step)


        # Save checkpoint.
        if source_metrics:
            source_metric = np.mean(source_metrics)
            self.writer.add_scalar(
                '{}/valid_avg'.format(self.metric_name), 
                source_metric, global_step
                )
        else:
            source_metric = 0
        print('Saving..')
        if not os.path.isdir(self.ckpoint_folder):
            os.makedirs(self.ckpoint_folder)
        state = {
            'net': net.state_dict(),
            self.metric_name: source_metric,
            'epoch': epoch,
        }

        if self.bcnn_module is not None:
            state['bcnn_module'] = self.bcnn_module.state_dict()

        if source_metric >= self.best_metric:
            print('Best validation {}!'.format(self.metric_name))
            self.best_metric = source_metric
            torch.save(state, self.best_ckpoint)

        if target_metrics:
            target_metric = np.mean(target_metrics)
            if target_metric > self.best_target_metric:
                self.best_target_metric = target_metric
                state[self.metric_name] = target_metric
                print('Best validation {} on target domain!'.format(self.metric_name))
                torch.save(state, self.best_target_ckpoint)
        else:
            target_metric = 0

        if (epoch + 1) % self.args.val_freq == 0 or epoch == self.total_epoch-1:
            if optimizer:
                state['optimizer'] = optimizer.state_dict()
            torch.save(state, self.last_ckpoint)

        if epoch==self.args.pre_epoch:
            torch.save(
                state, 
                os.path.join(
                    self.ckpoint_folder, 
                    'epoch_{}_seed_{}.ckpt.pth'.format(
                        self.args.pre_epoch, 
                        self.args.rand_seed
                        )
                    )
                )

        return source_metric, target_metric

    def test(self, net, testloaders):

        metrics = {name: 0 for name in testloaders.keys()}
        metrics["Source"] = 0.0
        metrics["Target Average"] = 0.0
        with tqdm(testloaders.items(), leave=False, desc='Domains: ') as d_iter:
            for name, loader in d_iter:

                labels = None

                metric, _ = self.infer(
                    net, self.device, loader, criterion=self.criterion, 
                    name=name, label_collection=labels)
                metrics[name] = metric
                if name == self.args.source:
                    metrics["Source"] = metric
                else:
                    metrics["Target Average"] += metric/(len(testloaders) - 1)
                d_iter.set_postfix_str("Domain {}: {} {:.3f}".format(name, self.metric_name, metric))
        logs = []
        logs.append("\t".join(metrics.keys()))

        numbers = []
        for name, metric in metrics.items():
            numbers.append("{:.3f}".format(metric))
        logs.append("\t".join(numbers))
        logs.append('Target Domain average performance: {:.3f}'.format(
            metrics["Target Average"]
            )
        )
        self.log('\n'.join(logs))
        # also save as json
        with open(self.log_path[:-4]+'.json', 'w') as f:
            json.dump(metrics, f, indent=4)


    def test_corrupted(self, net, testloaders, severity=1):
        """
        Testing on corrupted data
        """
        c_types = testloaders.keys()
        metrics = {name: [] for name in c_types}
        with tqdm(testloaders.items(), leave=True, desc='Severity {}: '.format(severity)) as d_iter:
            for name, loader in d_iter:
                metric, _ = self.infer(net, self.device, loader, name='{}-{}'.format(severity, name.split('_')[0]))
                metrics[name].append(metric)
                d_iter.set_postfix_str("Corruption {}: {} {:.3f}".format(name, self.metric_name, metric))

        target_metric_avg = 0
        target_count = 0
        for name, metric in metrics.items():
            a = np.array(metric)
            target_count += 1
            target_metric_avg += a

        logs = []
        logs.append("\t".join(metrics.keys()))

        if target_count > 0:
            target_metric_avg /= target_count

        numbers = []
        for name, metric in metrics.items():
            numbers.append("{:.3f}".format(metric[0]))
        logs.append("\t".join(numbers))
        if target_count > 0:
            logs.append('Corruption severity {} average performance: {:.3f}'.format(severity, target_metric_avg[0]))
        self.log('\n'.join(logs))

    def resume_model(self, net, test_latest=False, test_target=False):
        # Load checkpoint for testing
        assert os.path.isdir(self.ckpoint_folder), 'Error: no checkpoint directory {} found!'.format(self.ckpoint_folder)
        if test_latest:
            ckpoint_file = self.last_ckpoint
        elif test_target:
            ckpoint_file = self.best_target_ckpoint
        else:
            ckpoint_file = self.best_ckpoint

        print('==> Resuming from checkpoint {}'.format(self.ckpoint_folder))
        checkpoint = torch.load(ckpoint_file)
        if self.bcnn_module is not None:
            self.bcnn_module.load_state_dict(
                checkpoint['bcnn_module'], strict=False)

        net.load_state_dict(checkpoint['net'], strict=False)
        best_metric = checkpoint[self.metric_name]
        best_epoch = checkpoint['epoch']
        print("{} {} at {} epoch".format(
            self.metric_name, best_metric, best_epoch))

        # log epoch of tested checkpoint
        if test_latest:
            self.log("\nThe last model at {} epoch".format(best_epoch))
        else:
            self.log("\nThe best model at {} epoch".format(best_epoch))


    def log(self, string):
        print(string)
        with open(self.log_path, mode='a') as f:
            f.write(str(string))
            f.write('\n')

    @staticmethod
    def print_now():
        print(datetime.now())
