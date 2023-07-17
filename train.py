import os
import sys
sys.path.append(os.path.abspath(''))
import random
import argparse
from copy import copy
import torch
import torchvision 
from torchvision.datasets import ImageFolder
from torch.utils.data import dataloader

from lib.datasets.transforms import GreyToColor, IdentityTransform, ToGrayScale, LaplacianOfGaussianFiltering


from trainer_bnn import *
from lib.networks import get_network
from lib.utils.config import *

def main(args):
    # GPU and random seed
    print("Random Seed: ", args.rand_seed)
    if args.rand_seed is not None:
        random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        print(args.gpu_ids, type(args.gpu_ids))
        if type(args.gpu_ids) is list and len(args.gpu_ids) >= 0:
            torch.cuda.manual_seed_all(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.rand_seed)
        torch.set_num_threads(1)

    # DATALOADERS

    assert args.n_classes==10
    data_dir = "../MNIST/"
    domains = ['mnist10k', 'mnist_m', 'svhn', 'usps', 'synth']
    trg_domains = [dd for dd in domains if dd!=args.source]
    stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    print(args.data_name)
    print("SRC:{}; TRG:{}".format(args.source, domains))

    # transforms
    trans_list = []
    trans_list.append(
        transforms.RandomResizedCrop(args.image_size, scale=(0.5, 1))
        )
    if args.colorjitter:
        trans_list.append(transforms.ColorJitter(*[args.colorjitter] * 4))
    if args.data_name != 'digits':
        trans_list.append(transforms.RandomHorizontalFlip())
    if args.grey:
        trans_list.append(ToGrayScale(3))
    trans_list.append(transforms.ToTensor())
    if args.data_name=='digits':
        trans_list.append(GreyToColor())
    trans_list.append(transforms.Normalize(*stats))
    train_transform = transforms.Compose(trans_list)


    test_transform  = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        ToGrayScale(3) if args.grey else IdentityTransform(),
        transforms.ToTensor(),
        GreyToColor() if args.data_name=='digits' else IdentityTransform(),
        transforms.Normalize(*stats)
        ])

    ## datasets
    print("\n=========Preparing Data=========")
    assert args.source in domains, 'allowed data_name {}'.format(domains)


    trainset = get_dataset(
        args.source, root=data_dir, train=True, download=True,
        transform=train_transform
        )
    validsets = {
        domain: get_dataset(
            domain, root=data_dir, train=False, download=True, transform=test_transform) for domain in domains}

    testsets = validsets



    trainloaders = [
        torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, 
            num_workers=4
            )
        ]
    validloaders = {
        d: torch.utils.data.DataLoader(
            validsets[d], batch_size=args.batch_size, shuffle=False, 
            num_workers=2
            ) for d in validsets.keys()
        }
    testloaders = {
        d: torch.utils.data.DataLoader(
            testsets[d], batch_size=args.batch_size, shuffle=False, 
            num_workers=2
            ) for d in testsets.keys()
        }

    # MODEL
    print("\n=========Building Model=========")

    net = get_network(
            name=args.net, num_classes=args.n_classes, pretrained=True
            )

    trainer = BAL(args)
    trainer.train(
        net, 
        trainset,
        trainloaders, validloaders, testloaders=testloaders, 
        data_mean=(0.5, 0.5, 0.5), data_std=((0.5, 0.5, 0.5))
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    add_basic_args(parser)
    args = parser.parse_args()
    print(args.source)
    main(args)
