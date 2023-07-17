#!/usr/bin/env python
"""
Created by zhenlinxu on 11/23/2019
"""

from torchvision import transforms
from torchvision.datasets import MNIST, SVHN
from torchvision.datasets.usps import USPS

from .transforms import GreyToColor, IdentityTransform, ToGrayScale
from .mnist_m import MNISTM
from .synth_digit import SynthDigit
from .mnist_c import MNISTC
from .mnist_10k import MNIST10K


mnist = 'mnist'
mnist10k = 'mnist10k'
mnist_m = 'mnist_m'
mnist_c = 'mnist_c'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'


def get_dataset(name, **kwargs):


    if name == mnist:
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard]),
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = MNIST(**kwargs)

    elif name == mnist10k:
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard]),
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = MNIST10K(**kwargs)

    elif name == mnist_m:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']

        data = MNISTM(**kwargs)
    elif name == svhn:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if not kwargs['train']:
            kwargs['split'] = 'test'
        del kwargs['train']
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = SVHN(**kwargs)

    elif name == usps:
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            # transforms.Normalize(dataset_mean[name], dataset_std[name])
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
            # transforms.Normalize(dataset_mean[mnist], dataset_std[mnist])

        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = USPS(**kwargs)

    elif name == synth:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard])
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'download' in kwargs:
            del kwargs['download']
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = SynthDigit(**kwargs)

    elif name == mnist_c:
        if 'grey' in kwargs:
            grey = kwargs['grey']
        else:
            grey = False
        transform = transforms.Compose([
            transforms.Resize(kwargs['size']) if 'size' in kwargs else IdentityTransform(),
            ToGrayScale(3) if grey else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            transforms.Normalize(dataset_mean[standard], dataset_std[standard]),
        ])
        if 'transform' not in kwargs:
            kwargs['transform'] = transform
        if 'size' in kwargs:
            del kwargs['size']
        if 'grey' in kwargs:
            del kwargs['grey']
        data = MNISTC(**kwargs)


    else:
        raise NotImplementedError('{} data does not exists'.format(name))
    return data
