#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import numpy as np
import random
import collections
from torch.nn import Parameter

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).mean()
    return kl

class BConvModule(nn.Module):
    def __init__(self, kernel_size=3, in_channels=3, out_channels=3,
                 rand_bias=False,
                 mixing=False, sample=1,
                 identity_prob=0.0,
                 data_mean=None, data_std=None, clamp_output=False,mixing_ratio = None
                 ):
        """
        :param kernel_size:
        :param in_channels:
        :param out_channels:
        :param rand_bias:
        :param mixing: "random": output = (1-alpha)*input + alpha* randconv(input) where alpha is a random number sampled
                            from a distribution defined by res_dist
        :param identity_prob:
        :param data_mean:
        :param data_std:
        :param clamp_output:
        """

        super(BConvModule, self).__init__()

        # if the input is not normalized, we need to normalized with given mean and std (tensor of size 3)
        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

        # adjust output range based on given data mean and std, (clamp or norm)
        # clamp with clamp the value given that the was image pixel values [0,1]
        # normalize will linearly rescale the values to the allowed range
        # The allowed range is ([0, 1]-data_mean)/data_std in each color channel
        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (self.data_std is not None), "Need data mean/std to do output range adjust"
        self.register_buffer('range_up', None if not self.clamp_output else (torch.ones(3).reshape(3, 1, 1) - self.data_mean) / self.data_std)
        self.register_buffer('range_low', None if not self.clamp_output else (torch.zeros(3).reshape(3, 1, 1) - self.data_mean) / self.data_std)

        if isinstance(kernel_size, collections.Sequence) and len(kernel_size) == 1:
            kernel_size = kernel_size

        # if mixing:
        #     out_channels = in_channels

        # generate random conv layer
        print("Add Bayes Conv layer with kernel size {}, output channel {}".format(kernel_size, out_channels))

        self.bconv = MultiScaleBConv2d(in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_size,
                                             stride=1, rand_bias=rand_bias,
                                             clamp_output=self.clamp_output,
                                             range_low=self.range_low,
                                             range_up=self.range_up,
                                            sample=sample
                                             )


        # mixing mode
        self.mixing = mixing # In the mixing mode, a mixing connection exists between input and output of random conv layer
        self.sample = sample
        self.mixing_ratio = mixing_ratio
        # self.res_dist = res_dist
        self.res_test_weight = None
        if self.mixing:
            assert in_channels == out_channels or out_channels == 1, \
                'In mixing mode, in/out channels have to be equal or out channels is 1'
            if mixing_ratio is None:
                self.alpha = random.random()  # sample mixing weights from uniform distributin (0, 1)
            else:
                self.alpha = mixing_ratio

        self.identity_prob = identity_prob  # the probability that use original input
        if self.mixing:
            print("Mixing output")
        if self.clamp_output:
            print("Clamp output, range from {} to {}".format(self.range_low, self.range_up))


    def forward(self, input):
        """assume that the input is whightened"""
        kl_loss = 0

        ######## random conv ##########
        if not (self.identity_prob > 0 and torch.rand(1) < self.identity_prob):
            # whiten input and go through randconv
            output, kl_loss = self.bconv(input)

            if self.mixing:
                if self.sample !=1:
                    output = (self.alpha*output + (1-self.alpha)*input.unsqueeze(1).repeat(1, self.sample, 1, 1, 1))
                else:
                    output = self.alpha * output + (1 - self.alpha) * input
                output = torch.max(
                    torch.min(output, torch.tensor(1.0).cuda()),
                    torch.tensor(0.0).cuda()
                )

        else:
            output = input

        return output, kl_loss

    def parameters(self, recurse=True):
        return self.bconv.parameters()

    def trainable_parameters(self, recurse=True):
        return self.bconv.trainable_parameters()

    def whiten(self, input):
        return (input - self.data_mean) / self.data_std

    def dewhiten(self, input):
        return input * self.data_std + self.data_mean

    def randomize(self):
        self.bconv.randomize()

        if self.mixing and (self.mixing_ratio is not None):
            self.alpha = random.random()

    def set_test_res_weight(self, w):
        self.res_test_weight = w

class BConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rand_bias=True,
                 clamp_output=None, range_up=None, range_low=None, sample=1, priors=None, device='cuda', **kwargs):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param rand_bias:
        :param clamp_output:
        :param range_up:
        :param range_low:
        :param kwargs:
        """
        super(BConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=rand_bias, **kwargs)

        self.use_bias = rand_bias
        self.sample = sample
        self.device = device


        self.clamp_output = clamp_output
        self.register_buffer('range_up', None if not self.clamp_output else range_up)
        self.register_buffer('range_low', None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 1/math.sqrt(3*kernel_size*kernel_size),
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-1, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

    def randomize(self):

        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)


    def forward(self, input):

        sample = self.sample
        if self.training or sample:
            W_mu = self.W_mu.unsqueeze(0).repeat(sample, 1, 1, 1, 1)
            W_rho = self.W_rho.unsqueeze(0).repeat(sample, 1, 1, 1, 1)
            W_eps = torch.empty(W_mu.size()).normal_(0, 1).to(self.device)
            W_sigma = torch.log1p(torch.exp(W_rho))
            weight = W_mu + W_eps * W_sigma
            weight = weight.flatten(0, 1)

            if self.use_bias:
                bias_mu = self.bias_mu.unsqueeze(1).repeat(sample, 1)
                bias_rho = self.bias_rho.unsqueeze(1).repeat(sample, 1)
                bias_eps = torch.empty(bias_mu.size()).normal_(0, 1).to(self.device)
                bias_sigma = torch.log1p(torch.exp(bias_rho))
                bias = bias_mu + bias_eps * bias_sigma
                bias = bias.flatten(0, 1)
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        if sample != 1:
            out = out.view(out.size()[0], sample, out.size()[1] // sample, out.size()[2], out.size()[3])
        output = out


        if self.clamp_output == 'clamp':
            output = torch.max(torch.min(output, self.range_up), self.range_low)
        elif self.clamp_output == 'norm':
            output_low = torch.min(torch.min(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            output_up = torch.max(torch.max(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
            output = (output - output_low)/(output_up-output_low)*(self.range_up-self.range_low) + self.range_low

        kl_loss = self.kl_loss()

        return output, kl_loss

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, torch.log1p(torch.exp(self.W_rho)))
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, torch.log1p(torch.exp(self.bias_rho)))
        return kl





class MultiScaleBConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes,
                 rand_bias=True, clamp_output=False, range_up=None, range_low=None, sample=1, **kwargs
                 ):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size: sequence of kernel size, e.g. (1,3,5)
        :param bias:
        """
        super(MultiScaleBConv2d, self).__init__()

        self.clamp_output = clamp_output
        if clamp_output is True:
            clamp_output = 'clamp'
            range_up = torch.tensor(1.0).cuda()
            range_low=torch.tensor(0.001).cuda()
        self.register_buffer('range_up', None if not self.clamp_output else range_up)
        self.register_buffer('range_low', None if not self.clamp_output else range_low)
        if self.clamp_output:
            assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"

        self.multiscale_rand_convs = nn.ModuleDict(
            {str(kernel_size): BConv2d(in_channels, out_channels, kernel_size, padding = kernel_size // 2,
                                          rand_bias=rand_bias,
                                          clamp_output=clamp_output,
                                          range_low=self.range_low, range_up=self.range_up, sample=sample,
                                          **kwargs) for kernel_size in kernel_sizes})

        self.scales = kernel_sizes
        self.n_scales = len(kernel_sizes)
        self.randomize()

    def randomize(self):
        self.current_scale = str(self.scales[random.randint(0, self.n_scales-1)])
        self.multiscale_rand_convs[self.current_scale].randomize()

    def forward(self, input):
        output, kl_loss = self.multiscale_rand_convs[self.current_scale](input)
        return output, kl_loss



class data_whiten_layer(nn.Module):
    def __init__(self, data_mean, data_std):
        super(data_whiten_layer, self).__init__()
        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

    def forward(self, input):
        return (input - self.data_mean) / self.data_std

if __name__ == '__main__':
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    layer = BConvModule(in_channels=3,
                             out_channels=3,
                             kernel_size=[1, 3, 5, 7],
                             mixing=False,
                            sample = 1,
                            identity_prob=0.0,
                            rand_bias=False,
                            data_mean=data_mean,
                            data_std=data_std,
                            clamp_output=True,
                           mixing_ratio=0.5)
    layer.randomize()
    a = torch.normal(0,1, [1,3,128,128]).cuda()
    c = layer(a)
    print(c[0].shape)

