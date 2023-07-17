#!/usr/bin/env python

from .digit_net import DigitNet

network_map = {

    'digit': DigitNet,

}


def get_network(name, **kwargs):
    # if 'vgg' in name:
    #     return get_vgg(name, **kwargs)

    if name not in network_map:
        raise ValueError('Name of network unknown %s' % name)


    return network_map[name](**kwargs)


