#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def global_adagrad(args, epoch, w, w_global_prev, m_t_prev, v_t_prev):
    """
    FedAdaGrad: Global optimizer = AdaGrad
    """
    if args.global_opt == 'adagrad':
        beta1 = args.global_opt_beta1
        
    if epoch == 0:
        w_global = average_weights(w)
        m_t = None
        v_t = None
    elif epoch == 1:
        w_avg = average_weights(w)
        w_global = copy.deepcopy(w_avg)
        
        # Delta_t = w_avg - w_global_prev
        Delta_t = copy.deepcopy(w_avg)
        for key in Delta_t.keys():
            Delta_t[key] = Delta_t[key] - w_global_prev[key]
            
        m_t = Delta_t
        
        # v_t = Delta_t^2
        v_t = copy.deepcopy(Delta_t)
        for key in Delta_t.keys():
            v_t[key] = torch.square(Delta_t[key])
    else:
        w_avg = average_weights(w)
        
        # Delta_t = w_avg - w_global_prev
        Delta_t = copy.deepcopy(w_avg)
        for key in Delta_t.keys():
            Delta_t[key] = Delta_t[key] - w_global_prev[key]
            
        # m_t = beta1 * m_t_prev + (1-beta1) * Delta_t
        m_t = copy.deepcopy(Delta_t)                       # simplified case with beta1 = 0.0
            
        # v_t = v_t_prev + Delta_t**2
        v_t = copy.deepcopy(Delta_t)
        for key in Delta_t.keys():
            v_t[key] = v_t_prev[key] + torch.square(Delta_t[key])
        
        # w_global = w_global_prev + lr * (m_t/tau + sqrt(v_t))
        w_global = copy.deepcopy(w_global_prev)
        for key in w_global.keys():
            w_global[key] = w_global_prev[key] + args.lr * torch.div(m_t[key], args.tau + torch.sqrt(v_t[key]))
            
    return w_global, m_t, v_t


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
