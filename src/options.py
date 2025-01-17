#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=20, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    
    # Global optimizer
    parser.add_argument('--global_opt', type=str, default='adagrad', help="which is the global optimizer (adagrad, adam, yogi)")
    parser.add_argument('--global_opt_beta1', type=float, default=0.9, help="parameter beta1 for adaptive optimizers")
    parser.add_argument('--global_opt_beta2', type=float, default=0.99, help="parameter beta2 for adaptive optimizers")
    parser.add_argument('--global_opt_tau', type=float, default=1e-7, help="tau for adaptive optimizers")

    # Model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # Other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--local_optimizer', type=str, default='sgd', help="type of local optimizer")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits for  non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
