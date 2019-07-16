# coding:utf-8
import os
import random
import sys
import pickle
import argparse
from pprint import pprint

import numpy as np
from miniimagenet import MiniImagenet
from meta import Meta
import scipy.stats

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import torch.backends.cudnn as cudnn

from tqdm import tqdm
import time
from collections import OrderedDict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='epoch number', default=5)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--img_size', type=int, help='img_size', default=84)
    parser.add_argument('--img_c', type=int, help='imgc', default=3)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument('--cuda', action='store_true', help='use cuda?')

    option = parser.parse_args()
    pprint(option)
    return option

################################################################################
def train(mini, maml, opt, device):
    train_db = DataLoader(mini, opt.task_num, shuffle=True, num_workers=1, pin_memory=True)

    with tqdm(train_db) as db:
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            db.set_postfix(OrderedDict(train_acc='{}'.format(accs))) # tqdmの表示の値を追加

################################################################################
def test(mini_test, maml, opt, device):
    test_db = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)

    with tqdm(test_db) as db:
        accs_all_test = []
        for x_spt, y_spt, x_qry, y_qry in db:
            x_spt, y_spt = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device)
            x_qry, y_qry = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)

            db.set_postfix(OrderedDict(Test_acc='{}'.format(accs))) # tqdmの表示の値を追加

            # [b, update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Test acc:')
        pprint(accs)




if __name__ == '__main__':
    opt = get_parser()
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("GPU is not found, please run without --cuda")

    cudnn.benchmark = True

    if opt.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = Meta(opt).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    #  # batchsz here means total episode number
    mini = MiniImagenet('./miniimagenet/',
                        mode='train',
                        n_way=opt.n_way,
                        k_shot=opt.k_spt,
                        k_query=opt.k_qry,
                        batchsz=10000,
                        resize=opt.img_size)

    mini_test = MiniImagenet('./miniimagenet/',
                             mode='test',
                             n_way=opt.n_way,
                             k_shot=opt.k_spt,
                             k_query=opt.k_qry,
                             batchsz=100,
                             resize=opt.img_size)

    for epoch in range(1, opt.epoch+1):
        train(mini, maml, opt, device)
        test(mini_test, maml, opt, device)

    print('-- Finish Training --')
