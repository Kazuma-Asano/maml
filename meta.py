# # coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
from copy import deepcopy

# from learner import Learner
from network import define_network

class Meta(nn.Module):

    def __init__(self, opt):
        super(Meta, self).__init__()
        # (layer, [])
        config = [
            ('conv2d', [32, 3, 3, 3, 1, 0]), # conv2d:(ch_in:, ch_out:, k:, stride:, padding:)
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 2, 0]), # max_pool2d:(k:, stride:, padding:)
            ('conv2d', [32, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
            ('max_pool2d', [2, 1, 0]),
            ('flatten', []),
            ('linear', [opt.n_way, 32 * 5 * 5])
        ]
        self.update_lr = opt.update_lr
        self.meta_lr = opt.meta_lr
        self.n_way = opt.n_way # クラス数 default: 5 way
        self.k_spt = opt.k_spt # k shot default: 1 shot
        self.k_qry = opt.k_qry # query set ラベルなしデータ(これらをうまくnクラス分類できるようにする) default: 15 shot
        self.task_num = opt.task_num # meta batch size default: 4
        self.update_step = opt.update_step # task-level inner update steps (通常の学習でいえばミニバッチ？) default: 5
        self.update_step_test = opt.update_step_test # default: 10

        # self.net = Learner(config, opt.img_c, opt.img_size)
        self.net = define_network(num_class=self.n_way, pretrained=False)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad     : list of gradients
        :param max_norm : maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0

        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item()**2
            counter += 1
        total_norm = total_norm ** (1./2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w] ちっさいTrain 入力画像
        :param y_spt:   [b, setsz] ラベル
        :param x_qry:   [b, querysz, c_, h, w] 入力画像
        :param y_qry:   [b, querysz] ラベル
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)] # 初期化
        corrects = [0 for _ in range(self.update_step + 1)] # 初期化

        # タスク分回す
        for i in range(task_num):
            # 1. i番目のタスクを実行 (k = 0)
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # first_weight update前の loss & accuracy
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # first_weight update後の loss & accuracy
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True) # 更新後のパラメータを適用
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # i番目のタスクを実行 (k = 1 ~ k-1)
                logits = self.net(x_spt[i], fast_weights, bn_training=True)  # ここfast_weightsでいいの？？
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training = True)

                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k+1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k+1] = corrects[k+1] + correct

        # End all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # OPTIMIZE parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('Meta Update')
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        net = deepcopy(self.net)


        # mini train
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # first_weight update前の loss & accuracy
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # first_weight update後の loss & accuracy
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        ################  ################
        for k in range(1, self.update_step_test):
            # 1. i番目のタスクを実行
            logits = net(x_spt, fast_weights, bn_training=True) # ここfast_weightsでいいの？？
            loss = F.cross_entropy(logits, y_spt)

            # 2. gradの計算
            grad = torch.autograd.grad(loss, fast_weights)

            # 3.
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training = True)

            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            #losses_q[k+1] += loss_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k+1] = corrects[k+1] + correct

        del net

        accs = np.array(corrects) / querysz
        return accs

    ################################################################################
    def checkpoint(epoch, model):
        checkpointDir = './checkpoint/'
        os.makedirs(checkpointDir, exist_ok=True)
        model_out_path = './checkpoint/model_epoch{:03}.pth'.format(epoch)
        torch.save(model.state_dict(), model_out_path)
        print('---Checkpoint saved---\n')

if __name__ == '__main__':
    pass
