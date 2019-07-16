# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def define_network(num_class, pretrained=False):

    model = VGG19()
    # パラメータの適用 or 初期化
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    else:
        model.apply(initialize_weights) # 初期化

    if num_class != 1000:
        model.classifier = nn.Sequential (
            nn.Linear(512*7*7, 4096), # forward内の print(out.size()) でサイズ確認
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )

    # if gpu_ids != None:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    #     model.cuda()

    return model


# 初期化コンポーネント
def initialize_weights(self):
    for m in self.modules():
        # 畳み込み層 (Conv2)の初期化
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # バッチノーマライゼーションの初期化
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # 全結合層の初期化
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

################################################################################
class VGG19(nn.Module):
    def __init__(self, num_class=1000):
        super(VGG19, self).__init__()
        ########################################################################
        self.features = nn.Sequential(
            # inout : w x h x 3ch(RGB)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True), # 活性化関数
            # 2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # max pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # max pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # max pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 9
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # max pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 13
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 14
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 15
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 16
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # max pooling
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential (
            nn.Linear(512*7*7, 4096), # 第一引数はforward内の print(out_features.size()) でサイズ確認
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )


    def forward(self, x):
        out_features = self.features(x)
        # print(out_features.size())
        out_view = out_features.view(out_features.size(0), -1) # 1次元配列化
        out_classifier = self.classifier(out_view)
        return out_classifier


if __name__ == '__main__':

    """
    Testing
    $python network.py
    """

    # モデルの用意
    model = define_network(num_class=10, pretrained=False)
    print(model)

    # データの用意
    x = torch.FloatTensor( np.random.random((1, 3, 224, 224))) # (batch_size, channels, width, height)

    # GPUが使えるならば
    if torch.cuda.is_available():
        print(' # on GPU #')
        model.cuda()
        x.cuda()

    out = model(x)

    softmax = nn.Softmax(dim=1) # 値を0 ~ 1にして確率化
    predicts = softmax(out)
    # print(predicts)
    #
    loss = torch.sum(predicts)
    print(loss)
    loss.backward()
