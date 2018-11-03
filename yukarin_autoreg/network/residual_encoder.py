from typing import List

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class ResBlock(chainer.Chain):
    def __init__(self, n_channel: int, **kwargs) -> None:
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=n_channel, out_channels=n_channel, ksize=1, nobias=True, **kwargs)
            self.conv2 = L.Convolution2D(in_channels=n_channel, out_channels=n_channel, ksize=1, nobias=True, **kwargs)
            self.bn1 = L.BatchNormalization(n_channel)
            self.bn2 = L.BatchNormalization(n_channel)

    def __call__(self, x):
        h = x
        h = F.relu(self.bn1(self.conv1(h)))
        h = self.bn2(self.conv2(h))
        return h + x


class ResNet(chainer.ChainList):
    def __init__(self, n_channel: int, num_block: int, **kwargs) -> None:
        super().__init__(*[ResBlock(n_channel=n_channel, **kwargs) for _ in range(num_block)])

    def __call__(self, x):
        h = x
        for r in self:
            h = r(h)
        return h


class ResidualEncoder(chainer.Chain):
    def __init__(self, n_channel: int, num_block: int, **kwargs) -> None:
        super().__init__()
        with self.init_scope():
            self.pre_conv = L.Convolution2D(None, n_channel, ksize=1, nobias=True, **kwargs)
            self.pre_bn = L.BatchNormalization(n_channel)
            self.res_net = ResNet(n_channel=n_channel, num_block=num_block, **kwargs)
            self.post_conv = L.Convolution2D(n_channel, n_channel, ksize=1, **kwargs)

    def __call__(self, x):
        """
        :param x: (batch_size, lN, ?)
        :return: (batch_size, lN, n_channel)
        """
        h = F.expand_dims(F.transpose(x, (0, 2, 1)), axis=3)  # shape: (batch_size, ?, lN, 1)
        h = F.relu(self.pre_bn(self.pre_conv(h)))  # shape: (batch_size, n_channel, lN, 1)
        h = self.res_net(h)  # shape: (batch_size, n_channel, lN, 1)
        h = self.post_conv(h)  # shape: (batch_size, n_channel, lN, 1)
        h = F.transpose(F.squeeze(h, axis=3), (0, 2, 1))  # shape: (batch_size, lN, n_channel)
        return h
