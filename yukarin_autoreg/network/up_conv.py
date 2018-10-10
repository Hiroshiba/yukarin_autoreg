from typing import List

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class UpConv(chainer.ChainList):
    def __init__(self, scales: List[int], c_ksize: int = 3, residual: bool = False, **kwargs) -> None:
        super().__init__(*[
            L.Deconvolution2D(
                in_channels=1,
                out_channels=1,
                ksize=(s, c_ksize),
                stride=(s, 1),
                pad=(0, (c_ksize - 1) // 2),
                **kwargs,
            )
            for s in scales
        ])
        self.upsample = lambda x: \
            F.unpooling_2d(x, ksize=(np.prod(scales), 1), stride=(np.prod(scales), 1), cover_all=False)

        self.residual = residual

    def __call__(self, x):
        """
        :param x: (batch_size, lN, ?)
        :return: (batch_size, N, ?)
        """
        x = F.expand_dims(x, axis=1)  # shape: (batch_size, 1, lN, ?)
        h = x
        for c in self:
            h = F.relu(c(h))

        if self.residual:
            h += self.upsample(x)

        h = F.squeeze(h, axis=1)  # shape: (batch_size, N, ?)
        return h
