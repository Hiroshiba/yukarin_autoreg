from typing import List

import chainer
import chainer.functions as F
import chainer.links as L


class UpConv(chainer.ChainList):
    def __init__(self, scales: List[int], c_ksize: int = 3, **kwargs) -> None:
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

    def __call__(self, x):
        """
        :param x: (batch_size, lN, ?)
        :return: (batch_size, N, ?)
        """
        x = F.expand_dims(x, axis=1)  # shape: (batch_size, 1, lN, ?)
        for c in self:
            x = F.relu(c(x))
        x = F.squeeze(x, axis=1)  # shape: (batch_size, lN, ?)
        return x
