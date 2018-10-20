from functools import partial
from typing import Callable, Dict, Optional

import chainer
import numpy as np
from chainer.iterators import MultiprocessIterator
from chainer.training.updaters import StandardUpdater

from yukarin_autoreg.config import LossConfig
from yukarin_autoreg.dataset import BaseWaveDataset
from yukarin_autoreg.model import Model


class RandomDataset(BaseWaveDataset):
    def __len__(self):
        return 100

    def get_example(self, i):
        length = self.sampling_length
        wave = np.random.rand(length) * 2 - 1
        local = np.empty(shape=(length, 0), dtype=np.float32)
        silence = np.zeros(shape=(length,), dtype=np.bool)
        return self.convert_to_dict(wave, silence, local)


class LocalRandomDataset(RandomDataset):
    def get_example(self, i):
        d = super().get_example(i)
        d['local'] = np.stack((
            np.r_[np.NaN, d['target_coarse'].astype(np.float32) / 256],
            np.r_[np.NaN, d['target_fine'].astype(np.float32) / 256],
        ), axis=1)
        return d


class DownLocalRandomDataset(LocalRandomDataset):
    def __init__(self, scale: int, **kwargs) -> None:
        self.scale = scale
        super().__init__(**kwargs)

    def get_example(self, i):
        d = super().get_example(i)
        l = np.reshape(d['local'], (-1, self.scale * d['local'].shape[1]))
        l[np.isnan(l)] = 0
        d['local'] = l
        return d


def _create_optimizer(model):
    optimizer = chainer.optimizers.Adam(alpha=0.005)
    optimizer.setup(model)
    return optimizer


def setup_support(
        batch_size: int,
        gpu: Optional[int],
        network: chainer.Link,
        dataset: chainer.dataset.DatasetMixin,
):
    loss_config = LossConfig(clipping=None)
    model = Model(loss_config=loss_config, predictor=network)
    optimizer = _create_optimizer(model)
    train_iter = MultiprocessIterator(dataset, batch_size)

    if gpu is not None:
        model.to_gpu(gpu)

    converter = partial(chainer.dataset.convert.concat_examples)
    updater = StandardUpdater(
        device=gpu,
        iterator=train_iter,
        optimizer=optimizer,
        converter=converter,
    )

    reporter = chainer.Reporter()
    reporter.add_observer('main', model)

    return updater, reporter


def train_support(
        iteration: int,
        reporter: chainer.Reporter,
        updater: chainer.training.Updater,
        first_hook: Callable[[Dict], None] = None,
        last_hook: Callable[[Dict], None] = None,
):
    observation: Dict = {}
    for i in range(iteration):
        with reporter.scope(observation):
            updater.update()

        if i % 100 == 0:
            print(observation)

        if i == 0:
            if first_hook is not None:
                first_hook(observation)

    if last_hook is not None:
        last_hook(observation)
