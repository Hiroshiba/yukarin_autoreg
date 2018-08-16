from functools import partial
from typing import Callable, Dict, Optional

import chainer
import numpy as np
from chainer.iterators import MultiprocessIterator
from chainer.training.updaters import StandardUpdater

from yukarin_autoreg.dataset import encode_16bit, normalize
from yukarin_autoreg.model import Model


class RandomDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sampling_length: int) -> None:
        self.sampling_length = sampling_length

    def __len__(self):
        return 100

    def get_example(self, i):
        length = self.sampling_length
        wave = np.random.rand(length) * 2 - 1
        local = np.empty(shape=(length, 0), dtype=np.float32)
        silence = np.zeros(shape=(length,), dtype=np.bool)

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
            local=local[1:],
            silence=silence[1:],
        )


class LocalRandomDataset(RandomDataset):
    def get_example(self, i):
        d = super().get_example(i)
        d['local'] = np.stack((
            d['target_coarse'].astype(np.float32) / 256,
            d['target_fine'].astype(np.float32) / 256,
        ), axis=1)
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
    model = Model(network)
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
