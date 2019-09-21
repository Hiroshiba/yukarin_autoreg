from typing import Callable, Dict, Optional

import chainer
import numpy as np
from chainer.iterators import MultiprocessIterator
from chainer.training.updaters import StandardUpdater

from yukarin_autoreg.dataset import BaseWaveDataset
from yukarin_autoreg.model import Model
from yukarin_autoreg.utility.chainer_converter_utility import concat_optional


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
        if self.to_double:
            d['local'] = np.stack((
                d['encoded_coarse'].astype(np.float32) / 256,
                d['encoded_fine'].astype(np.float32) / 256,
            ), axis=1)
        else:
            d['local'] = np.stack((
                d['encoded_coarse'].astype(np.float32) / 256,
                d['encoded_coarse'].astype(np.float32) / 256,
            ), axis=1)
        return d


class DownLocalRandomDataset(LocalRandomDataset):
    def __init__(self, scale: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def get_example(self, i):
        d = super().get_example(i)
        l = np.reshape(d['local'], (-1, self.scale * d['local'].shape[1]))
        l[np.isnan(l)] = 0
        d['local'] = l
        return d


class SignWaveDataset(BaseWaveDataset):
    def __init__(
            self,
            sampling_rate: int,
            sampling_length: int,
            to_double: bool,
            bit: int,
            mulaw: bool,
            frequency: float = 440
    ) -> None:
        super().__init__(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            local_padding_size=0,
        )
        self.sampling_rate = sampling_rate
        self.frequency = frequency

    def __len__(self):
        return 100

    def get_example(self, i):
        rate = self.sampling_rate
        length = self.sampling_length
        rand = np.random.rand()

        wave = np.sin((np.arange(length) * self.frequency / rate + rand) * 2 * np.pi)
        local = np.empty(shape=(length, 0), dtype=np.float32)
        silence = np.zeros(shape=(length,), dtype=np.bool)
        return self.convert_to_dict(wave, silence, local)


def _create_optimizer(model):
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    return optimizer


def setup_support(
        batch_size: int,
        gpu: Optional[int],
        model: Model,
        dataset: chainer.dataset.DatasetMixin,
):
    optimizer = _create_optimizer(model)
    train_iter = MultiprocessIterator(dataset, batch_size)

    if gpu is not None:
        model.to_gpu(gpu)

    updater = StandardUpdater(
        device=gpu,
        iterator=train_iter,
        optimizer=optimizer,
        converter=concat_optional,
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

    print(observation)
    if last_hook is not None:
        last_hook(observation)
