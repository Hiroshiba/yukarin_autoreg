import chainer
import numpy as np

from yukarin_autoreg.config import DatasetConfig


def encode_16bit(wave):
    encoded = np.clip(wave * 2 ** 15, -2 ** 15, 2 ** 15 - 1).astype(np.int32) + 2 ** 15
    coarse = encoded // 256
    fine = encoded % 256
    return coarse, fine


def decode_16bit(coarse, fine):
    signal = coarse * 256 + fine
    signal -= 2 ** 15
    signal /= 2 ** 15
    return signal.astype(np.float32)


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def __len__(self):
        return 100

    def get_example(self, i):
        freq = 440
        rate = self.config.sampling_rate
        length = self.config.sampling_length
        rand = np.random.rand()
        wave = np.sin((np.arange(length + 1) * freq / rate + rand) * 2 * np.pi)

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=(coarse / 127.5 - 1).astype(np.float32),
            input_fine=(fine / 127.5 - 1).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
        )


def create(config: DatasetConfig):
    assert config.bit_size == 16

    return {
        'train': Dataset(config=config),
        'test': Dataset(config=config),
        'train_eval': Dataset(config=config),
    }
