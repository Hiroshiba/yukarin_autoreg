import glob
from pathlib import Path
from typing import List

import chainer
import numpy as np

from yukarin_autoreg.config import DatasetConfig
from yukarin_autoreg.wave import Wave


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
    def __init__(self, paths: List[Path], config: DatasetConfig) -> None:
        self.config = config

        waves = [
            Wave.load(p, self.config.sampling_rate).wave
            for p in paths
        ]
        self.wave = np.concatenate(waves)

    def __len__(self):
        return len(self.wave) // self.config.sampling_length - 1

    def get_example(self, i):
        length = self.config.sampling_length
        offset = i * length + np.random.randint(length)
        wave = self.wave[offset:offset + length]

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=(coarse / 127.5 - 1).astype(np.float32),
            input_fine=(fine / 127.5 - 1).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
        )


def create(config: DatasetConfig):
    assert config.bit_size == 16

    input_paths = [Path(p) for p in glob.glob(str(config.input_glob))]

    num_test = config.num_test
    np.random.RandomState(config.seed).shuffle(input_paths)
    train_paths = input_paths[num_test:]
    test_paths = input_paths[:num_test]
    train_for_evaluate_paths = train_paths[:num_test]

    return {
        'train': Dataset(train_paths, config=config),
        'test': Dataset(test_paths, config=config),
        'train_eval': Dataset(train_for_evaluate_paths, config=config),
    }
