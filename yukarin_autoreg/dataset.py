import glob
from pathlib import Path
from typing import List, Optional

import chainer
import librosa
import numpy as np

from yukarin_autoreg.config import DatasetConfig
from yukarin_autoreg.wave import Wave


def encode_16bit(wave):
    encoded = np.clip(wave * 2 ** 15, -2 ** 15, 2 ** 15 - 1).astype(np.int32) + 2 ** 15
    coarse = encoded // 256
    fine = encoded % 256
    return coarse, fine


def decode_16bit(coarse, fine):
    signal = (coarse * 256 + fine).astype(np.float32)
    signal /= 2 ** 16 - 1
    signal -= 0.5
    signal *= 2
    return signal


def normalize(b):
    return b / 127.5 - 1


def denormalize(b):
    return (b + 1) * 127.5


class SignWaveDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sampling_rate: int, sampling_length: int):
        self.sampling_rate = sampling_rate
        self.sampling_length = sampling_length

    def __len__(self):
        return 100

    def get_example(self, i):
        freq = 440
        rate = self.sampling_rate
        length = self.sampling_length
        rand = np.random.rand()
        wave = np.sin((np.arange(length + 1) * freq / rate + rand) * 2 * np.pi)

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
        )


class WavesDataset(chainer.dataset.DatasetMixin):
    def __init__(self, waves: List[np.array], sampling_length: int, top_db: Optional[float]):
        self.sampling_length = sampling_length
        wave = np.concatenate(waves)
        if top_db is not None:
            wave = librosa.effects.remix(wave, librosa.effects.split(wave, top_db=top_db))
        self.wave = wave

    def __len__(self):
        return len(self.wave) // self.sampling_length - 1

    def get_example(self, i):
        length = self.sampling_length
        offset = i * length + np.random.randint(length)
        wave = self.wave[offset:offset + length]

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
        )

    @staticmethod
    def load_waves(paths: List[Path], sampling_rate: int):
        waves = [
            Wave.load(p, sampling_rate).wave
            for p in paths
        ]
        return waves


def create(config: DatasetConfig):
    assert config.bit_size == 16

    if config.sign_wave_dataset:
        return {
            'train': SignWaveDataset(sampling_rate=config.sampling_rate, sampling_length=config.sampling_length),
            'test': SignWaveDataset(sampling_rate=config.sampling_rate, sampling_length=config.sampling_length),
            'train_eval': SignWaveDataset(sampling_rate=config.sampling_rate, sampling_length=config.sampling_length),
        }

    paths = sorted([Path(p) for p in glob.glob(str(config.input_glob))])
    waves = WavesDataset.load_waves(paths, sampling_rate=config.sampling_rate)
    np.random.RandomState(config.seed).shuffle(waves)

    num_test = config.num_test
    trains = waves[num_test:]
    tests = waves[:num_test]
    evals = trains[:num_test]

    return {
        'train': WavesDataset(trains, sampling_length=config.sampling_length, top_db=config.silence_top_db),
        'test': WavesDataset(tests, sampling_length=config.sampling_length, top_db=config.silence_top_db),
        'train_eval': WavesDataset(evals, sampling_length=config.sampling_length, top_db=config.silence_top_db),
    }
