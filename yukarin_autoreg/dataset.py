import glob
from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Union

import chainer
import numpy as np

from yukarin_autoreg.config import DatasetConfig
from yukarin_autoreg.sampling_data import SamplingData
from yukarin_autoreg.wave import Wave


class Input(NamedTuple):
    wave: Wave
    silence: SamplingData
    local: SamplingData


class LazyInput(NamedTuple):
    path_wave: Path
    path_silence: Path
    path_local: Path

    def generate(self):
        return Input(
            wave=Wave.load(self.path_wave),
            silence=SamplingData.load(self.path_silence),
            local=SamplingData.load(self.path_local),
        )


def encode_16bit(wave):
    encoded = ((wave + 1) * 2 ** 15).astype(np.int32)
    encoded[encoded == 2 ** 16] = 2 ** 16 - 1
    coarse = encoded // 256
    fine = encoded % 256
    return coarse, fine


def decode_16bit(coarse, fine):
    signal = (coarse * 256 + fine) / 2 ** 15 - 1
    return signal.astype(np.float32)


def normalize(b):
    return b / 127.5 - 1


class SignWaveDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sampling_rate: int, sampling_length: int) -> None:
        self.sampling_rate = sampling_rate
        self.sampling_length = sampling_length

    def __len__(self):
        return 100

    def get_example(self, i):
        freq = 440
        rate = self.sampling_rate
        length = self.sampling_length
        rand = np.random.rand()
        wave = np.sin((np.arange(length) * freq / rate + rand) * 2 * np.pi)

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


class WavesDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            inputs: List[Union[Input, LazyInput]],
            sampling_rate: int,
            sampling_length: int,
    ) -> None:
        self.inputs = inputs
        self.sampling_rate = sampling_rate
        self.sampling_length = sampling_length

    def __len__(self):
        return len(self.inputs)

    def get_example(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        sr = self.sampling_rate
        sl = self.sampling_length

        wave = input.wave.wave
        local_data = input.local

        length = len(local_data.array) * (sr // local_data.rate)
        assert abs(length - len(wave)) < sr

        offset = np.random.randint(length - sl)

        wave = wave[offset:offset + sl]
        local = local_data.resample(sr, index=offset, length=sl)
        silence = np.squeeze(input.silence.resample(sr, index=offset, length=sl))

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
            local=local[1:],
            silence=silence[1:],
        )


def create(config: DatasetConfig):
    assert config.bit_size == 16

    if config.sign_wave_dataset:
        return {
            'train': SignWaveDataset(sampling_rate=config.sampling_rate, sampling_length=config.sampling_length),
            'test': SignWaveDataset(sampling_rate=config.sampling_rate, sampling_length=config.sampling_length),
            'train_eval': SignWaveDataset(sampling_rate=config.sampling_rate, sampling_length=config.sampling_length),
        }

    wave_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_wave_glob))}

    silence_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_silence_glob))}
    assert set(wave_paths.keys()) == set(silence_paths.keys())

    local_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_local_glob))}
    assert set(wave_paths.keys()) == set(local_paths.keys())

    fn_list = sorted(wave_paths.keys())

    inputs = [
        LazyInput(
            path_wave=wave_paths[fn],
            path_silence=silence_paths[fn],
            path_local=local_paths[fn],
        )
        for fn in fn_list
    ]
    np.random.RandomState(config.seed).shuffle(inputs)

    num_test = config.num_test
    trains = inputs[num_test:]
    tests = inputs[:num_test]
    evals = trains[:num_test]

    _Dataset = partial(
        WavesDataset,
        sampling_rate=config.sampling_rate,
        sampling_length=config.sampling_length,
    )
    return {
        'train': _Dataset(trains),
        'test': _Dataset(tests),
        'train_eval': _Dataset(evals),
    }
