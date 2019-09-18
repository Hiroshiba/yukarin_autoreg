import glob
from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Union

import chainer
import numpy as np

from yukarin_autoreg.config import DatasetConfig
from yukarin_autoreg.data import encode_16bit, encode_single, decode_single, encode_mulaw
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


class BaseWaveDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            sampling_length: int,
            to_double: bool,
            bit: int,
            mulaw: bool,
            local_padding_size: int,
    ) -> None:
        self.sampling_length = sampling_length
        self.to_double = to_double
        self.bit = bit
        self.mulaw = mulaw
        self.local_padding_size = local_padding_size

    @staticmethod
    def extract_input(
            sampling_length: int,
            wave_data: Wave,
            silence_data: SamplingData,
            local_data: SamplingData,
            local_padding_size: int,
            padding_value=0,
    ):
        """
        :return:
            wave: (sampling_length, )
            silence: (sampling_length, )
            local: (sampling_length // scale + pad, )
        """
        sr = wave_data.sampling_rate
        sl = sampling_length

        assert sr % local_data.rate == 0
        l_scale = int(sr // local_data.rate)

        length = len(local_data.array) * l_scale
        assert abs(length - len(wave_data.wave)) < l_scale * 4, f'{abs(length - len(wave_data.wave))} {l_scale}'

        l_length = length // l_scale
        l_sl = sl // l_scale
        l_offset = np.random.randint(l_length - l_sl)
        offset = l_offset * l_scale

        assert local_padding_size % l_scale == 0
        l_pad = local_padding_size // l_scale

        wave = wave_data.wave[offset:offset + sl]
        silence = np.squeeze(silence_data.resample(sr, index=offset, length=sl))

        # local
        l_start, l_end = l_offset - l_pad, l_offset + l_sl + l_pad
        if l_start < 0 or l_end > l_length:
            shape = list(local_data.array.shape)
            shape[0] = l_sl + l_pad * 2
            local = np.ones(shape=shape, dtype=local_data.array.dtype) * padding_value
            if l_start < 0:
                p_start = -l_start
                l_start = 0
            else:
                p_start = 0
            if l_end > l_length:
                p_end = l_sl + l_pad * 2 - (l_end - l_length)
                l_end = l_length
            else:
                p_end = l_sl + l_pad * 2
            local[p_start:p_end] = local_data.array[l_start:l_end]
        else:
            local = local_data.array[l_start:l_end]

        return wave, silence, local

    @staticmethod
    def add_noise(wave: np.ndarray, gaussian_noise_sigma: float):
        if gaussian_noise_sigma > 0:
            wave += np.random.normal(0, gaussian_noise_sigma)
            wave[wave > 1.0] = 1.0
            wave[wave < -1.0] = -1.0
        return wave

    def convert_to_dict(self, wave: np.ndarray, silence: np.ndarray, local: np.ndarray):
        if self.mulaw:
            wave = encode_mulaw(wave, mu=2 ** self.bit)
        if self.to_double:
            assert self.bit == 16
            encoded_coarse, encoded_fine = encode_16bit(wave)
            coarse = decode_single(encoded_coarse).astype(np.float32)
            fine = decode_single(encoded_fine).astype(np.float32)[:-1]
        else:
            encoded_coarse = encode_single(wave, bit=self.bit)
            encoded_fine = None
            coarse = wave.ravel().astype(np.float32)
            fine = None
        return dict(
            coarse=coarse,
            fine=fine,
            encoded_coarse=encoded_coarse,
            encoded_fine=encoded_fine,
            local=local,
            silence=silence[1:],
        )

    def make_input(
            self,
            wave_data: Wave,
            silence_data: SamplingData,
            local_data: SamplingData,
            gaussian_noise_sigma: float,
    ):
        for _ in range(300):
            wave, silence, local = self.extract_input(
                self.sampling_length,
                wave_data,
                silence_data,
                local_data,
                self.local_padding_size,
            )
            if not silence.all():
                break
            print('retry to pick not silence')
        else:
            # not raise but return. (because this method will be called in multiprocess.Process)
            return Exception('cannot pick not silence data')
        wave = self.add_noise(wave=wave, gaussian_noise_sigma=gaussian_noise_sigma)
        d = self.convert_to_dict(wave, silence, local)
        return d


class WavesDataset(BaseWaveDataset):
    def __init__(
            self,
            inputs: List[Union[Input, LazyInput]],
            sampling_length: int,
            to_double: bool,
            bit: int,
            mulaw: bool,
            local_padding_size: int,
            gaussian_noise_sigma: float,
    ) -> None:
        super().__init__(
            sampling_length=sampling_length,
            to_double=to_double,
            bit=bit,
            mulaw=mulaw,
            local_padding_size=local_padding_size,
        )
        self.inputs = inputs
        self.gaussian_noise_sigma = gaussian_noise_sigma

    def __len__(self):
        return len(self.inputs)

    def get_example(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.make_input(
            wave_data=input.wave,
            silence_data=input.silence,
            local_data=input.local,
            gaussian_noise_sigma=self.gaussian_noise_sigma,
        )


def create(config: DatasetConfig):
    if not config.only_coarse:
        assert config.bit_size == 16

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
    num_train = config.num_train if config.num_train is not None else len(inputs) - num_test

    trains = inputs[num_test:][:num_train]
    tests = inputs[:num_test]
    evals = trains[:num_test]

    _Dataset = partial(
        WavesDataset,
        sampling_length=config.sampling_length,
        to_double=not config.only_coarse,
        bit=config.bit_size,
        mulaw=config.mulaw,
        local_padding_size=config.local_padding_size,
        gaussian_noise_sigma=config.gaussian_noise_sigma,
    )
    return {
        'train': _Dataset(trains),
        'test': _Dataset(tests),
        'train_eval': _Dataset(evals),
    }
