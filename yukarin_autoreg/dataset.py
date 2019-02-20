import chainer
import glob
import numpy as np
from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Union

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


def encode_8bit(wave):
    coarse = ((wave + 1) * 2 ** 7).astype(np.int32)
    coarse[coarse == 2 ** 8] = 2 ** 8 - 1
    return coarse


def decode_16bit(coarse, fine):
    signal = (coarse * 256 + fine) / (2 ** 16 - 1) * 2 - 1
    return signal.astype(np.float32)


def decode_8bit(coarse):
    signal = coarse / (2 ** 8 - 1) * 2 - 1
    return signal.astype(np.float32)


def normalize(b):
    return b / 127.5 - 1


class BaseWaveDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            sampling_length: int,
    ) -> None:
        self.sampling_length = sampling_length

    @staticmethod
    def extract_input(sampling_length: int, wave_data: Wave, silence_data: SamplingData, local_data: SamplingData):
        """
        :return:
            wave: (sampling_length, )
            silence: (sampling_length, )
            local: (sampling_length // scale, )
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

        wave = wave_data.wave[offset:offset + sl]
        silence = np.squeeze(silence_data.resample(sr, index=offset, length=sl))
        local = local_data.array[l_offset:l_offset + l_sl]
        return wave, silence, local

    @staticmethod
    def add_noise(wave: np.ndarray, gaussian_noise_sigma: float):
        if gaussian_noise_sigma > 0:
            wave += np.random.normal(0, gaussian_noise_sigma)
            wave[wave > 1.0] = 1.0
            wave[wave < -1.0] = -1.0
        return wave

    @staticmethod
    def convert_to_dict(wave: np.ndarray, silence: np.ndarray, local: np.ndarray):
        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
            silence=silence[1:],
            local=local,
        )

    def make_input(
            self,
            wave_data: Wave,
            silence_data: SamplingData,
            local_data: SamplingData,
            gaussian_noise_sigma: float,
    ):
        wave, silence, local = self.extract_input(self.sampling_length, wave_data, silence_data, local_data)
        wave = self.add_noise(wave=wave, gaussian_noise_sigma=gaussian_noise_sigma)
        d = self.convert_to_dict(wave, silence, local)
        return d


class SignWaveDataset(BaseWaveDataset):
    def __init__(self, sampling_rate: int, sampling_length: int) -> None:
        super().__init__(sampling_length=sampling_length)
        self.sampling_rate = sampling_rate

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
        return self.convert_to_dict(wave, silence, local)


class WavesDataset(BaseWaveDataset):
    def __init__(
            self,
            inputs: List[Union[Input, LazyInput]],
            sampling_length: int,
            gaussian_noise_sigma: float,
    ) -> None:
        super().__init__(sampling_length=sampling_length)
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
    assert (config.bit_size == 16 and not config.only_coarse) or (config.bit_size == 8 and config.only_coarse)

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
        sampling_length=config.sampling_length,
        gaussian_noise_sigma=config.gaussian_noise_sigma,
    )
    return {
        'train': _Dataset(trains),
        'test': _Dataset(tests),
        'train_eval': _Dataset(evals),
    }
