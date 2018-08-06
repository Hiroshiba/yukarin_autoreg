import glob
from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Optional

import chainer
import numpy as np

from yukarin_autoreg.config import DatasetConfig
from yukarin_autoreg.wave import Wave


class Input(NamedTuple):
    wave: np.ndarray
    local: np.ndarray


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


def load_local_and_interp(path: Path, sampling_rate: int):
    input_dict = np.load(path)
    local, local_rate = input_dict['array'], input_dict['rate']
    assert sampling_rate % local_rate == 0, f'{sampling_rate} {local_rate}'

    local = np.repeat(local, sampling_rate // local_rate)[:, np.newaxis]
    return local


def load_input(
        path_wave: Path,
        path_silence: Optional[Path],
        path_local: Optional[Path],
):
    w = Wave.load(path_wave)
    wave = w.wave
    sr = w.sampling_rate
    length = len(wave)

    if path_local is not None:
        local = load_local_and_interp(path_local, sampling_rate=sr)

        # trim wave
        assert abs(len(local) - length) < sr
        length = len(local)
        wave = wave[:length]
    else:
        local = np.empty(shape=(length, 0), dtype=wave.dtype)

    if path_silence is not None:
        silence_dict = np.load(path_silence)
        silence, rate = silence_dict['array'], silence_dict['rate']
        assert sr == rate

        silence = silence[:length]

        wave = wave[~silence]
        local = local[~silence]

    return Input(
        wave=wave,
        local=local,
    )


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
        wave = np.sin((np.arange(length + 1) * freq / rate + rand) * 2 * np.pi)

        local = np.empty(shape=(length, 0), dtype=np.float32)

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
            local=local,
        )


class WavesDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            inputs: List[Input],
            sampling_length: int,
    ) -> None:
        self.inputs = inputs
        self.sampling_length = sampling_length

    def __len__(self):
        return len(self.inputs)

    def get_example(self, i):
        wave = self.inputs[i].wave
        local = self.inputs[i].local
        assert len(wave) == len(local)
        assert local.ndim == 2

        offset = np.random.randint(len(wave) - self.sampling_length)
        wave = wave[offset:offset + self.sampling_length]
        local = local[offset:offset + self.sampling_length]

        coarse, fine = encode_16bit(wave)
        return dict(
            input_coarse=normalize(coarse).astype(np.float32),
            input_fine=normalize(fine).astype(np.float32)[:-1],
            target_coarse=coarse[1:],
            target_fine=fine[1:],
            local=local[1:],
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

    if config.input_silence_glob is not None:
        silence_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_silence_glob))}
        assert set(wave_paths.keys()) == set(silence_paths.keys())
    else:
        silence_paths = None

    if config.input_local_glob is not None:
        local_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_local_glob))}
        assert set(wave_paths.keys()) == set(local_paths.keys())
    else:
        local_paths = None

    fn_list = sorted(wave_paths.keys())

    inputs = [
        load_input(
            path_wave=wave_paths[fn],
            path_silence=silence_paths[fn] if silence_paths is not None else None,
            path_local=local_paths[fn] if local_paths is not None else None,
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
    )
    return {
        'train': _Dataset(trains),
        'test': _Dataset(tests),
        'train_eval': _Dataset(evals),
    }
