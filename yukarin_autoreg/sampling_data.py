from pathlib import Path
from typing import Dict, NamedTuple

import numpy as np


class SamplingData(NamedTuple):
    array: np.ndarray  # shape: (N, ?)
    rate: float

    def resample(self, sampling_rate: float, index: int, length: int):
        assert sampling_rate % self.rate == 0, f'{sampling_rate} {self.rate}'
        scale = int(sampling_rate // self.rate)

        ni = index // scale
        nl = length // scale + 2

        array = self.array[ni:ni + nl]
        if scale > 1:
            array = np.repeat(array, scale, axis=0)

        i = index - ni * scale
        return array[i:i + length]

    @staticmethod
    def load(path: Path):
        d: Dict = np.load(str(path), allow_pickle=True).item()
        array, rate = d['array'], d['rate']

        if array.ndim == 1:
            array = array[:, np.newaxis]

        return SamplingData(
            array=array,
            rate=rate,
        )
