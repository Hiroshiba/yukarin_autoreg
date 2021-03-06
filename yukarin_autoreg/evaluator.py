from pathlib import Path
from typing import Optional

import chainer
import numpy as np
from acoustic_feature_extractor.data.spectrogram import to_melcepstrum
from acoustic_feature_extractor.data.wave import Wave
from chainer import Chain

from yukarin_autoreg.generator import Generator, SamplingPolicy

_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)


def _mcd(x: np.ndarray, y: np.ndarray) -> float:
    z = x - y
    r = np.sqrt((z * z).sum(axis=1)).mean()
    return _logdb_const * r


def calc_mcd(
    path1: Optional[Path] = None,
    path2: Optional[Path] = None,
    wave1: Optional[Wave] = None,
    wave2: Optional[Wave] = None,
):
    wave1 = Wave.load(path1) if wave1 is None else wave1
    wave2 = Wave.load(path2) if wave2 is None else wave2
    assert wave1.sampling_rate == wave2.sampling_rate

    sampling_rate = wave1.sampling_rate

    min_length = min(len(wave1.wave), len(wave2.wave))
    wave1.wave = wave1.wave[:min_length]
    wave2.wave = wave2.wave[:min_length]

    mc1 = to_melcepstrum(
        x=wave1.wave,
        sampling_rate=sampling_rate,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
        order=24,
    )
    mc2 = to_melcepstrum(
        x=wave2.wave,
        sampling_rate=sampling_rate,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
        order=24,
    )
    return _mcd(mc1, mc2)


class GenerateEvaluator(Chain):
    def __init__(
        self,
        generator: Generator,
        time_length: float,
        local_padding_time_length: float,
        sampling_policy: SamplingPolicy = SamplingPolicy.random,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.time_length = time_length
        self.local_padding_time_length = local_padding_time_length
        self.sampling_policy = sampling_policy

    def __call__(
        self,
        wave: np.ndarray,
        local: Optional[np.ndarray],
        speaker_num: Optional[np.ndarray] = None,
    ):
        batchsize = len(wave)
        wave = chainer.cuda.to_cpu(wave)

        wave_output = self.generator.generate(
            time_length=self.time_length + self.local_padding_time_length * 2,
            sampling_policy=self.sampling_policy,
            num_generate=batchsize,
            local_array=local,
            speaker_nums=speaker_num,
        )

        mcd_list = []
        for wi, wo in zip(wave, wave_output):
            wi = Wave(wave=wi, sampling_rate=wo.sampling_rate)

            if self.local_padding_time_length > 0:
                pad = int(wo.sampling_rate * self.local_padding_time_length)
                wo.wave = wo.wave[pad:-pad]

            mcd = calc_mcd(wave1=wi, wave2=wo)
            mcd_list.append(mcd)

        scores = {"mcd": (self.generator.xp.asarray(mcd_list).mean(), batchsize)}

        chainer.report(scores, self)
        return scores
