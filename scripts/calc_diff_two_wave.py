import argparse
from pathlib import Path

import librosa
import numpy
import pysptk

from yukarin_autoreg.wave import Wave

_logdb_const = 10.0 / numpy.log(10.0) * numpy.sqrt(2.0)


def mcd(x: numpy.ndarray, y: numpy.ndarray) -> float:
    z = x - y
    r = numpy.sqrt((z * z).sum(axis=1)).mean()
    return r


def melcepstrum(
        x: numpy.ndarray,
        sampling_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        order: int,
):
    sp = numpy.abs(librosa.stft(y=x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)) ** 2
    sp = sp.T

    sp[sp < 1e-5] = 1e-5
    mc = pysptk.sp2mc(sp, order=order, alpha=pysptk.util.mcepalpha(sampling_rate))
    return mc


def calc_diff_two_wave(
        path1: Path,
        path2: Path,
):
    wave1 = Wave.load(path1)
    wave2 = Wave.load(path2)
    assert wave1.sampling_rate == wave2.sampling_rate

    sampling_rate = wave1.sampling_rate

    min_length = min(len(wave1.wave), len(wave2.wave))
    wave1.wave = wave1.wave[:min_length]
    wave2.wave = wave2.wave[:min_length]

    mc1 = melcepstrum(x=wave1.wave, sampling_rate=sampling_rate, n_fft=2048, win_length=1024, hop_length=256, order=24)
    mc2 = melcepstrum(x=wave2.wave, sampling_rate=sampling_rate, n_fft=2048, win_length=1024, hop_length=256, order=24)
    return mcd(mc1, mc2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path1', type=Path)
    parser.add_argument('path2', type=Path)
    arguments = parser.parse_args()

    diff = calc_diff_two_wave(
        path1=arguments.path1,
        path2=arguments.path2,
    )
    print(diff)
