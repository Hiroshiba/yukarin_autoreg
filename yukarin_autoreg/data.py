from typing import Optional

import librosa
import numpy
import numpy as np
import pysptk
import scipy.signal


def encode_16bit(wave: np.ndarray):
    encoded = ((wave + 1) * 2 ** 15).astype(np.int32)
    encoded[encoded == 2 ** 16] = 2 ** 16 - 1
    coarse = encoded // 256
    fine = encoded % 256
    return coarse, fine


def encode_single(wave: np.ndarray, bit: int = 8):
    if 8 < bit:
        wave = wave.astype(np.float64)

    coarse = ((wave + 1) * 2 ** (bit - 1)).astype(np.int32)
    coarse[coarse == 2 ** bit] = 2 ** bit - 1
    return coarse


def decode_16bit(coarse, fine):
    signal = (coarse * 256 + fine) / (2 ** 16 - 1) * 2 - 1
    return signal.astype(np.float32) if hasattr(signal, 'astype') else signal


def decode_single(coarse, bit: int = 8):
    signal = coarse / (2 ** bit - 1) * 2 - 1
    return signal.astype(np.float32) if hasattr(signal, 'astype') else signal


def encode_mulaw(x: np.ndarray, mu: int):
    mu = mu - 1
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return y


def decode_mulaw(x: np.ndarray, mu: int):
    mu = mu - 1
    y = np.sign(x) * ((1 + mu) ** np.abs(x) - 1) / mu
    return y


def to_log_melspectrogram(
        x: np.ndarray,
        sampling_rate: int,
        preemph: Optional[float],
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        min_level_db: float,
        normalize: bool,
):
    # pre emphasis
    if preemph is not None:
        x = scipy.signal.lfilter([1, -preemph], [1], x)

    # to mel spectrogram
    sp = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    mel_basis = librosa.filters.mel(sampling_rate, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    sp = np.dot(mel_basis, sp)

    # to log scale
    min_level = 10 ** (min_level_db / 20)
    sp = 20 * np.log10(np.maximum(min_level, sp))

    # normalize
    if normalize:
        sp = np.clip((sp - min_level_db) / -min_level_db, 0, 1)

    return sp.astype(np.float32).T[:-1]


def to_melcepstrum(
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
