import multiprocessing

import argparse
import glob
import librosa
import numpy as np
import tqdm
from functools import partial
from pathlib import Path
from typing import Optional

from utility.json_utility import save_arguments
from yukarin_autoreg.wave import Wave


def process(
        path: Path,
        output_directory: Path,
        sampling_rate: int,
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        fmin: float,
        fmax: float,
        threshold: Optional[float],
        normalize: Optional[bool],
):
    wave = Wave.load(path, sampling_rate)

    sp = librosa.stft(
        wave.wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    mel = librosa.filters.mel(
        sampling_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    ms = np.dot(mel, np.abs(sp)).T[:-1]

    if threshold is not None:
        ms[ms < threshold] = threshold

    ms = librosa.amplitude_to_db(ms.astype(np.float32), top_db=80) - 20

    if normalize:
        ms = np.clip((ms / 100) + 1, 0, 1)

    rate = sampling_rate / hop_length

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=ms, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--sampling_rate', '-sr', type=int, default=24000)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--win_length', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--fmin', type=float, default=125)
    parser.add_argument('--fmax', type=float, default=7600)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--normalize', action='store_true')
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    sampling_rate: int = config.sampling_rate
    n_mels: int = config.n_mels
    n_fft: int = config.n_fft
    win_length: int = config.win_length
    hop_length: int = config.hop_length
    fmin: float = config.fmin
    fmax: float = config.fmax
    threshold: float = config.threshold
    normalize: bool = config.normalize

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        threshold=threshold,
        normalize=normalize,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
