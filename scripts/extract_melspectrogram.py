import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import tqdm

from utility.json_utility import save_arguments
from yukarin_autoreg.data import to_log_melspectrogram
from yukarin_autoreg.wave import Wave


def process(
        path: Path,
        output_directory: Path,
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
    wave = Wave.load(path, sampling_rate)

    ms = to_log_melspectrogram(
        x=wave.wave,
        sampling_rate=sampling_rate,
        preemph=preemph,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        min_level_db=min_level_db,
        normalize=normalize,
    )

    rate = sampling_rate / hop_length

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=ms, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--sampling_rate', '-sr', type=int, default=24000)
    parser.add_argument('--preemph', type=float, default=0.97)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--win_length', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--fmin', type=float, default=125)
    parser.add_argument('--fmax', type=float, default=12000)
    parser.add_argument('--min_level_db', type=float, default=-100)
    parser.add_argument('--disable_normalize', action='store_true')
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    sampling_rate: int = config.sampling_rate
    preemph: Optional[float] = config.preemph
    n_mels: int = config.n_mels
    n_fft: int = config.n_fft
    win_length: int = config.win_length
    hop_length: int = config.hop_length
    fmin: float = config.fmin
    fmax: float = config.fmax
    min_level_db: float = config.min_level_db
    normalize: bool = not config.disable_normalize

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        preemph=preemph,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        min_level_db=min_level_db,
        normalize=normalize,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths), desc='extract_melspectrogram'))


if __name__ == '__main__':
    main()
