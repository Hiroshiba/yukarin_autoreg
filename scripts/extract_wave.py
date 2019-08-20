import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tqdm

from yukarin_autoreg.utility.json_utility import save_arguments
from yukarin_autoreg.wave import Wave


def process(
        path: Path,
        output_directory: Path,
        sampling_rate: int,
        clipping_range: Optional[Tuple[float, float]],
        clipping_auto: bool,
):
    w = Wave.load(path, sampling_rate).wave

    if clipping_range is not None:
        w = np.clip(w, clipping_range[0], clipping_range[1]) / np.max(np.abs(clipping_range))

    if clipping_auto:
        w /= np.abs(w).max() * 0.999

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=w, rate=sampling_rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--sampling_rate', '-sr', type=int)
    parser.add_argument('--clipping_range', '-cr', type=float, nargs=2, help='(min, max)')
    parser.add_argument('--clipping_auto', '-ca', action='store_true')
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    sampling_rate: int = config.sampling_rate
    clipping_range: Optional[Tuple[float, float]] = config.clipping_range
    clipping_auto: bool = config.clipping_auto

    assert clipping_range is None or not clipping_auto

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        clipping_range=clipping_range,
        clipping_auto=clipping_auto,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
