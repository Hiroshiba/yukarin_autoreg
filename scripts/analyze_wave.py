import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import NamedTuple

import tqdm

from yukarin_autoreg.wave import Wave


class WaveData(NamedTuple):
    max: float
    min: float


def process(
        path: Path,
        sampling_rate: int,
):
    w = Wave.load(path, sampling_rate).wave
    return WaveData(
        max=w.max(),
        min=w.min(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--sampling_rate', '-sr', type=int)
    config = parser.parse_args()

    input_glob = config.input_glob
    sampling_rate: int = config.sampling_rate

    paths = [Path(p) for p in glob.glob(str(input_glob))]

    _process = partial(
        process,
        sampling_rate=sampling_rate,
    )

    pool = multiprocessing.Pool()
    all_data = list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))

    print('max', max([d.max for d in all_data]))
    print('min', min([d.min for d in all_data]))


if __name__ == '__main__':
    main()
