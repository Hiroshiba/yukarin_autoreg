import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import tqdm

from utility.json_utility import save_arguments
from yukarin_autoreg.sampling_data import SamplingData


def process(
        paths: List[Path],
        output_directory: Path,
        rate: int,
        mode: str,
        error_time_length: float,
):
    assert all(paths[0].stem == p.stem for p in paths[1:])

    datas = [SamplingData.load(p) for p in paths]
    scales = [(rate // d.rate) for d in datas]
    arrays: List[np.ndarray] = [d.resample(
        sampling_rate=rate,
        index=0,
        length=int(len(d.array) * s)
    ) for d, s in zip(datas, scales)]

    # assert that nearly length
    max_length = max(len(a) for a in arrays)
    for a in arrays:
        assert abs((max_length - len(a)) / rate) <= error_time_length, f'{max_length}, {len(a)}, {paths}'

    if mode == 'min':
        min_length = min(len(a) for a in arrays)
        array = np.concatenate([a[:min_length] for a in arrays], axis=1).astype(np.float32)
    elif mode == 'max':
        arrays = [
            np.pad(a, ((0, max_length - len(a)), (0, 0))) if len(a) < max_length else a
            for a in arrays
        ]
        array = np.concatenate(arrays, axis=1).astype(np.float32)
    else:
        raise ValueError(mode)

    out = output_directory / (paths[0].stem + '.npy')
    np.save(str(out), dict(array=array, rate=rate))


def process_ignore_error(*args, **kwargs):
    try:
        return process(*args, **kwargs)
    except Exception as e:
        return e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob_list', '-igl', nargs='+')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--rate', '-r', type=int)
    parser.add_argument('--mode', '-m', choices=['min', 'max'], default='min')
    parser.add_argument('--error_time_length', '-etl', type=float, default=0.015)
    parser.add_argument('--ignore_error', '-ig', action='store_true')
    config = parser.parse_args()

    input_glob_list = config.input_glob_list
    output_directory: Path = config.output_directory
    ignore_error: bool = config.ignore_error
    rate: int = config.rate
    mode: str = config.mode
    error_time_length: float = config.error_time_length

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths_list = [
        [Path(p) for p in p_list]
        for p_list in zip(*[sorted(glob.glob(input_glob)) for input_glob in input_glob_list])
    ]
    _process = partial(
        process if not ignore_error else process_ignore_error,
        output_directory=output_directory,
        rate=rate,
        mode=mode,
        error_time_length=error_time_length,
    )

    pool = multiprocessing.Pool()
    results = list(tqdm.tqdm(pool.imap_unordered(_process, paths_list), total=len(paths_list)))

    if ignore_error:
        errors = list(filter(None, results))
        print(f'num of error: {len(errors)}')

        for e in errors:
            print(e)


if __name__ == '__main__':
    main()
