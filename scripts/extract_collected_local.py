import argparse
import glob
import multiprocessing
import numpy as np
import tqdm
from functools import partial
from pathlib import Path
from typing import List

from utility.json_utility import save_arguments
from yukarin_autoreg.sampling_data import SamplingData


def process(
        paths: List[Path],
        output_directory: Path,
        rate: int,
):
    assert all(paths[0].stem == p.stem for p in paths[1:])

    datas = [SamplingData.load(p) for p in paths]
    scales = [(rate // d.rate) for d in datas]
    arrays = [d.resample(
        sampling_rate=rate,
        index=0,
        length=int(len(d.array) * s)
    ) for d, s in zip(datas, scales)]

    # assert that nearly length
    max_length = max(len(a) for a in arrays)
    for s, a in zip(scales, arrays):
        assert 0.97 < max_length / len(a) < 1.03, f'{max_length}, {len(a)}'

    min_length = min(len(a) for a in arrays)
    array = np.concatenate([a[:min_length] for a in arrays], axis=1).astype(np.float32)

    out = output_directory / (paths[0].stem + '.npy')
    np.save(str(out), dict(array=array, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob_list', '-igl', nargs='+')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--rate', '-r', type=int)
    config = parser.parse_args()

    input_glob_list = config.input_glob_list
    output_directory: Path = config.output_directory
    rate: int = config.rate

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths_list = [
        [Path(p) for p in p_list]
        for p_list in zip(*[sorted(glob.glob(input_glob)) for input_glob in input_glob_list])
    ]
    _process = partial(
        process,
        output_directory=output_directory,
        rate=rate,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths_list), total=len(paths_list)))


if __name__ == '__main__':
    main()
