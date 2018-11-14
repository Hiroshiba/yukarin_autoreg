import argparse
import glob
import multiprocessing
import numpy as np
import pyworld
import tqdm
from functools import partial
from pathlib import Path

from utility.json_utility import save_arguments
from yukarin_autoreg.wave import Wave


def process(
        path: Path,
        output_directory: Path,
        sampling_rate: int,
        frame_period: float,
        f0_floor: float,
        f0_ceil: float,
):
    w = Wave.load(path, sampling_rate).wave.astype(np.float64)

    f0, t = pyworld.harvest(
        w,
        sampling_rate,
        frame_period=frame_period,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
    )
    f0 = pyworld.stonemask(w, f0, t, sampling_rate)

    f0_log = np.zeros_like(f0)
    f0_log[f0 != 0] = np.log(f0[f0 != 0])

    rate = int(1000 // frame_period)

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=f0_log, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--sampling_rate', '-sr', type=int)
    parser.add_argument('--frame_period', '-fp', type=float, default=5.0)
    parser.add_argument('--f0_floor', '-ff', type=int, default=71.0)
    parser.add_argument('--f0_ceil', '-fc', type=int, default=800.0)
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    sampling_rate: int = config.sampling_rate
    frame_period: float = config.frame_period
    f0_floor: float = config.f0_floor
    f0_ceil: float = config.f0_ceil

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        frame_period=frame_period,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
