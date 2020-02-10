import argparse
import glob
import multiprocessing
from pathlib import Path

from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from tqdm import tqdm


def process(
        wave_path: Path,
        local_path: Path,
):
    wave_data = Wave.load(wave_path)
    local_data = SamplingData.load(local_path)

    sr = wave_data.sampling_rate

    assert sr % local_data.rate == 0
    l_scale = int(sr // local_data.rate)

    length = len(local_data.array) * l_scale
    if abs(length - len(wave_data.wave)) > l_scale:
        return Exception(f'{wave_path.stem}: {len(wave_data.wave) - length}')


def process_wrapper(args):
    return process(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_glob')
    parser.add_argument('--local_glob')
    config = parser.parse_args()

    wave_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.wave_glob))}
    fn_list = sorted(wave_paths.keys())

    local_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.local_glob))}
    assert set(fn_list) == set(local_paths.keys())

    pool = multiprocessing.Pool()
    it = pool.imap_unordered(process_wrapper, [(wave_paths[fn], local_paths[fn]) for fn in fn_list])
    errors = list(tqdm(it))

    for error in filter(None, errors):
        print(error)
