import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import tqdm

from utility.json_utility import save_arguments
from yukarin_autoreg.wave import Wave


def process(
        path: Path,
        output_directory: Path,
        sampling_rate: int,
        frame_length: int,
        hop_length: int,
        top_db: int,
        normalize: bool,
):
    assert sampling_rate % hop_length == 0

    w = Wave.load(path, sampling_rate).wave

    mse = librosa.feature.rmse(w, frame_length=frame_length, hop_length=hop_length) ** 2
    array = librosa.power_to_db(mse.squeeze(), top_db=top_db)[:-1]

    if normalize:
        array = np.clip((array - array.max()) / top_db + 1, 0, 1)

    rate = sampling_rate // hop_length

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=array, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--sampling_rate', '-sr', type=int)
    parser.add_argument('--frame_length', '-fl', type=int, default=800)
    parser.add_argument('--hop_length', '-hl', type=int, default=200)
    parser.add_argument('--top_db', '-td', type=int, default=80)
    parser.add_argument('--normalize', '-n', action='store_true')
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    sampling_rate: int = config.sampling_rate
    frame_length: int = config.frame_length
    hop_length: int = config.hop_length
    top_db: int = config.top_db
    normalize: bool = config.normalize

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        sampling_rate=sampling_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        top_db=top_db,
        normalize=normalize,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
