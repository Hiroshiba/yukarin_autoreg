import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import tqdm
from kiritan_singing_label_reader import MidiNoteReader

from utility.json_utility import save_arguments
from yukarin_autoreg.midi_feature import MidiFeature


def process(
        path: Path,
        output_directory: Path,
        pitch_range: Tuple[int, int],
        with_position: bool,
        rate: int,
):
    notes = MidiNoteReader(path).get_notes()
    array = MidiFeature(notes=notes, pitch_range=pitch_range, rate=rate).make_array(with_position)

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=array, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--pitch_range', '-pr', nargs=2, type=int, default=(53, 76))
    parser.add_argument('--without_position', '-wp', action='store_true')
    parser.add_argument('--rate', '-r', type=int, default=100)
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    pitch_range: Tuple[int, int] = config.pitch_range
    with_position: bool = not config.without_position
    rate: int = config.rate

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        pitch_range=pitch_range,
        with_position=with_position,
        rate=rate,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
