import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import tqdm

from yukarin_autoreg.phoneme import PhonemeType, phoneme_type_to_class
from yukarin_autoreg.utility.json_utility import save_arguments
from yukarin_autoreg.wave import Wave


def process(
        paths: Tuple[Path, Path],
        output_directory: Path,
        phoneme_type: PhonemeType,
        sampling_rate: int,
):
    wave_path, phoneme_path = paths

    phoneme_class = phoneme_type_to_class[phoneme_type]
    phonemes = phoneme_class.load_julius_list(phoneme_path)

    length = len(Wave.load(wave_path, sampling_rate=sampling_rate).wave)
    array = np.ones((length,), dtype=np.bool)

    for p in filter(lambda p: p.phoneme != phoneme_class.space_phoneme, phonemes):
        s = int(round(p.start * sampling_rate))
        e = int(round(p.end * sampling_rate))
        array[s:e + 1] = False

    out = output_directory / (wave_path.stem + '.npy')
    np.save(str(out), dict(array=array, rate=sampling_rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wave_glob', '-iwg')
    parser.add_argument('--input_phoneme_glob', '-ipg')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--phoneme_type', '-pt', type=PhonemeType, default=PhonemeType.seg_kit)
    parser.add_argument('--sampling_rate', '-sr', type=int)
    config = parser.parse_args()

    input_wave_glob = config.input_wave_glob
    input_phoneme_glob = config.input_phoneme_glob
    output_directory: Path = config.output_directory
    phoneme_type: PhonemeType = config.phoneme_type
    sampling_rate: int = config.sampling_rate

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    wave_paths = sorted(Path(p) for p in glob.glob(input_wave_glob))
    phoneme_paths = sorted(Path(p) for p in glob.glob(input_phoneme_glob))
    assert len(wave_paths) == len(phoneme_paths)

    paths = list(zip(wave_paths, phoneme_paths))

    _process = partial(
        process,
        output_directory=output_directory,
        phoneme_type=phoneme_type,
        sampling_rate=sampling_rate,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
