import argparse
import glob
import multiprocessing
import numpy as np
import tqdm
from functools import partial
from pathlib import Path
from typing import Sequence

from utility.json_utility import save_arguments
from yukarin_autoreg.linguistic_feature import LinguisticFeature
from yukarin_autoreg.phoneme import Phoneme


def process(
        path: Path,
        output_directory: Path,
        rate: int,
        types: Sequence[LinguisticFeature.Type],
):
    ps = Phoneme.load_julius_list(path)
    array = LinguisticFeature(phonemes=ps, rate=rate, types=types).make_array()

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=array, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--with_pre_post', '-wpp', action='store_true')
    parser.add_argument('--with_duration', '-wd', action='store_true')
    parser.add_argument('--with_relative_pos', '-wrp', action='store_true')
    parser.add_argument('--rate', '-r', type=int, default=100)
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    with_pre_post: bool = config.with_pre_post
    with_duration: bool = config.with_duration
    with_relative_pos: bool = config.with_relative_pos
    rate: int = config.rate

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    # Linguistic Feature Type
    types = [LinguisticFeature.Type.PHONEME]

    if with_pre_post:
        types += [LinguisticFeature.Type.PRE_PHONEME, LinguisticFeature.Type.POST_PHONEME]

    if with_duration:
        types += [LinguisticFeature.Type.PHONEME_DURATION]

        if with_pre_post:
            types += [LinguisticFeature.Type.PRE_PHONEME_DURATION, LinguisticFeature.Type.POST_PHONEME_DURATION]

    if with_relative_pos:
        types += [LinguisticFeature.Type.POS_IN_PHONEME]

    print('types:', [t.value for t in types])

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        rate=rate,
        types=types,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
