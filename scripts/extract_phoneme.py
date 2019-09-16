import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import tqdm

from utility.json_utility import save_arguments
from yukarin_autoreg.linguistic_feature import LinguisticFeature
from yukarin_autoreg.phoneme import PhonemeType, phoneme_type_to_class


def process(
        path: Path,
        output_directory: Path,
        phoneme_type: PhonemeType,
        rate: int,
        types: Sequence[LinguisticFeature.FeatureType],
):
    phoneme_class = phoneme_type_to_class[phoneme_type]
    ps = phoneme_class.load_julius_list(path)
    array = LinguisticFeature(phonemes=ps, phoneme_class=phoneme_class, rate=rate, feature_types=types).make_array()

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=array, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--phoneme_type', '-pt', type=PhonemeType, default=PhonemeType.seg_kit)
    parser.add_argument('--with_pre_post', '-wpp', action='store_true')
    parser.add_argument('--with_duration', '-wd', action='store_true')
    parser.add_argument('--with_relative_pos', '-wrp', action='store_true')
    parser.add_argument('--rate', '-r', type=int, default=100)
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    phoneme_type: PhonemeType = config.phoneme_type
    with_pre_post: bool = config.with_pre_post
    with_duration: bool = config.with_duration
    with_relative_pos: bool = config.with_relative_pos
    rate: int = config.rate

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    # Linguistic Feature Type
    types = [LinguisticFeature.FeatureType.PHONEME]

    if with_pre_post:
        types += [LinguisticFeature.FeatureType.PRE_PHONEME, LinguisticFeature.FeatureType.POST_PHONEME]

    if with_duration:
        types += [LinguisticFeature.FeatureType.PHONEME_DURATION]

        if with_pre_post:
            types += [
                LinguisticFeature.FeatureType.PRE_PHONEME_DURATION,
                LinguisticFeature.FeatureType.POST_PHONEME_DURATION,
            ]

    if with_relative_pos:
        types += [LinguisticFeature.FeatureType.POS_IN_PHONEME]

    print('types:', [t.value for t in types])

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        phoneme_type=phoneme_type,
        rate=rate,
        types=types,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
