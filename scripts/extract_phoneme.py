import argparse
import glob
import multiprocessing
import numpy as np
import tqdm
from functools import partial
from pathlib import Path
from typing import List

from utility.json_utility import save_arguments
from yukarin_autoreg.phoneme import Phoneme


def _make_phoneme_array(ps: List[Phoneme], rate: int):
    to_index = lambda t: int(t * rate)
    ma = to_index(ps[-1].end) + 1
    onehots = np.zeros((ma, Phoneme.num_phoneme), dtype=bool)

    for p in ps:
        s = to_index(p.start)
        e = to_index(p.end)
        onehots[s:e, p.phoneme_id] = True

    return onehots


def process(
        path: Path,
        output_directory: Path,
        with_pre_post: bool,
        rate: int,
):
    ps = Phoneme.load_julius_list(path)
    array = _make_phoneme_array(ps, rate=rate)

    if with_pre_post:
        ps_pre = [
            Phoneme(phoneme=ps[i - 1].phoneme if i > 0 else Phoneme.space_phoneme, start=p.start, end=p.end)
            for i, p in enumerate(ps)
        ]
        array_pre = _make_phoneme_array(ps_pre, rate=rate)

        ps_post = [
            Phoneme(phoneme=ps[i + 1].phoneme if i < len(ps) - 1 else Phoneme.space_phoneme, start=p.start, end=p.end)
            for i, p in enumerate(ps)
        ]
        array_post = _make_phoneme_array(ps_post, rate=rate)

        array = np.concatenate((array_pre, array, array_post), axis=1)

    out = output_directory / (path.stem + '.npy')
    np.save(str(out), dict(array=array, rate=rate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', '-ig')
    parser.add_argument('--output_directory', '-od', type=Path)
    parser.add_argument('--with_pre_post', '-wpp', action='store_true')
    parser.add_argument('--rate', '-r', type=int, default=100)
    config = parser.parse_args()

    input_glob = config.input_glob
    output_directory: Path = config.output_directory
    with_pre_post: bool = config.with_pre_post
    rate: int = config.rate

    output_directory.mkdir(exist_ok=True)
    save_arguments(config, output_directory / 'arguments.json')

    paths = [Path(p) for p in glob.glob(str(input_glob))]
    _process = partial(
        process,
        output_directory=output_directory,
        with_pre_post=with_pre_post,
        rate=rate,
    )

    pool = multiprocessing.Pool()
    list(tqdm.tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


if __name__ == '__main__':
    main()
