"""生成済みのファイルなどを集めてtfeventsを作る。"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, DefaultDict

import librosa
import numpy
from tensorboardX import SummaryWriter

from scripts.calc_diff_two_wave import calc_diff_two_wave


def _to_nums(p: Path, _re=re.compile(r'\d+')):
    return tuple(map(int, _re.findall(str(p))))


def collect_to_tfevents(
        input_dir: Path,
        output_dir: Optional[Path],
        filename_suffix: str,
        audio_tag_format: str,
        diff_tag: str,
        iteration_format: str,
        remove_exist: bool,
        expected_wave_dir: Optional[Path],
):
    if output_dir is None:
        output_dir = input_dir

    if remove_exist:
        for p in output_dir.glob(f'*tfevents*{filename_suffix}'):
            p.unlink()

    flag_calc_diff = expected_wave_dir is not None

    summary_writer = SummaryWriter(logdir=str(output_dir), filename_suffix=filename_suffix)

    diffs: DefaultDict[int, List[float]] = defaultdict(list)
    for p in sorted(input_dir.rglob('*'), key=_to_nums):
        if p.is_dir():
            continue

        if 'tfevents' in p.name:
            continue

        rp = p.relative_to(input_dir)
        iteration = int(iteration_format.format(p=p, rp=rp))

        # audio
        if p.suffix in ['.wav']:
            wave, sr = librosa.load(str(p), sr=None)
            summary_writer.add_audio(
                tag=audio_tag_format.format(p=p, rp=rp),
                snd_tensor=wave,
                sample_rate=sr,
                global_step=iteration,
            )

        # diff
        if flag_calc_diff and p.name.endswith('_woc.wav'):
            wave_id = p.name[:-8]
            expected = next(expected_wave_dir.glob(f'{wave_id}.*'))

            diff = calc_diff_two_wave(path1=expected, path2=p)
            diffs[iteration].append(diff)

    if flag_calc_diff:
        for iteration, values in sorted(diffs.items()):
            summary_writer.add_scalar(
                tag=diff_tag,
                scalar_value=numpy.mean(values),
                global_step=iteration,
            )

    summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path)
    parser.add_argument('--output_dir', type=Path, default=None)
    parser.add_argument('--filename_suffix', type=str, default='')
    parser.add_argument('--audio_tag_format', type=str, default='{p.name}')
    parser.add_argument('--diff_tag', type=str, default='diff')
    parser.add_argument('--iteration_format', type=str, default='{rp.parent}')
    parser.add_argument('--disable_remove_exist', action='store_true')
    parser.add_argument('--expected_wave_dir', type=Path, default=None)
    args = parser.parse_args()

    collect_to_tfevents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filename_suffix=args.filename_suffix,
        audio_tag_format=args.audio_tag_format,
        diff_tag=args.diff_tag,
        iteration_format=args.iteration_format,
        remove_exist=not args.disable_remove_exist,
        expected_wave_dir=args.expected_wave_dir,
    )
