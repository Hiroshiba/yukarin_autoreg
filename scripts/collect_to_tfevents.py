"""生成済みのファイルなどを集めてtfeventsを作る。"""

import argparse
import re
from pathlib import Path
from typing import Optional

import librosa
from tensorboardX import SummaryWriter


def _to_nums(p: Path, _re=re.compile(r'\d+')):
    return tuple(map(int, _re.findall(str(p))))


def collect_to_tfevents(
        input_dir: Path,
        output_dir: Optional[Path],
        filename_suffix: str,
        tag_format: str,
        iteration_format: str,
        remove_exist: bool,
):
    if output_dir is None:
        output_dir = input_dir

    if remove_exist:
        for p in output_dir.glob(f'*tfevents*{filename_suffix}'):
            p.unlink()

    summary_writer = SummaryWriter(logdir=str(output_dir), filename_suffix=filename_suffix)

    for p in sorted(input_dir.rglob('*'), key=_to_nums):
        if p.is_dir():
            continue

        if 'tfevents' in p.name:
            continue

        rp = p.relative_to(input_dir)
        iteration = int(iteration_format.format(p=p, rp=rp))

        if p.suffix in ['.wav']:
            wave, sr = librosa.load(str(p), sr=None)
            summary_writer.add_audio(
                tag=tag_format.format(p=p, rp=rp),
                snd_tensor=wave,
                sample_rate=sr,
                global_step=iteration,
            )

    summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path)
    parser.add_argument('--output_dir', type=Path, default=None)
    parser.add_argument('--filename_suffix', type=str, default='')
    parser.add_argument('--tag_format', type=str, default='{p.name}')
    parser.add_argument('--iteration_format', type=str, default='{rp.parent}')
    parser.add_argument('--disable_remove_exist', action='store_true')
    args = parser.parse_args()

    collect_to_tfevents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filename_suffix=args.filename_suffix,
        tag_format=args.tag_format,
        iteration_format=args.iteration_format,
        remove_exist=not args.disable_remove_exist,
    )
