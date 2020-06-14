import argparse
import re
from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np
import pyworld
from acoustic_feature_extractor.data.f0 import F0
from acoustic_feature_extractor.data.wave import Wave
from tqdm import tqdm


def _sort_path(paths: Iterable[Path]):
    return sorted(paths, key=lambda p: tuple(map(int, re.findall(r"(\d+)", str(p)))))


def process(
    input_paths: Tuple[Path, Path], output_dir: Path,
):
    input_wave, input_f0 = input_paths

    wave_data = Wave.load(input_wave)
    f0_data = F0.load(input_f0)

    y = wave_data.wave.astype(np.float64)
    sr = wave_data.sampling_rate

    f0 = np.exp(f0_data.array[:, 0].astype(np.float64))
    if f0_data.with_vuv:
        f0[~f0_data.array[:, 1]] = 0

    t = np.arange(0, len(f0), dtype=np.float64) / f0_data.rate
    sp = pyworld.cheaptrick(y, f0, t, sr)
    ap = pyworld.d4c(y, f0, t, sr)

    y = pyworld.synthesize(f0, sp, ap, sr)

    out = output_dir / f"{input_f0.stem}.wav"
    librosa.output.write_wav(out, y.astype(np.float32), sr)


def autotune(
    input_wave_glob: Path, input_f0_glob: Path, output_dir: Path,
):
    output_dir.mkdir(exist_ok=True)

    input_waves = _sort_path(map(Path, glob(str(input_wave_glob))))
    input_f0s = _sort_path(map(Path, glob(str(input_f0_glob))))
    assert len(input_waves) == len(input_f0s)

    input_paths = list(zip(input_waves, input_f0s))

    it = Pool().imap_unordered(partial(process, output_dir=output_dir,), input_paths)
    list(tqdm(it, total=len(input_paths), desc="autotune"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wave_glob", "-iwg", type=Path)
    parser.add_argument("--input_f0_glob", "-ifg", type=Path)
    parser.add_argument("--output_dir", "-o", type=Path)
    autotune(**vars(parser.parse_args()))
