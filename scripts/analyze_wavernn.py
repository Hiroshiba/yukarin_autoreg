import argparse
import glob
import re
from pathlib import Path

import chainer
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda

from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.dataset import encode_16bit, normalize
from yukarin_autoreg.network import create_predictor
from yukarin_autoreg.sampling_data import SamplingData
from yukarin_autoreg.wave import Wave

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-md', type=Path)
parser.add_argument('--model_iteration', '-mi', type=int)
parser.add_argument('--model_config', '-mc', type=Path)
parser.add_argument('--time_length', '-tl', type=float, default=1)
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

model_dir: Path = arguments.model_dir
model_iteration: int = arguments.model_iteration
model_config: Path = arguments.model_config
time_length: float = arguments.time_length
gpu: int = arguments.gpu


def _extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(model_dir: Path, iteration: int = None, prefix: str = 'main_'):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        fn = prefix + '{}.npz'.format(iteration)
        model_path = model_dir / fn
    return model_path


def main():
    config = create_config(model_config)
    model_path = _get_predictor_model_path(model_dir, model_iteration)

    sr = config.dataset.sampling_rate

    model = create_predictor(config.model)
    chainer.serializers.load_npz(str(model_path), model)
    if gpu is not None:
        model.to_gpu(gpu)
        cuda.get_device_from_id(gpu).use()

    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    wave_paths = sorted([Path(p) for p in glob.glob(str(config.dataset.input_wave_glob))])
    local_paths = sorted([Path(p) for p in glob.glob(str(config.dataset.input_local_glob))])
    assert len(wave_paths) == len(local_paths)

    np.random.RandomState(config.dataset.seed).shuffle(wave_paths)
    np.random.RandomState(config.dataset.seed).shuffle(local_paths)
    wave_path = wave_paths[0]
    local_path = local_paths[0]

    length = int(sr * time_length)
    w = Wave.load(wave_path, sampling_rate=sr).wave[:length]
    l = SamplingData.load(local_path).resample(sr, index=0, length=int(time_length * sr))
    coarse, fine = encode_16bit(w)

    c, f, hc, hf = model(
        c_array=normalize(model.xp.asarray(coarse)).astype(np.float32)[np.newaxis],
        f_array=normalize(model.xp.asarray(fine)).astype(np.float32)[:-1][np.newaxis],
        l_array=model.xp.asarray(l)[1:][np.newaxis],
    )

    c = chainer.functions.softmax(c)

    c = chainer.cuda.to_cpu(c[0].data)
    f = chainer.cuda.to_cpu(f[0].data)

    fig = plt.figure(figsize=[32 * time_length, 10])

    plt.imshow(c, aspect='auto', interpolation='nearest')
    plt.colorbar()

    plt.plot((w + 1) * 127.5, 'g', linewidth=0.1, label='true')
    plt.plot(np.argmax(c, axis=0), 'r', linewidth=0.1, label='predicted')
    plt.legend()

    fig.savefig('output.eps')


if __name__ == '__main__':
    main()
