import argparse
import glob
import re
from pathlib import Path

import chainer
import matplotlib.pyplot as plt
import numpy as np
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from chainer import cuda

from yukarin_autoreg.config import create_from_json as create_config
from yukarin_autoreg.data import encode_16bit, decode_single
from yukarin_autoreg.model import create_predictor

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-md', type=Path)
parser.add_argument('--model_iteration', '-mi', type=int)
parser.add_argument('--model_config', '-mc', type=Path)
parser.add_argument('--time_length', '-tl', type=float, default=1)
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()


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
    model_dir: Path = arguments.model_dir
    model_iteration: int = arguments.model_iteration
    model_config: Path = arguments.model_config
    time_length: float = arguments.time_length
    gpu: int = arguments.gpu

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
    w_data = Wave.load(wave_path, sampling_rate=sr)
    l_data = SamplingData.load(local_path)

    length = int(sr * time_length)
    l_scale = int(sr // l_data.rate)
    l_sl = length // l_scale
    length = l_sl * l_scale

    w = w_data.wave[:length]
    l = l_data.array[:l_sl]
    coarse, fine = encode_16bit(w)

    c, f, hc, hf = model(
        c_array=decode_single(model.xp.asarray(coarse)).astype(np.float32)[np.newaxis],
        f_array=decode_single(model.xp.asarray(fine)).astype(np.float32)[:-1][np.newaxis],
        l_array=model.xp.asarray(l)[np.newaxis],
    )

    c = chainer.functions.softmax(c)

    c = chainer.cuda.to_cpu(c[0].data)
    f = chainer.cuda.to_cpu(f[0].data)

    fig = plt.figure(figsize=[32 * time_length, 10])

    plt.imshow(c, aspect='auto', interpolation='nearest')
    plt.colorbar()

    plt.plot((w + 1) * 127.5, 'g', linewidth=0.1, label='true')
    plt.plot(np.argmax(c, axis=0) + np.argmax(f, axis=0) / 256, 'r', linewidth=0.1, label='predicted')
    plt.legend()

    fig.savefig('output.eps')


if __name__ == '__main__':
    main()
