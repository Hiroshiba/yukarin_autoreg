import argparse
import time
from collections import namedtuple

import chainer
import chainerx
import numpy as np

from yukarin_autoreg.dataset import normalize
from yukarin_autoreg.model import create_predictor

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

gpu: int = arguments.gpu


def main():
    device = f'cuda:{arguments.gpu}'

    if arguments.gpu >= 0:
        chainerx.using_device(device)

    config = namedtuple('ModelConfig', [
        'upconv_scales',
        'upconv_residual',
        'upconv_channel_ksize',
        'residual_encoder_channel',
        'residual_encoder_num_block',
        'dual_softmax',
        'bit_size',
        'hidden_size',
        'local_size',
    ])(
        upconv_scales=[],
        upconv_residual=False,
        upconv_channel_ksize=0,
        residual_encoder_channel=None,
        residual_encoder_num_block=None,
        dual_softmax=True,
        bit_size=16,
        hidden_size=896,
        local_size=0,
    )
    model = create_predictor(config)
    model.to_device(device)

    batch_size = 4
    length = 1024

    # non recurrent
    for i in range(10):
        c_array = chainerx.array(np.random.normal(size=(batch_size, length + 1)).astype(np.float32), device=device)
        f_array = chainerx.array(np.random.normal(size=(batch_size, length)).astype(np.float32), device=device)
        l_array = chainerx.array(np.empty((batch_size, length + 1, 0), dtype=np.float32), device=device)

        start = time.time()
        with chainer.using_config('train', False), \
             chainer.using_config('enable_backprop', False), \
             chainerx.no_backprop_mode():
            _, _, hidden_coarse, hidden_fine = model(c_array, f_array, l_array)
        elapsed = time.time() - start
        print(f'non recurrent time: {elapsed}')

    # recurrent
    for i in range(10):
        c = chainerx.array(np.random.normal(size=(1,)).astype(np.float32), device=device)
        f = chainerx.array(np.random.normal(size=(1,)).astype(np.float32), device=device)
        l_array = chainerx.array(np.empty((length, 0), dtype=np.float32)[np.newaxis], device=device)

        hc, hf = None, None

        start = time.time()
        for i in range(length):
            with chainer.using_config('train', False), \
                 chainer.using_config('enable_backprop', False), \
                 chainerx.no_backprop_mode():
                c, f, hc, hf = model.forward_one(
                    prev_c=c,
                    prev_f=f,
                    prev_l=l_array[:, i],
                    hidden_coarse=hc,
                    hidden_fine=hf,
                )

            c = model.sampling(c, maximum=True)
            f = model.sampling(f, maximum=True)

            c = normalize(c.astype(np.float32))
            f = normalize(f.astype(np.float32))

        elapsed = time.time() - start
        print(f'recurrent time: {elapsed}')


if __name__ == '__main__':
    main()
