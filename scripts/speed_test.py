import argparse
import time
from collections import namedtuple

import numpy as np
from chainer import cuda

from yukarin_autoreg.model import create_predictor

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

gpu: int = arguments.gpu


def main():
    if arguments.gpu >= 0:
        cuda.get_device_from_id(arguments.gpu).use()

    config = namedtuple('ModelConfig', [
        'hidden_size',
        'bit_size',
        'local_size',
        'upconv_scales',
        'upconv_residual',
        'upconv_channel_ksize',
        'residual_encoder_channel',
        'residual_encoder_num_block',
    ])(
        hidden_size=896,
        bit_size=16,
        local_size=0,
        upconv_scales=[],
        upconv_residual=False,
        upconv_channel_ksize=0,
        residual_encoder_channel=None,
        residual_encoder_num_block=None,
    )
    model = create_predictor(config)
    model.to_gpu(arguments.gpu)

    batch_size = 4
    length = 1024

    for i in range(10):
        c_array = model.xp.random.rand(batch_size, length + 1).astype(np.float32)
        f_array = model.xp.random.rand(batch_size, length).astype(np.float32)
        l_array = model.xp.empty((batch_size, length + 1, 0), dtype=np.float32)

        start = time.time()
        _, _, hidden_coarse, hidden_fine = model(c_array, f_array, l_array)
        elapsed = time.time() - start

        # print(hidden_coarse.data > 0)
        print(f'elapsed time: {elapsed}')


if __name__ == '__main__':
    main()
