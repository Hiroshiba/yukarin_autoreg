import argparse
import time

import numpy as np
from chainer import cuda

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.model import create_predictor

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

gpu: int = arguments.gpu


def main():
    if arguments.gpu >= 0:
        cuda.get_device_from_id(arguments.gpu).use()

    config = ModelConfig(
        hidden_size=768,
        bit_size=16,
    )
    model = create_predictor(config)
    model.to_gpu(arguments.gpu)

    batch_size = 4
    length = 256

    for i in range(10):
        c_array = model.xp.random.rand(batch_size, length + 1).astype(np.float32)
        f_array = model.xp.random.rand(batch_size, length).astype(np.float32)

        start = time.time()
        _, _, hidden_coarse, hidden_fine = model(c_array, f_array)

        elapsed = time.time() - start

        # print(hidden_coarse.data > 0)
        print(f'elapsed time: {elapsed}')

    # for i in range(10):
    #     c_start = model.xp.random.rand(batch_size).astype(np.float32)
    #     f_start = model.xp.random.rand(batch_size).astype(np.float32)
    #
    #     c, f = c_start, f_start
    #     hc = hf = None
    #
    #     start = time.time()
    #     for _ in range(length):
    #         with chainer.using_config('train', False):
    #             c, f, hc, hf = model.forward_one(prev_c=c, prev_f=f, hidden_coarse=hc, hidden_fine=hf)
    #         c = model.sampling(c, maximum=True)
    #         f = model.sampling(f, maximum=True)
    #
    #     elapsed = time.time() - start
    #
    #     print(c, f)
    #     print(f'elapsed time: {elapsed}')


if __name__ == '__main__':
    main()
