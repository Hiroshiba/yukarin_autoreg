import argparse
import time

import chainer
import numpy as np

from yukarin_autoreg.config import ModelConfig
from yukarin_autoreg.model import create_predictor
from yukarin_autoreg.utility.forward_utility import fast_forward_one

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
arguments = parser.parse_args()

gpu: int = arguments.gpu


def main():
    config = ModelConfig(
        dual_softmax=False,
        bit_size=12,
        gaussian=False,
        input_categorical=True,
        conditioning_size=None,
        embedding_size=16,
        hidden_size=896,
        linear_hidden_size=896,
        local_size=0,
        local_scale=None,
        local_layer_num=None,
        speaker_size=0,
        speaker_embedding_size=None,
        weight_initializer=None,
    )
    model = create_predictor(config)
    model.to_device(gpu)

    xp = model.xp

    batch_size = 4
    length = 1024

    # # non recurrent
    # for i in range(10):
    #     x_array = xp.array(np.random.randint(0, config.bit_size ** 2, size=(batch_size, length + 1)).astype(np.int32))
    #     l_array = xp.array(np.empty((batch_size, length + 1, 0), dtype=np.float32))
    #
    #     start = time.time()
    #     with chainer.using_config('train', False), \
    #          chainer.using_config('enable_backprop', False):
    #         _, hidden = model(x_array=x_array, l_array=l_array)
    #     elapsed = time.time() - start
    #     print(f'non recurrent time: {elapsed}')

    # recurrent
    for i in range(10):
        # not fast recurrent
        x_array = xp.array(np.random.randint(0, config.bit_size ** 2, size=(batch_size, length)).astype(np.int32))
        l_array = xp.array(np.empty((batch_size, length, 0), dtype=np.float32))

        h = None

        start = time.time()
        for i in range(length):
            with chainer.using_config('train', False), \
                 chainer.using_config('enable_backprop', False):
                x, h = model.forward_one(
                    prev_x=x_array[:, i],
                    prev_l=l_array[:, i],
                    hidden=h,
                )

            x = model.sampling(x, maximum=False)

        elapsed = time.time() - start
        print(f'not fast recurrent time: {elapsed}')

        # fast recurrent
        x_array = xp.array(np.random.randint(0, config.bit_size ** 2, size=(batch_size, length)).astype(np.int32))
        l_array = xp.array(np.empty((batch_size, length, 0), dtype=np.float32))

        h = None

        start = time.time()
        for i in range(length):
            x, h = fast_forward_one(
                prev_x=x_array[:, i],
                prev_l=l_array[:, i],
                hidden=h,
                input_categorical=config.input_categorical,
                x_embedder_W=model.x_embedder.W.data if model.input_categorical else None,
                gru_w=[v.data for v in model.gru.ws[0]],
                gru_b=[v.data for v in model.gru.bs[0]],
                gru_n_layers=model.gru.n_layers,
                gru_direction=model.gru.direction,
                gru_out_size=model.gru.out_size,
                O1_W=model.O1.W.data,
                O1_b=model.O1.b.data,
                O2_W=model.O2.W.data,
                O2_b=model.O2.b.data,
                xp=model.xp,
            )

            x = model.sampling(x, maximum=False)

        elapsed = time.time() - start
        print(f'fast recurrent time: {elapsed}')


if __name__ == '__main__':
    main()
