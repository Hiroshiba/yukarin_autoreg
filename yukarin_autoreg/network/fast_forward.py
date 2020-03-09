from typing import Union

import cupy as cp
import numpy as np
from chainer.backends import cuda as chainer_cuda

from yukarin_autoreg.network.wave_rnn import WaveRNN

ArrayLike = Union[np.ndarray, cp.ndarray]


def get_fast_forward_params(model: WaveRNN):
    xp = model.xp

    x_embedder_W = model.x_embedder.W.data

    gru_w = model.gru.ws[0]
    gru_xw = xp.concatenate([gru_w[0].data, gru_w[1].data, gru_w[2].data], 0).T
    gru_hw = xp.concatenate([gru_w[3].data, gru_w[4].data, gru_w[5].data], 0).T

    gru_b = model.gru.bs[0]
    gru_xb = xp.concatenate([gru_b[0].data, gru_b[1].data, gru_b[2].data], 0)
    gru_hb = xp.concatenate([gru_b[3].data, gru_b[4].data, gru_b[5].data], 0)

    O1_W = model.O1.W.data.T
    O1_b = model.O1.b.data
    O2_W = model.O2.W.data.T
    O2_b = model.O2.b.data

    return dict(
        x_embedder_W=x_embedder_W,
        gru_xw=gru_xw,
        gru_hw=gru_hw,
        gru_xb=gru_xb,
        gru_hb=gru_hb,
        O1_W=O1_W,
        O1_b=O1_b,
        O2_W=O2_W,
        O2_b=O2_b,
    )


@cp.fuse()
def calc_gru_r(W_r_x, U_r_h):
    # r = xp.tanh((W_r_x + U_r_h) * half) * half + half
    r = W_r_x
    r += U_r_h
    r *= 0.5
    cp.tanh(r, r)
    r *= 0.5
    r += 0.5
    return r


@cp.fuse()
def calc_gru_z(W_z_x, U_z_h):
    # z = xp.tanh((W_z_x + U_z_h) * half) * half + half
    z = W_z_x
    z += U_z_h
    z *= 0.5
    cp.tanh(z, z)
    z *= 0.5
    z += 0.5
    return z


@cp.fuse()
def calc_gru_h_bar(r, U_x, W_x):
    # h_bar = xp.tanh(W_x + r * U_x)
    r *= U_x
    r += W_x
    cp.tanh(r, r)
    return r


@cp.fuse()
def calc_gru_hidden(hidden, z, h_bar):
    # new_hidden = z * hidden + (one - z) * h_bar
    hidden *= z
    z *= -1
    z += 1
    h_bar *= z
    hidden += h_bar
    return hidden


@cp.fuse()
def gru_element_wise(hidden, W_r_x, W_z_x, W_x, U_r_h, U_z_h, U_x):
    r = calc_gru_r(W_r_x, U_r_h)
    z = calc_gru_z(W_z_x, U_z_h)
    h_bar = calc_gru_h_bar(r, U_x, W_x)
    return calc_gru_hidden(hidden, z, h_bar)


def fast_forward_one(
        prev_x: ArrayLike,
        prev_l: ArrayLike,
        hidden: ArrayLike,
        x_embedder_W: ArrayLike,
        gru_xw: ArrayLike,
        gru_hw: ArrayLike,
        gru_xb: ArrayLike,
        gru_hb: ArrayLike,
        O1_W: ArrayLike,
        O1_b: ArrayLike,
        O2_W: ArrayLike,
        O2_b: ArrayLike,
        w_gru_x: ArrayLike,
        w_gru_h: ArrayLike,
        w_out_x1: ArrayLike,
        w_out_x2: ArrayLike,
        xp=np,
):
    prev_xl = xp.concatenate((x_embedder_W[prev_x], prev_l), axis=1)  # (batch_size, ?)

    # gru_x = prev_xl.dot(gru_xw) + gru_xb
    gru_x = w_gru_x
    prev_xl.dot(gru_xw, gru_x)
    gru_x += gru_xb

    # gru_h = hidden.dot(gru_hw) + gru_hb
    gru_h = w_gru_h
    hidden.dot(gru_hw, gru_h)
    gru_h += gru_hb

    size = gru_x.shape[1] // 3
    W_r_x, W_z_x, W_x = gru_x[:, :size], gru_x[:, size:size * 2], gru_x[:, size * 2:]
    U_r_h, U_z_h, U_x = gru_h[:, :size], gru_h[:, size:size * 2], gru_h[:, size * 2:]
    new_hidden = gru_element_wise(hidden, W_r_x, W_z_x, W_x, U_r_h, U_z_h, U_x)

    # out_x = new_hidden.dot(O1_W) + O1_b
    out_x1 = w_out_x1
    new_hidden.dot(O1_W, out_x1)
    out_x1 += O1_b

    cp.maximum(out_x1, 0.0, out_x1)

    # out_x = out_x.dot(O2_W) + O2_b
    out_x2 = w_out_x2
    out_x1.dot(O2_W, out_x2)
    out_x2 += O2_b
    return out_x2, new_hidden


@cp.fuse()
def _support_choice(dist, rand):
    return cp.log(dist) + rand


def fast_generate(
        length: int,
        x: ArrayLike,
        l_array: ArrayLike,
        h: ArrayLike,
        x_embedder_W: ArrayLike,
        gru_xw: ArrayLike,
        gru_hw: ArrayLike,
        gru_xb: ArrayLike,
        gru_hb: ArrayLike,
        O1_W: ArrayLike,
        O1_b: ArrayLike,
        O2_W: ArrayLike,
        O2_b: ArrayLike,
        xp=np,
):
    batchsize = len(x)
    w_gru_x = xp.empty((batchsize, len(gru_xb)), dtype=h.dtype)
    w_gru_h = xp.empty((batchsize, len(gru_hb)), dtype=h.dtype)
    w_out_x1 = xp.empty((batchsize, len(O1_b)), dtype=h.dtype)
    w_out_x2 = xp.empty((batchsize, len(O2_b)), dtype=h.dtype)

    # random cache
    random_cache = cp.random.gumbel(size=(length, batchsize, len(O2_b)), dtype=h.dtype)

    for i in range(length):
        dist, h = fast_forward_one(
            prev_x=x,
            prev_l=l_array[:, i],
            hidden=h,
            x_embedder_W=x_embedder_W,
            gru_xw=gru_xw,
            gru_hw=gru_hw,
            gru_xb=gru_xb,
            gru_hb=gru_hb,
            O1_W=O1_W,
            O1_b=O1_b,
            O2_W=O2_W,
            O2_b=O2_b,
            w_gru_x=w_gru_x,
            w_gru_h=w_gru_h,
            w_out_x1=w_out_x1,
            w_out_x2=w_out_x2,
            xp=xp,
        )

        # softmax
        dist = chainer_cuda.cudnn.softmax_forward(dist, 1, chainer_cuda.libcudnn.CUDNN_SOFTMAX_ACCURATE)

        # sampling
        # x = (cp.log(dist) + cp.random.gumbel(size=dist.shape, dtype=dist.dtype)).argmax(axis=1)
        x = _support_choice(dist, random_cache[i]).argmax(axis=1)
