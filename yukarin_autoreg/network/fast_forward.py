from typing import Union

import cupy as cp
import numpy as np

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
        xp=np,
):
    half = xp.array(0.5, dtype=xp.float32)
    zero = xp.array(0.0, dtype=xp.float32)
    one = xp.array(1.0, dtype=xp.float32)

    prev_xl = xp.concatenate((x_embedder_W[prev_x], prev_l), axis=1)  # (batch_size, ?)

    # gru_x = prev_xl.dot(gru_xw) + gru_xb
    gru_x = prev_xl.dot(gru_xw)
    gru_x += gru_xb

    # gru_h = hidden.dot(gru_hw) + gru_hb
    gru_h = hidden.dot(gru_hw)
    gru_h += gru_hb

    size = gru_x.shape[1] // 3
    W_r_x, W_z_x, W_x = gru_x[:, :size], gru_x[:, size:size * 2], gru_x[:, size * 2:]
    U_r_h, U_z_h, U_x = gru_h[:, :size], gru_h[:, size:size * 2], gru_h[:, size * 2:]

    # r = xp.tanh((W_r_x + U_r_h) * half) * half + half
    r = W_r_x
    r += U_r_h
    r *= half
    xp.tanh(r, r)
    r *= half
    r += half

    # z = xp.tanh((W_z_x + U_z_h) * half) * half + half
    z = W_z_x
    z += U_z_h
    z *= half
    xp.tanh(z, z)
    z *= half
    z += half

    # h_bar = xp.tanh(W_x + r * U_x)
    r *= U_x
    r += W_x
    xp.tanh(r, r)
    h_bar = r

    # new_hidden = z * hidden + (one - z) * h_bar
    hidden *= z
    z *= -1
    z += one
    h_bar *= z
    hidden += h_bar
    new_hidden = hidden

    # out_x = new_hidden.dot(O1_W) + O1_b
    out_x = new_hidden.dot(O1_W)
    out_x += O1_b

    xp.maximum(out_x, zero, out_x)

    # out_x = out_x.dot(O2_W) + O2_b
    out_x = out_x.dot(O2_W)
    out_x += O2_b
    return out_x, new_hidden


def _random_choice_p(
        prob: ArrayLike,
        xp=np,
):
    cumsum = xp.cumsum(prob)
    rand = xp.random.random() * cumsum[-1]
    return xp.searchsorted(cumsum, rand, side='right')


def fast_sampling(
        dist: ArrayLike,
        xp=np,
):
    dist -= xp.max(dist, axis=1, keepdims=True)
    xp.exp(dist, dist)
    dist /= xp.sum(dist, axis=1, keepdims=True)

    sampled = xp.zeros((dist.shape[0],), dtype=xp.int32)
    for i in range(dist.shape[0]):
        sampled[i] = _random_choice_p(dist[i], xp=xp)

    return sampled


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
            xp=xp,
        )
        x = fast_sampling(
            dist,
            xp=xp,
        )
