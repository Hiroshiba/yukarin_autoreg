from typing import Union

import cupy as cp
import numba
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


@numba.jit
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
    half = xp.array(0.5, dtype=np.float32)
    zero = xp.array(0.0, dtype=np.float32)
    one = xp.array(1.0, dtype=np.float32)

    prev_xl = xp.concatenate((x_embedder_W[prev_x], prev_l), axis=1)  # (batch_size, ?)

    gru_x = prev_xl.dot(gru_xw) + gru_xb
    gru_h = hidden.dot(gru_hw) + gru_hb

    size = gru_x.shape[1] // 3
    W_r_x, W_z_x, W_x = gru_x[:, :size], gru_x[:, size:size * 2], gru_x[:, size * 2:]
    U_r_h, U_z_h, U_x = gru_h[:, :size], gru_h[:, size:size * 2], gru_h[:, size * 2:]

    r = xp.tanh((W_r_x + U_r_h) * half) * half + half
    z = xp.tanh((W_z_x + U_z_h) * half) * half + half

    h_bar = xp.tanh(W_x + r * U_x)
    new_hidden = z * hidden + (one - z) * h_bar

    out_x = new_hidden.dot(O1_W) + O1_b
    out_x = xp.maximum(out_x, zero)
    out_x = out_x.dot(O2_W) + O2_b  # (batch_size, ?)
    return out_x, new_hidden


@numba.jit('float32[:, :](float32[:, :])')
def _max_axis1_keepdims(array):
    out = np.zeros((array.shape[0], 1), dtype=array.dtype)
    for i in range(array.shape[0]):
        out[i] = array[i].max()
    return out


@numba.jit
def _random_choice_p(prob):
    cumsum = np.cumsum(prob)
    rand = np.random.random() * cumsum[-1]
    return np.searchsorted(cumsum, rand, side="right")


@numba.jit('int32[:](float32[:, :])')
def fast_sampling(dist: np.ndarray):
    dist -= _max_axis1_keepdims(dist)
    dist = np.exp(dist)
    dist /= _max_axis1_keepdims(dist)

    sampled = np.zeros((dist.shape[0],), dtype=np.int32)
    for i in range(dist.shape[0]):
        sampled[i] = _random_choice_p(dist[i])

    return sampled


@numba.jit
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
        )
        x = fast_sampling(dist)
