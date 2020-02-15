from typing import Union, Optional, Sequence

import cupy as cp
import numpy as np

ArrayLike = Union[np.ndarray, cp.ndarray]


def fast_forward_one(
        prev_x: ArrayLike,
        prev_l: ArrayLike,
        hidden: Optional[ArrayLike],
        input_categorical: bool,
        x_embedder_W: ArrayLike,
        gru_w: Sequence[ArrayLike],
        gru_b: Sequence[ArrayLike],
        gru_n_layers: int,
        gru_direction: int,
        gru_out_size: int,
        O1_W: ArrayLike,
        O1_b: ArrayLike,
        O2_W: ArrayLike,
        O2_b: ArrayLike,
        xp=np,
):
    dtype = O1_W.dtype
    half = dtype.type(0.5)
    zero = dtype.type(0)
    one = dtype.type(1)

    batch_size = prev_x.shape[0]

    if input_categorical:
        x = prev_x[:, np.newaxis]
        W = x_embedder_W
        prev_x = W[x].reshape([batch_size, -1])  # (batch_size, ?)
    else:
        prev_x = prev_x.reshape([batch_size, 1])  # (batch_size, 1)

    prev_xl = xp.concatenate((prev_x, prev_l), axis=1)  # (batch_size, ?)

    if hidden is None:
        shape = (gru_n_layers * gru_direction, len(prev_xl), gru_out_size)
        hx = xp.zeros(shape, dtype=dtype)
        hidden = hx[0]

    x = prev_xl
    h = hidden
    w = gru_w
    b = gru_b

    xw = xp.concatenate([w[0], w[1], w[2]], axis=0)
    hw = xp.concatenate([w[3], w[4], w[5]], axis=0)
    xb = xp.concatenate([b[0], b[1], b[2]], axis=0)
    hb = xp.concatenate([b[3], b[4], b[5]], axis=0)

    gru_x = x.dot(xw.T) + xb
    gru_h = h.dot(hw.T) + hb

    W_r_x, W_z_x, W_x = xp.split(gru_x, 3, axis=1)
    U_r_h, U_z_h, U_x = xp.split(gru_h, 3, axis=1)

    r = xp.tanh((W_r_x + U_r_h) * half) * half + half
    z = xp.tanh((W_z_x + U_z_h) * half) * half + half

    h_bar = xp.tanh(W_x + r * U_x)
    h = z * hidden + (one - z) * h_bar

    new_hidden = h

    out_x = new_hidden.dot(O1_W.T) + O1_b
    out_x = xp.maximum(out_x, zero)
    out_x = out_x.dot(O2_W.T) + O2_b  # (batch_size, ?)
    return out_x, new_hidden
