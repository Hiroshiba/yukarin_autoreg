from typing import Optional, Union, Tuple

import chainer
import chainer.functions as F
import numpy as np
from chainer import Chain

from yukarin_autoreg.config import LossConfig, ModelConfig
from yukarin_autoreg.network.wave_rnn import WaveRNN


def create_predictor(config: ModelConfig):
    predictor = WaveRNN(
        dual_softmax=config.dual_softmax,
        bit_size=config.bit_size,
        conditioning_size=config.conditioning_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        linear_hidden_size=config.linear_hidden_size,
        local_size=config.local_size,
        local_scale=config.local_scale,
        local_layer_num=config.local_layer_num,
    )
    return predictor


class Model(Chain):
    def __init__(
            self,
            loss_config: LossConfig,
            predictor: WaveRNN,
            local_padding_size: int,
    ) -> None:
        super().__init__()
        self.loss_config = loss_config
        with self.init_scope():
            self.predictor = predictor
        self.local_padding_size = local_padding_size

    def __call__(
            self,
            input_coarse: np.ndarray,
            input_fine: Optional[np.ndarray],
            encoded_coarse: np.ndarray,
            encoded_fine: np.ndarray,
            local: Optional[np.ndarray],
            silence: np.ndarray,
    ):
        assert input_fine is None
        out_c_array, _ = self.predictor(
            x_array=encoded_coarse,
            l_array=local,
            local_padding_size=self.local_padding_size,
        )
        out_f_array = None

        losses = dict()

        target_coarse = encoded_coarse[:, 1:]
        nll_coarse = F.softmax_cross_entropy(out_c_array, target_coarse, reduce='no')
        if self.loss_config.eliminate_silence:
            nll_coarse = nll_coarse[~silence]
        losses['nll_coarse'] = F.mean(nll_coarse)

        loss = nll_coarse

        if not self.loss_config.disable_fine:
            target_fine = encoded_fine[:, 1:]
            nll_fine = F.softmax_cross_entropy(out_f_array, target_fine, reduce='no')
            if self.loss_config.eliminate_silence:
                nll_fine = nll_fine[~silence]
            loss += nll_fine
            losses['nll_fine'] = F.mean(nll_fine)

        loss = F.mean(loss)
        losses['loss'] = loss

        chainer.report(losses, self)
        return loss
