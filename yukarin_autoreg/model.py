from typing import Optional

import chainer
import chainer.functions as F
import numpy as np
from chainer import Chain

from yukarin_autoreg.config import LossConfig, ModelConfig
from yukarin_autoreg.network import WaveRNN


def create_predictor(config: ModelConfig):
    predictor = WaveRNN(
        upconv_scales=config.upconv_scales,
        upconv_residual=config.upconv_residual,
        upconv_channel_ksize=config.upconv_channel_ksize,
        residual_encoder_channel=config.residual_encoder_channel,
        residual_encoder_num_block=config.residual_encoder_num_block,
        dual_softmax=config.dual_softmax,
        bit_size=config.bit_size,
        hidden_size=config.hidden_size,
        local_size=config.local_size,
        bug_fixed_gru_dimension=config.bug_fixed_gru_dimension,
    )
    return predictor


class Model(Chain):
    def __init__(self, loss_config: LossConfig, predictor: WaveRNN) -> None:
        super().__init__()
        self.loss_config = loss_config
        with self.init_scope():
            self.predictor = predictor

    def __call__(
            self,
            input_coarse: np.ndarray,
            input_fine: Optional[np.ndarray],
            target_coarse: np.ndarray,
            target_fine: np.ndarray,
            local: Optional[np.ndarray],
            silence: np.ndarray,
    ):
        out_c_array, out_f_array, _, _ = self.predictor(
            c_array=input_coarse,
            f_array=input_fine,
            l_array=local,
        )

        losses = dict()

        nll_coarse = F.softmax_cross_entropy(out_c_array, target_coarse, reduce='no')
        if self.loss_config.eliminate_silence:
            nll_coarse = nll_coarse[~silence]
        losses['nll_coarse'] = F.mean(nll_coarse)

        loss = nll_coarse

        if not self.loss_config.disable_fine:
            nll_fine = F.softmax_cross_entropy(out_f_array, target_fine, reduce='no')
            if self.loss_config.eliminate_silence:
                nll_fine = nll_fine[~silence]
            loss += nll_fine
            losses['nll_fine'] = F.mean(nll_fine)

        loss = F.mean(loss)
        losses['loss'] = loss

        chainer.report(losses, self)
        return loss
