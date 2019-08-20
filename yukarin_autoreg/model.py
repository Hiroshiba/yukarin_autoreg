from typing import Optional, Union

import chainer
import chainer.functions as F
import numpy as np
from chainer import Chain

from yukarin_autoreg.config import LossConfig, ModelConfig
from yukarin_autoreg.network import WaveRNN
from yukarin_autoreg.network.univ_wave_rnn import UnivWaveRNN


def create_predictor(config: ModelConfig):
    if not config.use_univ_wavernn:
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
    else:
        predictor = UnivWaveRNN(
            dual_softmax=config.dual_softmax,
            bit_size=config.bit_size,
            conditioning_size=config.conditioning_size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            linear_hidden_size=config.linear_hidden_size,
            local_size=config.local_size,
            local_scale=config.local_scale,
        )
    return predictor


class Model(Chain):
    def __init__(self, loss_config: LossConfig, predictor: Union[WaveRNN, UnivWaveRNN]) -> None:
        super().__init__()
        self.loss_config = loss_config
        with self.init_scope():
            self.predictor = predictor

    def __call__(
            self,
            input_coarse: np.ndarray,
            input_fine: Optional[np.ndarray],
            encoded_coarse: np.ndarray,
            encoded_fine: np.ndarray,
            local: Optional[np.ndarray],
            silence: np.ndarray,
    ):
        if isinstance(self.predictor, WaveRNN):
            out_c_array, out_f_array, _, _ = self.predictor(
                c_array=input_coarse,
                f_array=input_fine,
                l_array=local,
            )
        else:
            assert input_fine is None
            out_c_array, _ = self.predictor(
                x_array=encoded_coarse,
                l_array=local,
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
