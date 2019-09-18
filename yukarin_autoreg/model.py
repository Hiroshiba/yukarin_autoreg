from typing import Optional

import chainer
import chainer.functions as F
import numpy as np
from chainer import Chain

from yukarin_autoreg.config import LossConfig, ModelConfig
from yukarin_autoreg.network.wave_rnn import WaveRNN
from yukarin_autoreg.utility.chainer_initializer_utility import get_weight_initializer


def create_predictor(config: ModelConfig):
    predictor = WaveRNN(
        dual_softmax=config.dual_softmax,
        bit_size=config.bit_size,
        gaussian=config.gaussian,
        input_categorical=config.input_categorical,
        conditioning_size=config.conditioning_size,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        linear_hidden_size=config.linear_hidden_size,
        local_size=config.local_size,
        local_scale=config.local_scale,
        local_layer_num=config.local_layer_num,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        weight_initializer=get_weight_initializer(config.weight_initializer),
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
            coarse: np.ndarray,
            fine: Optional[np.ndarray],
            encoded_coarse: np.ndarray,
            encoded_fine: np.ndarray,
            local: Optional[np.ndarray],
            silence: np.ndarray,
            speaker_num: Optional[np.ndarray],
    ):
        assert fine is None

        if self.predictor.input_categorical:
            x_array = encoded_coarse
        else:
            x_array = coarse

        out_c_array, _ = self.predictor(
            x_array=x_array,
            l_array=local,
            s_one=speaker_num,
            local_padding_size=self.local_padding_size,
        )
        out_f_array = None

        losses = dict()

        if not self.predictor.gaussian:
            target_coarse = encoded_coarse[:, 1:]
            nll_coarse = F.softmax_cross_entropy(out_c_array, target_coarse, reduce='no')
        else:
            target_coarse = coarse[:, 1:]
            nll_coarse = F.gaussian_nll(target_coarse, mean=out_c_array[:, 0], ln_var=out_c_array[:, 1], reduce='no')

        if self.loss_config.eliminate_silence:
            nll_coarse = nll_coarse[~silence]
        losses['nll_coarse'] = F.mean(nll_coarse)

        loss = nll_coarse

        if not self.loss_config.disable_fine:
            if not self.predictor.gaussian:
                target_fine = encoded_fine[:, 1:]
                nll_fine = F.softmax_cross_entropy(out_f_array, target_fine, reduce='no')
            else:
                target_fine = fine[:, 1:]
                nll_fine = F.gaussian_nll(target_fine, mean=out_f_array[:, 0], ln_var=out_f_array[:, 1], reduce='no')

            if self.loss_config.eliminate_silence:
                nll_fine = nll_fine[~silence]
            losses['nll_fine'] = F.mean(nll_fine)

            loss += nll_fine

        loss = F.mean(loss)
        losses['loss'] = loss

        chainer.report(losses, self)
        return loss
