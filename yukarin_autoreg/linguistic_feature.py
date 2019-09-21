from enum import Enum
from typing import List, Sequence, Union, Type

import numpy as np

from yukarin_autoreg.phoneme import SegKitPhoneme, BasePhoneme


class LinguisticFeature(object):
    class FeatureType(str, Enum):
        PHONEME = 'PHONEME'
        PRE_PHONEME = 'PRE_PHONEME'
        POST_PHONEME = 'POST_PHONEME'
        PHONEME_DURATION = 'PHONEME_DURATION'
        PRE_PHONEME_DURATION = 'PRE_PHONEME_DURATION'
        POST_PHONEME_DURATION = 'POST_PHONEME_DURATION'
        POS_IN_PHONEME = 'POS_IN_PHONEME'

        def is_phoneme(self):
            return self in (
                self.PHONEME,
                self.PRE_PHONEME,
                self.POST_PHONEME,
                self.PHONEME_DURATION,
                self.PRE_PHONEME_DURATION,
                self.POST_PHONEME_DURATION,
            )

    def __init__(
            self,
            phonemes: List[BasePhoneme],
            phoneme_class: Type[BasePhoneme],
            rate: int,
            feature_types: Sequence[Union[FeatureType, str]],
    ) -> None:
        self.phonemes = phonemes
        self.phoneme_class = phoneme_class
        self.rate = rate
        self.types = [self.FeatureType(t) for t in feature_types]

    def get_dim(self, t: FeatureType) -> int:
        return {
            t.PHONEME: self.phoneme_class.num_phoneme,
            t.PRE_PHONEME: self.phoneme_class.num_phoneme,
            t.POST_PHONEME: self.phoneme_class.num_phoneme,
            t.PHONEME_DURATION: 1,
            t.PRE_PHONEME_DURATION: 1,
            t.POST_PHONEME_DURATION: 1,
            t.POS_IN_PHONEME: 2,
        }[t]

    def sum_dims(self, types: List[FeatureType]):
        return sum(self.get_dim(t) for t in types)

    def _to_index(self, t: float):
        return int(round(t * self.rate))

    def _to_time(self, i: Union[int, np.ndarray]):
        return i / self.rate

    @property
    def len_array(self):
        return self._to_index(self.phonemes[-1].end) + 1

    def _get_phoneme(self, i: int):
        if 0 <= i < len(self.phonemes):
            return self.phonemes[i]
        elif i < 0:
            return SegKitPhoneme(
                phoneme=SegKitPhoneme.space_phoneme,
                start=self.phonemes[0].start,
                end=self.phonemes[0].start,
            )
        else:
            return SegKitPhoneme(
                phoneme=SegKitPhoneme.space_phoneme,
                start=self.phonemes[-1].end,
                end=self.phonemes[-1].end,
            )

    def _make_phoneme_array(self):
        types = list(filter(self.FeatureType.is_phoneme, self.types))

        array = np.zeros((len(self.phonemes), self.sum_dims(types)), dtype=np.float32)
        for i in range(len(self.phonemes)):
            features = [
                {
                    self.FeatureType.PHONEME: self._get_phoneme(i).onehot,
                    self.FeatureType.PRE_PHONEME: self._get_phoneme(i - 1).onehot,
                    self.FeatureType.POST_PHONEME: self._get_phoneme(i + 1).onehot,
                    self.FeatureType.PHONEME_DURATION: self._get_phoneme(i).duration,
                    self.FeatureType.PRE_PHONEME_DURATION: self._get_phoneme(i - 1).duration,
                    self.FeatureType.POST_PHONEME_DURATION: self._get_phoneme(i + 1).duration,
                }[t]
                for t in types
            ]
            array[i] = np.concatenate([np.asarray(f).reshape(1, -1) for f in features], axis=1)
        return array

    def make_array(self):
        phoneme_array = self._make_phoneme_array()

        array = np.zeros((self.len_array, self.sum_dims(self.types)), dtype=np.float32)
        for i, p in enumerate(self.phonemes):
            s = self._to_index(p.start)
            e = self._to_index(p.end)

            features = [np.repeat(phoneme_array[i].reshape(1, -1), repeats=e - s + 1, axis=0)]

            if self.FeatureType.POS_IN_PHONEME in self.types:
                pos_start = (self._to_time(np.arange(s, e + 1)) - p.start).reshape(-1, 1)
                pos_end = p.duration - pos_start
                features.append(pos_start)
                features.append(pos_end)

            array[s:e + 1] = np.concatenate(features, axis=1)
        return array
