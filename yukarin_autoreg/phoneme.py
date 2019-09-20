from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np


class BasePhoneme(object):
    phoneme_list = None
    num_phoneme = None
    space_phoneme = None

    def __init__(
            self,
            phoneme: str,
            start: float,
            end: float,
    ) -> None:
        self.phoneme = phoneme
        self.start = np.round(start, decimals=2)
        self.end = np.round(end, decimals=2)

    def __repr__(self):
        return f'Phoneme(phoneme=\'{self.phoneme}\', start={self.start}, end={self.end})'

    def verify(self):
        assert self.phoneme in self.phoneme_list, f'{self.phoneme} is not defined.'

    @property
    def phoneme_id(self):
        return self.phoneme_list.index(self.phoneme)

    @property
    def duration(self):
        return self.end - self.start

    @property
    def onehot(self):
        array = np.zeros(self.num_phoneme, dtype=bool)
        array[self.phoneme_id] = True
        return array

    @classmethod
    def parse(cls, s: str):
        """
        >>> BasePhoneme.parse('1.7425000 1.9125000 o:')
        Phoneme(phoneme='o:', start=1.74, end=1.91)
        """
        words = s.split()
        return cls(
            start=float(words[0]),
            end=float(words[1]),
            phoneme=words[2],
        )

    @classmethod
    @abstractmethod
    def convert(cls, phonemes: List['BasePhoneme']) -> List['BasePhoneme']:
        pass

    @classmethod
    def load_julius_list(cls, path: Path):
        phonemes = [
            cls.parse(s)
            for s in path.read_text().split('\n')
            if len(s) > 0
        ]
        phonemes = cls.convert(phonemes)

        for phoneme in phonemes:
            phoneme.verify()
        return phonemes


class SegKitPhoneme(BasePhoneme):
    phoneme_list = (
        'a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:', 'N', 'w', 'y', 'j',
        'my', 'ky', 'dy', 'by', 'gy', 'ny', 'hy', 'ry', 'py',
        'p', 't', 'k', 'ts', 'ch', 'b', 'd', 'g', 'z',
        'm', 'n', 's', 'sh', 'h', 'f', 'r', 'q', 'sp',
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = 'sp'

    @classmethod
    def convert(cls, phonemes: List['SegKitPhoneme']):
        if 'sil' in phonemes[0].phoneme:
            phonemes[0].phoneme = cls.space_phoneme
        if 'sil' in phonemes[-1].phoneme:
            phonemes[-1].phoneme = cls.space_phoneme
        return phonemes


class JvsPhoneme(BasePhoneme):
    phoneme_list = (
        'pau', 'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm',
        'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh', 't', 'ts', 'u', 'v', 'w', 'y', 'z',
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = 'pau'

    @classmethod
    def convert(cls, phonemes: List['SegKitPhoneme']):
        if 'sil' in phonemes[0].phoneme:
            phonemes[0].phoneme = cls.space_phoneme
        if 'sil' in phonemes[-1].phoneme:
            phonemes[-1].phoneme = cls.space_phoneme
        return phonemes


class PhonemeType(str, Enum):
    seg_kit = 'seg_kit'
    jvs = 'jvs'


phoneme_type_to_class = {
    PhonemeType.seg_kit: SegKitPhoneme,
    PhonemeType.jvs: JvsPhoneme,
}
