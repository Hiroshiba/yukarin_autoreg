from pathlib import Path

import numpy as np


class Phoneme(object):
    phoneme_list = (
        'a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:', 'N', 'w', 'y', 'j',
        'my', 'ky', 'dy', 'by', 'gy', 'ny', 'hy', 'ry', 'py',
        'p', 't', 'k', 'ts', 'ch', 'b', 'd', 'g', 'z',
        'm', 'n', 's', 'sh', 'h', 'f', 'r', 'q', 'sp',
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = 'sp'

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

    @staticmethod
    def parse(s: str):
        """
        >>> Phoneme.parse('1.7425000 1.9125000 o:')
        Phoneme(phoneme='o:', start=1.74, end=1.91)
        """
        words = s.split()
        return Phoneme(
            start=float(words[0]),
            end=float(words[1]),
            phoneme=words[2],
        )

    @staticmethod
    def load_julius_list(path: Path):
        phonemes = [
            Phoneme.parse(s)
            for s in path.read_text().split('\n')
            if len(s) > 0
        ]
        if 'sil' in phonemes[0].phoneme:
            phonemes[0].phoneme = Phoneme.space_phoneme
        if 'sil' in phonemes[-1].phoneme:
            phonemes[-1].phoneme = Phoneme.space_phoneme

        for phoneme in phonemes:
            phoneme.verify()
        return phonemes
