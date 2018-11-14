import numpy as np
from pathlib import Path


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
        assert phoneme in self.phoneme_list, f'{phoneme} is not defined.'

        self.phoneme = phoneme
        self.start = np.round(start, decimals=2)
        self.end = np.round(end, decimals=2)

    def __repr__(self):
        return f'Phoneme(phoneme=\'{self.phoneme}\', start={self.start}, end={self.end})'

    @property
    def phoneme_id(self):
        return self.phoneme_list.index(self.phoneme)

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
        return [
            Phoneme.parse(s)
            for s in path.read_text().split('\n')
            if len(s) > 0
        ]
