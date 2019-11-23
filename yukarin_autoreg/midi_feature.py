from typing import List, Union, Tuple

import numpy as np
from kiritan_singing_label_reader.midi_note import Note


class MidiFeature(object):
    def __init__(
            self,
            notes: List[Note],
            pitch_range: Tuple[int, int],
            rate: int,
    ) -> None:
        self.notes = notes
        self.pitch_range = pitch_range
        self.rate = rate

    def _to_index(self, t: float):
        return int(round(t * self.rate))

    def _to_time(self, i: Union[int, np.ndarray]):
        return i / self.rate

    @property
    def len_array(self):
        return self._to_index(self.notes[-1].end) + 1

    def _make_note_array(self):
        dim = self.pitch_range[1] - self.pitch_range[0] + 1
        array = np.zeros((len(self.notes), dim), dtype=np.float32)

        for i, note in enumerate(self.notes):
            feature = np.zeros(dim, dtype=bool)
            feature[note.pitch - self.pitch_range[0]] = True
            array[i] = feature

        return array

    def make_array(self, with_position=True):
        note_array = self._make_note_array()

        dim = note_array.shape[1] + 2 if with_position else 0
        array = np.zeros((self.len_array, dim), dtype=np.float32)
        for i, note in enumerate(self.notes):
            s = self._to_index(note.start)
            e = self._to_index(note.end)

            features = [np.repeat(note_array[i].reshape(1, -1), repeats=e - s + 1, axis=0)]

            if with_position:
                pos_start = (self._to_time(np.arange(s, e + 1)) - note.start).reshape(-1, 1)
                pos_end = note.end - note.start - pos_start
                features.append(pos_start)
                features.append(pos_end)

            array[s:e + 1] = np.concatenate(features, axis=1)
        return array
