from typing import List, NamedTuple
from itertools import groupby
import numpy as np
import torch
from collections import defaultdict
from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        unique = np.array([ind for ind, _ in groupby(inds)])
        return super().decode(unique[unique != self.char2ind[self.EMPTY_TOK]])

    def bs_iteration(self, state, frame, beam_size):
        new_state = defaultdict(float)
        for (pref, last), pref_proba in state.items():
            for next_char_id, next_char_proba in enumerate(frame):
                next_char = self.ind2char[next_char_id]
                if next_char != last and next_char != self.EMPTY_TOK:
                    new_state[(pref + next_char, next_char)] += pref_proba + next_char_proba
                else:
                    new_state[(pref, next_char)] += pref_proba + next_char_proba
        new_state = list(new_state.items())
        new_state.sort(reverse=True, key = lambda x: x[1])
        return dict(new_state[:beam_size])

            
    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        states = {('', self.EMPTY_TOK): 1}
        for frame in probs:
            states = self.bs_iteration(states, frame, beam_size)
        states = list(states.items())
        states.sort(reverse=True, key = lambda x: x[1])
        return [Hypothesis(h[0][0], h[1]) for h in states]
