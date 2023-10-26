from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        for log_probs_mat, length, target_text in zip(log_probs, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred_text = self.text_encoder.ctc_beam_search(log_probs_mat[:length], length, beam_size=self.beam_size)[0].text
            else:
                pred_text = ''
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
