"""Batch padding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from utils.tokenizer import SimpleCharTokenizer


@dataclass
class CausalLMDataCollator:
    """Pads variable-length fields for causal LM training."""

    tokenizer: SimpleCharTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention = [f["attention_mask"] for f in features]
        batch_labels = [f["labels"] for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=self.label_pad_token_id
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # Include any extra fields untouched (e.g., task_id).
        for key in features[0]:
            if key in batch:
                continue
            batch[key] = torch.stack([f[key] for f in features])
        return batch
