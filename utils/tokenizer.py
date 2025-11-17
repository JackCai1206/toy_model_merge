"""Lightweight character-level tokenizer compatible with Hugging Face Trainer."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from transformers import PreTrainedTokenizer


BASE_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ADDITIONAL_CHARS = ["[", "]", "/", " ", ":", "|", "-", "_", ".", ",", "<", ">", "="]


class SimpleCharTokenizer(PreTrainedTokenizer):
    """Minimal character-level tokenizer tailored to the reverse-region tasks."""

    def __init__(self) -> None:
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<sep>"]
        self._all_tokens = special_tokens + BASE_CHARS + ADDITIONAL_CHARS
        self._token_to_id = {token: idx for idx, token in enumerate(self._all_tokens)}
        self._id_to_token = {idx: tok for tok, idx in self._token_to_id.items()}

        super().__init__(
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            sep_token="<sep>",
        )

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._token_to_id)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        if token_ids_1 is not None:
            return [self.bos_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [
                self.eos_token_id
            ]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        if token_ids_1 is not None:
            return (
                [1]
                + [0] * len(token_ids_0)
                + [1]
                + [0] * len(token_ids_1)
                + [1]
            )
        return [1] + [0] * len(token_ids_0) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        if token_ids_1 is not None:
            return [0] * (len(token_ids_0) + len(token_ids_1) + 3)
        return [0] * (len(token_ids_0) + 2)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> Tuple[str]:
        import os

        os.makedirs(save_directory, exist_ok=True)
        filename = "vocab.txt"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        vocab_path = os.path.join(save_directory, filename)
        with open(vocab_path, "w", encoding="utf-8") as handle:
            for idx in range(len(self._id_to_token)):
                handle.write(self._id_to_token[idx] + "\n")
        return (vocab_path,)


def build_tokenizer() -> SimpleCharTokenizer:
    """Helper used by scripts to obtain a fresh tokenizer instance."""

    return SimpleCharTokenizer()


def encode_prompt_with_sep(tokenizer: SimpleCharTokenizer, prompt: str) -> List[int]:
    """Encode prompt prefix followed by the separator token."""

    ids = [tokenizer.bos_token_id]
    if prompt:
        ids.extend(tokenizer.encode(prompt, add_special_tokens=False))
    ids.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
    return ids
