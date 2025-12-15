import pytest

from arithemtic_scaling_law.run_cot_scaling_experiment import _parse_answer
from algorithm_composition.utils.tokenizer import SimpleCharTokenizer


@pytest.fixture()
def tokenizer() -> SimpleCharTokenizer:
    return SimpleCharTokenizer()


def _ids(tokenizer: SimpleCharTokenizer, text: str) -> list[int]:
    # Mirror eval code: decode generated suffix, no special tokens.
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A42", 42),
        ("fooA007bar", 7),
        ("A-12", 12),  # '-' stops digit scan; falls back to regex.
        ("blah=5", 5),
        ("a=1=b=2", 2),
        ("no markers 123 end", 123),
    ],
)
def test_parse_answer_basic(text: str, expected: int, tokenizer: SimpleCharTokenizer):
    pred = _parse_answer(_ids(tokenizer, text), tokenizer)
    assert pred == expected


def test_parse_answer_prefers_last_A_digits_over_other_numbers(tokenizer: SimpleCharTokenizer):
    # Ensures we don't grab intermediate numbers before the final answer marker.
    text = "step1=3 step2=9 A12"
    pred = _parse_answer(_ids(tokenizer, text), tokenizer)
    assert pred == 12


def test_parse_answer_prefers_equals_when_no_A(tokenizer: SimpleCharTokenizer):
    text = "1+1=2 blah 999"
    pred = _parse_answer(_ids(tokenizer, text), tokenizer)
    assert pred == 2


def test_parse_answer_returns_none_when_no_numbers(tokenizer: SimpleCharTokenizer):
    pred = _parse_answer(_ids(tokenizer, "no digits here"), tokenizer)
    assert pred is None
