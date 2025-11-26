import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.gen_reverse_regions import RegionSpan, apply_task


def _build_tokens_and_spans():
    tokens = ["(", "C", "A", "B", ")", "[", "Z", "X", "Y", "]"]
    spans = {
        "A": RegionSpan(label="A", start=1, end=4),
        "B": RegionSpan(label="B", start=6, end=9),
    }
    return tokens, spans


def test_apply_task_b_sorts_span_tokens():
    tokens, spans = _build_tokens_and_spans()

    output = apply_task(tokens, spans, "B")

    # Only the B span should change and it should be sorted ascending.
    assert output[:6] == tokens[:6]
    assert output[6:9] == ["X", "Y", "Z"]
    assert output[9:] == tokens[9:]


def test_apply_task_c_composes_reverse_and_sort():
    tokens, spans = _build_tokens_and_spans()

    output = apply_task(tokens, spans, "C")

    # A span is reversed, B span is sorted.
    assert output[:5] == ["(", "B", "A", "C", ")"]
    assert output[6:9] == ["X", "Y", "Z"]
