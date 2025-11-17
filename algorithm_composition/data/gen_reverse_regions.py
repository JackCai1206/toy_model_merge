"""Synthetic generator for commutative vs. non-commutative reverse tasks."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Sequence, Tuple


# Shared alphabet and bracket markers for every task family.
PAYLOAD_ALPHABET: Tuple[str, ...] = tuple(
    [chr(i) for i in range(ord("A"), ord("Z") + 1)] + list("012345")
)
REGION_MARKERS = {"A": ("(", ")"), "B": ("[", "]")}


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameterization shared across both task families."""

    seq_len_range: Tuple[int, int] = (48, 96)
    min_inner_length: int = 2
    max_attempts: int = 100


@dataclass
class RegionSpan:
    label: str
    start: int  # inclusive
    end: int  # exclusive


@dataclass
class RegionSample:
    family: str
    tokens: List[str]
    spans: Dict[str, RegionSpan]


def _random_payload(rng: random.Random, length: int) -> List[str]:
    return [rng.choice(PAYLOAD_ALPHABET) for _ in range(length)]


def _sample_interval(length: int, rng: random.Random, min_inner: int) -> Tuple[int, int]:
    """Sample (start, end) positions such that substring length ≥ ``min_inner``."""

    if length < min_inner + 1:
        raise ValueError("Payload too short for requested interval length.")
    max_start = length - (min_inner + 1)
    if max_start < 0:
        raise ValueError("No valid start indices under current constraints.")
    start = rng.randint(0, max_start)
    end_min = start + min_inner
    end = rng.randint(end_min, length - 1)
    return start, end


def _classify_topology(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> str:
    """Classify whether the two spans are disjoint, crossing, nested, etc."""

    a_start, a_end = span_a
    b_start, b_end = span_b
    if a_end < b_start or b_end < a_start:
        return "disjoint"
    if (a_start < b_start < a_end < b_end) or (b_start < a_start < b_end < a_end):
        return "crossing"
    if (a_start < b_start and b_end <= a_end) or (b_start < a_start and a_end <= b_end):
        return "nested"
    return "other"


def _render_with_brackets(
    payload: List[str],
    span_a: Tuple[int, int],
    span_b: Tuple[int, int],
) -> Tuple[List[str], Dict[str, RegionSpan]]:
    """Insert bracket markers for span A and B and return tokens/spans."""

    tokens = list(payload)
    inserts: List[Tuple[int, str]] = []
    for label, (start, end) in (("A", span_a), ("B", span_b)):
        open_tok, close_tok = REGION_MARKERS[label]
        inserts.append((end, close_tok))
        inserts.append((start, open_tok))
    for idx, tok in sorted(inserts, key=lambda item: item[0], reverse=True):
        tokens.insert(idx, tok)
    spans = _extract_spans(tokens)
    return tokens, spans


def _extract_spans(tokens: Sequence[str]) -> Dict[str, RegionSpan]:
    """Locate the interior token ranges for each labeled region."""

    starts: Dict[str, int] = {}
    spans: Dict[str, RegionSpan] = {}
    for idx, tok in enumerate(tokens):
        for label, (open_tok, close_tok) in REGION_MARKERS.items():
            if tok == open_tok:
                starts[label] = idx + 1
            elif tok == close_tok:
                start = starts.get(label)
                if start is None:
                    raise ValueError(f"Closing token '{close_tok}' appeared before open '{open_tok}'.")
                spans[label] = RegionSpan(label=label, start=start, end=idx)
    missing = set(REGION_MARKERS) - set(spans)
    if missing:
        raise ValueError(f"Failed to locate spans for regions: {sorted(missing)}")
    return spans


def _build_sample(
    rng: random.Random,
    family: str,
    config: GeneratorConfig,
    allowed_topologies: Sequence[str] | None,
) -> RegionSample:
    min_seq, max_seq = config.seq_len_range
    for _ in range(config.max_attempts):
        total_len = rng.randint(min_seq, max_seq)
        payload = _random_payload(rng, total_len)
        span_a = _sample_interval(total_len, rng, config.min_inner_length)
        span_b = _sample_interval(total_len, rng, config.min_inner_length)
        topology = _classify_topology(span_a, span_b)
        if allowed_topologies is not None and topology not in allowed_topologies:
            continue
        tokens, spans = _render_with_brackets(payload, span_a, span_b)
        family_label = family if family in {"C", "NC"} else "AB"
        return RegionSample(family=family_label, tokens=tokens, spans=spans)
    raise RuntimeError(
        f"Failed to sample topology '{family}' under the provided config (allowed={allowed_topologies})."
    )


def generate_sample(
    rng: random.Random, family: str, config: GeneratorConfig | None = None
) -> RegionSample:
    """Sample a single sequence for the requested family."""

    cfg = config or GeneratorConfig()
    family = family.upper()
    allowed_map = {
        "C": ("disjoint",),
        "NC": ("crossing",),
        "GENERAL": ("disjoint", "crossing"),
    }
    allowed = allowed_map.get(family)
    if allowed is None:
        raise ValueError(f"Unknown family '{family}'. Expected one of: {sorted(allowed_map)}.")
    return _build_sample(rng, family, cfg, allowed)


def reverse_inside(tokens: Sequence[str], span: RegionSpan) -> List[str]:
    """Apply ReverseInside operator within the provided span."""

    updated = list(tokens)
    updated[span.start : span.end] = reversed(updated[span.start : span.end])
    return updated


def apply_task(tokens: Sequence[str], spans: Dict[str, RegionSpan], task: str) -> List[str]:
    """Apply task operator(s) returning a fresh token list."""

    task = task.upper()
    order_map = {
        "A": ("A",),
        "B": ("B",),
        "C": ("A", "B"),
    }
    order = order_map.get(task)
    if order is None:
        raise ValueError(f"Unsupported task '{task}'.")

    current = list(tokens)
    for label in order:
        span = spans[label]
        current = reverse_inside(current, span)
    return current


def tokens_to_text(tokens: Sequence[str]) -> str:
    """Render tokens as a contiguous character string."""

    return "".join(tokens)


def check_commutativity_once(sample: RegionSample) -> Tuple[bool, str, str]:
    """Return (is_commutative, A∘B text, B∘A text) for a sample."""

    a_then_b = apply_task(sample.tokens, sample.spans, "C")
    b_then_a = reverse_inside(reverse_inside(sample.tokens, sample.spans["B"]), sample.spans["A"])
    text_ab = tokens_to_text(a_then_b)
    text_ba = tokens_to_text(b_then_a)
    return a_then_b == b_then_a, text_ab, text_ba


def sample_and_check(
    family: str, trials: int = 200, rng_seed: int = 0, config: GeneratorConfig | None = None
) -> float:
    """Empirically estimate the commutativity rate for a family."""

    rng = random.Random(rng_seed)
    cfg = config or GeneratorConfig()
    matches = 0
    for _ in range(trials):
        sample = generate_sample(rng, family, cfg)
        equal, _, _ = check_commutativity_once(sample)
        matches += int(equal)
    return matches / trials
