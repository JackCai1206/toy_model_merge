"""Data generator for fully-bracketed arithmetic with collapsed CoT supervision.

Implements:
* Random full binary expression trees with ops in {+, -, *}.
* Evaluation modulo 100 with post-order (atomic) chain-of-thought.
* Stochastic collapsing of consecutive steps with an upper cap per block.
* JSONL writer for downstream scaling-law experiments.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


MODULUS = 100


@dataclass
class Node:
    op: str | None = None  # "+", "-", "*", or None for leaves
    left: "Node | None" = None
    right: "Node | None" = None
    val: int | None = None  # raw integer value for leaves
    expr: str | None = None  # string representation of the subexpression
    value: int | None = None  # evaluated value modulo MODULUS


def generate_random_tree(num_ops: int, rng: random.Random, max_val: int = 99, easy_mul: bool = True) -> Node:
    """Generate a random full binary tree with the requested number of internal ops.

    If `easy_mul` is True, only allow multiplication nodes when num_ops == 1 and force
    their children to be simple leaves sampled as 2-digit x 1-digit to avoid slow
    resampling downstream.
    """

    if num_ops < 0:
        raise ValueError("num_ops must be non-negative")
    if num_ops == 0:
        return Node(val=rng.randint(0, max_val))

    # Pick an operator with constraints if easy_mul is enabled.
    if easy_mul and num_ops > 1:
        op_choices = ["+", "-"]
    else:
        op_choices = ["+", "-", "*"]
    op = rng.choice(op_choices)

    if easy_mul and op == "*":
        # Force a simple 2-digit x 1-digit multiplication.
        if rng.random() < 0.5:
            left = Node(val=rng.randint(10, min(99, max_val)))
            right = Node(val=rng.randint(0, min(9, max_val)))
        else:
            left = Node(val=rng.randint(0, min(9, max_val)))
            right = Node(val=rng.randint(10, min(99, max_val)))
        return Node(op=op, left=left, right=right)

    left_ops = rng.randint(0, num_ops - 1)
    right_ops = num_ops - 1 - left_ops
    left = generate_random_tree(left_ops, rng, max_val, easy_mul=easy_mul)
    right = generate_random_tree(right_ops, rng, max_val, easy_mul=easy_mul)
    return Node(op=op, left=left, right=right)


def eval_and_build_atomic_steps(node: Node, steps: List[Dict[str, Any]]) -> Tuple[int, str]:
    """Evaluate the tree modulo MODULUS and append atomic steps in post-order."""

    if node.op is None:
        if node.val is None:
            raise ValueError("Leaf node missing value.")
        node.value = node.val % MODULUS
        node.expr = str(node.val)
        return node.value, node.expr

    if node.left is None or node.right is None:
        raise ValueError("Internal node missing children.")

    lv, le = eval_and_build_atomic_steps(node.left, steps)
    rv, re = eval_and_build_atomic_steps(node.right, steps)

    if node.op == "+":
        val = (lv + rv) % MODULUS
    elif node.op == "-":
        val = (lv - rv) % MODULUS
    elif node.op == "*":
        val = (lv * rv) % MODULUS
    else:
        raise ValueError(f"Unsupported operator: {node.op}")

    expr = f"({le} {node.op} {re})"
    node.value = val
    node.expr = expr
    steps.append(
        {
            "expr": expr,
            "value": val,
            "op": node.op,
            "left_value": lv,
            "right_value": rv,
            "node": node,
        }
    )
    return val, expr


def collapse_steps_with_cap(
    atomic_steps: Sequence[Dict[str, Any]],
    rng: random.Random,
    q_keep: float = 0.6,
    max_steps_per_block: int = 3,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    """Collapse consecutive atomic steps into capped blocks and render visible CoT."""

    if not 0.0 < q_keep <= 1.0:
        raise ValueError("q_keep must be in (0, 1].")
    if max_steps_per_block < 1:
        raise ValueError("max_steps_per_block must be >= 1.")

    if not atomic_steps:
        return [], []

    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for step in atomic_steps:
        start_new = False
        if not current:
            start_new = True
        elif len(current) >= max_steps_per_block:
            start_new = True
        else:
            if rng.random() < q_keep:
                start_new = True

        if start_new:
            if current:
                blocks.append(current)
            current = [step]
        else:
            current.append(step)

    if current:
        blocks.append(current)

    root_node: Node | None = atomic_steps[-1].get("node") if atomic_steps else None
    if root_node is None:
        return [], blocks

    collapsed_ids: set[int] = set()
    states: List[str] = [render_expression_with_collapses(root_node, collapsed_ids)]
    for block in blocks:
        for step in block:
            node = step.get("node")
            if isinstance(node, Node):
                collapsed_ids.add(id(node))
        states.append(render_expression_with_collapses(root_node, collapsed_ids))

    return states, blocks


def render_expression_with_collapses(node: Node | None, collapsed_ids: set[int]) -> str:
    """Render the full expression after collapsing the nodes whose ids are in `collapsed_ids`."""

    if node is None:
        return ""

    node_id = id(node)
    if node_id in collapsed_ids and node.value is not None:
        return str(int(node.value) % MODULUS)

    if node.op is None:
        if node.expr is not None:
            return node.expr.replace(" ", "")
        if node.val is not None:
            return str(node.val)
        return "0"

    left_rendered = render_expression_with_collapses(node.left, collapsed_ids)
    right_rendered = render_expression_with_collapses(node.right, collapsed_ids)
    return f"({left_rendered}{node.op}{right_rendered})"


def build_atomic_cot(atomic_steps: Sequence[Dict[str, Any]]) -> List[str]:
    return [f"{str(s['expr']).replace(' ', '')}={int(s['value']) % MODULUS}" for s in atomic_steps]


def evaluate_expr_mod(expr: str, modulus: int = MODULUS) -> int:
    """Safely evaluate a bracketed arithmetic expression modulo `modulus`."""

    node = ast.parse(expr, mode="eval")

    def visit(n) -> int:
        if isinstance(n, ast.Expression):
            return visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return int(n.value) % modulus
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return (-visit(n.operand)) % modulus
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
            left = visit(n.left)
            right = visit(n.right)
            if isinstance(n.op, ast.Add):
                return (left + right) % modulus
            if isinstance(n.op, ast.Sub):
                return (left - right) % modulus
            return (left * right) % modulus
        raise ValueError(f"Unsupported expression node: {ast.dump(n)}")

    return visit(node)


def parse_visible_step(step: str) -> Tuple[str, int]:
    """Extract (expr, value) from a visible CoT sentence."""

    body = step.strip().rstrip(".")
    if "=" not in body:
        raise ValueError(f"Step missing '=': {step}")
    expr_part, value_part = body.rsplit("=", maxsplit=1)
    expr = expr_part.strip()
    try:
        value = int(value_part.strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse value in step: {step}") from exc
    return expr, value


def sanity_check_example(example: Dict, max_steps_per_block: int, modulus: int = MODULUS) -> None:
    """Run basic consistency checks on a generated example."""

    expr = example["expression"]
    answer = example["answer"]
    atomic_cot = example["atomic_cot"]
    visible_cot = example["visible_cot"]
    block_sizes = example.get("block_sizes", [])

    if evaluate_expr_mod(expr, modulus) != answer:
        raise ValueError("Expression evaluation mismatch.")

    # Verify each visible state still evaluates to the final answer.
    if not visible_cot:
        raise ValueError("Visible CoT is empty.")
    for state in visible_cot:
        if evaluate_expr_mod(state, modulus) != answer:
            raise ValueError(f"Visible state mismatch: {state}")

    # Verify block size cap (if provided).
    if block_sizes:
        for size in block_sizes:
            if size < 1 or size > max_steps_per_block:
                raise ValueError(f"Block size {size} violates cap {max_steps_per_block}.")

    # Atomic steps length should align with complexity.
    if example["num_atomic_steps"] != example["complexity_k"]:
        raise ValueError("Atomic step count should equal complexity k.")

    # Atomic CoT lines should evaluate correctly.
    for line in atomic_cot:
        sub_expr, sub_val = parse_visible_step(line)
        if evaluate_expr_mod(sub_expr, modulus) != sub_val:
            raise ValueError(f"Atomic CoT mismatch: {line}")


def generate_example(
    k: int,
    rng: random.Random,
    q_keep: float,
    max_steps_per_block: int,
    max_val: int = 99,
    easy_multiplication: bool = True,
) -> Dict:
    """Generate a single example dictionary following the spec."""

    tree = generate_random_tree(k, rng, max_val=max_val, easy_mul=easy_multiplication)
    atomic_steps: List[Dict[str, int | str]] = []
    value, expr = eval_and_build_atomic_steps(tree, atomic_steps)

    visible_cot, blocks = collapse_steps_with_cap(
        atomic_steps=atomic_steps,
        rng=rng,
        q_keep=q_keep,
        max_steps_per_block=max_steps_per_block,
    )

    atomic_cot = build_atomic_cot(atomic_steps)
    block_sizes = [len(block) for block in blocks]

    return {
        "expression": expr,
        "answer": value,
        "complexity_k": k,
        "num_atomic_steps": len(atomic_steps),
        "num_visible_steps": len(visible_cot),
        "visible_cot": visible_cot,
        "atomic_cot": atomic_cot,
        "block_sizes": block_sizes,
        "q_keep": q_keep,
        "max_steps_per_block": max_steps_per_block,
        "max_val": max_val,
    }


def generate_dataset(
    *,
    k_min: int,
    k_max: int,
    examples_per_k: int,
    q_keep: float,
    max_steps_per_block: int,
    seed: int,
    output_path: str,
    max_val: int = 99,
    easy_multiplication: bool = True,
    run_sanity_checks: bool = True,
    progress_every: int = 1000,
) -> None:
    """Write a JSONL dataset to `output_path`."""

    rng = random.Random(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for k in range(k_min, k_max + 1):
            for _ in range(examples_per_k):
                example = generate_example(
                    k=k,
                    rng=rng,
                    q_keep=q_keep,
                    max_steps_per_block=max_steps_per_block,
                    max_val=max_val,
                    easy_multiplication=easy_multiplication,
                )
                if run_sanity_checks:
                    sanity_check_example(example, max_steps_per_block)
                handle.write(json.dumps(example) + "\n")
                total += 1
                if progress_every and total % progress_every == 0:
                    print(f"Wrote {total} examples (latest k={k}).")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate bracketed arithmetic CoT data (mod 100).")
    parser.add_argument("--k_min", type=int, default=1, help="Minimum complexity k.")
    parser.add_argument("--k_max", type=int, default=6, help="Maximum complexity k.")
    parser.add_argument("--examples_per_k", type=int, default=1000, help="Examples per complexity level.")
    parser.add_argument("--q_keep", type=float, default=0.6, help="Probability of starting a new CoT block.")
    parser.add_argument(
        "--max_steps_per_block",
        type=int,
        default=3,
        help="Maximum atomic steps collapsed into a single visible substitution chain.",
    )
    parser.add_argument("--max_val", type=int, default=99, help="Maximum leaf integer value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default="arithmetic_cot_mod100.jsonl", help="Output JSONL path.")
    parser.add_argument("--skip_sanity_checks", action="store_true", help="Disable per-example sanity checks.")
    parser.add_argument("--progress_every", type=int, default=1000, help="Print progress every N examples (0 to disable).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    generate_dataset(
        k_min=args.k_min,
        k_max=args.k_max,
        examples_per_k=args.examples_per_k,
        q_keep=args.q_keep,
    max_steps_per_block=args.max_steps_per_block,
        seed=args.seed,
        output_path=args.output,
        max_val=args.max_val,
        run_sanity_checks=not args.skip_sanity_checks,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
