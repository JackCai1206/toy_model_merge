import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.datasets import build_mixed_task_schedule


def test_build_mixed_task_schedule_balances_atomic_tasks():
    schedule = build_mixed_task_schedule(
        dataset_size=20,
        primary_task="C",
        auxiliary_tasks=("A", "B"),
        auxiliary_fraction=0.5,
        seed=123,
    )

    assert len(schedule) == 20
    counter = collections.Counter(schedule)
    assert counter["C"] == 10
    assert abs(counter["A"] - counter["B"]) <= 1
    assert set(schedule) <= {"A", "B", "C"}


def test_build_mixed_task_schedule_is_deterministic():
    args = dict(
        dataset_size=15,
        primary_task="C",
        auxiliary_tasks=("A", "B"),
        auxiliary_fraction=0.3,
        seed=99,
    )
    schedule_first = build_mixed_task_schedule(**args)
    schedule_second = build_mixed_task_schedule(**args)
    assert schedule_first == schedule_second


def test_build_mixed_task_schedule_clamps_fraction():
    schedule = build_mixed_task_schedule(
        dataset_size=6,
        primary_task="C",
        auxiliary_tasks=("A", "B"),
        auxiliary_fraction=1.5,
        seed=7,
    )
    assert schedule.count("C") == 0
    assert schedule.count("A") + schedule.count("B") == 6
