# Algorithm Composition 2

Self-contained version of the composition experiments that operates on whole strings instead of bracketed regions.

- **Tasks**:  
  - `A`: reverse the entire string.  
  - `B`: rotate the entire string left by `k` characters (default `k=3`).  
  - `C`: apply `A` then `B` to the full string.
- **Data**: `data/simple_tasks.py` samples random alphanumeric payloads and applies the task operators to the whole string.
- **Training utilities**: `utils/sample_complexity.py` copies the sample-complexity loop from the arithmetic-scaling code so the folder has no external imports.

## Scripts

- `train_atomic_task.py`: train task `A` or `B` from scratch (writes to `artifacts/atomic`).
- `train_joint_ab.py`: joint pretraining on tasks `A` and `B`.
- `finetune_composed.py`: fine-tune the composed task `C` from any checkpoint (joint or merged).
- `compare_sample_complexity.py`: merge separately trained `A`/`B` models and fine-tune `C` from both the joint and merged checkpoints to compare sample complexity.

## Quickstart

```bash
# Train atomic tasks (seed 0 shown as an example)
python -u train_atomic_task.py --task A --seed 0
python -u train_atomic_task.py --task B --seed 0

# Train the joint A/B model
python -u train_joint_ab.py --seed 0

# Compare fine-tuning sample complexity for C
python -u compare_sample_complexity.py --seed 0
```

All paths default to the `algorithm_composition_2` subtree to keep the experiments isolated from neighboring projects.
