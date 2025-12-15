---
applyTo: '**'
---
Style reminder for this repo/session: Keep changes minimal and direct. Don’t add helper functions, env-var parsing, new CLI flags, or “best-effort” optional logic unless I explicitly request configurability. Prefer setting existing framework knobs in-place (e.g., TrainingArguments / Trainer settings) with the smallest diff across the fewest files. After editing, do a quick import/smoke check and report only: what changed + verification result.

Try to find the .venv at the root of the repo and use it whenever you need to run python code.
