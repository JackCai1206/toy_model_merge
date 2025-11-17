import os
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers.modeling_outputs import CausalLMOutput

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
VENV_SITE = os.path.join(ROOT, ".venv", "lib", "python3.13", "site-packages")
if os.environ.get("USE_PROJECT_VENV_FOR_TESTS") == "1":
    if os.path.isdir(VENV_SITE) and VENV_SITE not in sys.path:
        sys.path.insert(0, VENV_SITE)

from utils.collators import CausalLMDataCollator
from utils.tokenizer import SimpleCharTokenizer
from utils.training import run_iterative_training_loop


class TinyDataset(Dataset):
    def __init__(self, tokenizer: SimpleCharTokenizer, length: int = 4) -> None:
        self.length = length
        ids = tokenizer.encode("ABCD", add_special_tokens=True)
        self.sequence = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        del idx
        input_ids = self.sequence.clone()
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TinyAutoregressiveModel(nn.Module):
    def __init__(self, tokenizer: SimpleCharTokenizer, hidden_size: int = 8) -> None:
        super().__init__()
        vocab_size = tokenizer.vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        del attention_mask
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return CausalLMOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))


class RisingEval:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model) -> dict:
        del model
        self.calls += 1
        score = min(1.0, 0.25 * self.calls)
        return {"eval_exact": score, "eval_acc_min": score}


def test_iterative_training_smoke(tmp_path):
    tokenizer = SimpleCharTokenizer()
    train_dataset = TinyDataset(tokenizer, length=8)
    eval_dataset = TinyDataset(tokenizer, length=2)
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)
    greedy_eval = RisingEval()

    def model_builder():
        return TinyAutoregressiveModel(tokenizer, hidden_size=8)

    output_dir = tmp_path / "smoke"
    trainer, callback, history = run_iterative_training_loop(
        model_builder=model_builder,
        initial_model=model_builder(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        greedy_eval_fn=greedy_eval,
        output_dir=str(output_dir),
        per_device_batch_size=1,
        per_device_eval_batch_size=1,
        grad_accum=1,
        max_steps=10,
        initial_eval_steps=2,
        eval_refine_rounds=2,
        metric_name="eval_acc_min",
        rollback_branches=2,
        success_threshold=0.2,
    )

    assert callback.best_step is not None
    assert history, "round history should record at least one round"
    # Ensure the final trainer saved checkpoints into the requested directory.
    assert (output_dir / "checkpoint-2").exists()
