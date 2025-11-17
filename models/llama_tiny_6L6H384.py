"""Factory for a NanoGPT-scale LLaMA-style causal LM."""

from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM


def build_nano_llama(
    vocab_size: int,
    max_position_embeddings: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> LlamaForCausalLM:
    """Instantiate a 6L/6H/384d causal model."""

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_key_value_heads=6,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    return LlamaForCausalLM(config)
