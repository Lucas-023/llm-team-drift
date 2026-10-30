"""Shared fixtures. The tiny model has random weights but the real Qwen2.5 tokenizer and chat
template, so every code path (prompts, letter tokens, LoRA/full, checkpoints) is exercised
without downloading the 1 GB model."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:  # offline and not cached
        pytest.skip(f"cannot get the Qwen tokenizer: {e}")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = [cfg.eos_token_id, tok.pad_token_id]
    path = tmp_path_factory.mktemp("tiny-qwen")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture
def make_cfg(tiny_model, tmp_path):
    from drift.utils import load_config

    def _make(*overrides):
        return load_config(str(ROOT / "configs/base.yaml"),
                           [f"model={tiny_model}", f"out_dir={tmp_path}", "batch_games=4",
                            "group_size=4", "micro_bs=8", "n_eval_games=16", *overrides])
    return _make
