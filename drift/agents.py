"""Speaker and listener, behind one `Team` object with two modes.

lora: one frozen base LLM with a LoRA adapter per agent. The reference policy for
      the KL term is the same base with the adapter disabled (no extra copy).
full: a separate full copy per agent, all weights trained, plus a frozen copy as the
      reference. Used as the control for "does LoRA limit drift?" (E7).

Every forward pass goes through `team.ctx(role)`, which activates the right weights and
mixed precision. role=None means the reference/base model.
On a V100 (no bf16): weights fp32 + fp16 autocast (cfg dtype=float32, amp=fp16).
"""

from __future__ import annotations

import contextlib
import copy

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .game import LETTERS, Game, listener_prompt, speaker_prompt

SPEAKER, LISTENER = "speaker", "listener"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def chat(tok, user_text: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)


def lora_config(cfg) -> LoraConfig:
    return LoraConfig(r=cfg["lora_rank"], lora_alpha=2 * cfg["lora_rank"], lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    return tok


def load_causal_lm(name: str, cfg, device):
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPES[cfg["dtype"]]).to(device)


class Team:
    def __init__(self, cfg, device=None, ckpt: str | None = None, listener_ckpt: str | None = None):
        """ckpt: a run's saved checkpoint dir (runs/<run>/final) to load both agents from.
        listener_ckpt: take the listener from another checkpoint instead (cross-seed pairing)."""
        self.cfg = cfg
        self.mode = cfg["finetune"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp_dtype = DTYPES.get(cfg["amp"])  # None = no autocast
        self.tok = load_tokenizer(cfg["model"])
        base = load_causal_lm(cfg["model"], cfg, self.device)
        self.listeners = [LISTENER]
        src = {SPEAKER: ckpt, LISTENER: listener_ckpt or ckpt}
        if self.mode == "lora":
            if ckpt:
                self.model = PeftModel.from_pretrained(base, f"{src[SPEAKER]}/{SPEAKER}",
                                                       adapter_name=SPEAKER, is_trainable=True)
                self.model.load_adapter(f"{src[LISTENER]}/{LISTENER}", adapter_name=LISTENER,
                                        is_trainable=True)
            else:
                self.model = get_peft_model(base, lora_config(cfg), adapter_name=SPEAKER)
                self.model.add_adapter(LISTENER, lora_config(cfg))
        elif self.mode == "full":
            base.requires_grad_(False)
            self.ref = base.eval()
            self.models = {r: (load_causal_lm(f"{src[r]}/{r}", cfg, self.device) if ckpt
                               else copy.deepcopy(base)).requires_grad_(True)
                           for r in (SPEAKER, LISTENER)}
        else:
            raise ValueError(f"finetune must be lora or full, got {self.mode}")

    # ---- weights / modes

    def amp(self):
        if self.amp_dtype is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    @contextlib.contextmanager
    def ctx(self, role: str | None):
        """Yields the module to call for `role` (None = frozen reference), under autocast."""
        with self.amp():
            if self.mode == "full":
                yield self.ref if role is None else self.models[role]
            elif role is None:
                with self.model.disable_adapter():
                    yield self.model
            else:
                self.model.set_adapter(role)
                # PEFT's set_adapter freezes the inactive adapters, which would silently
                # stop their updates, so re-enable grads on every LoRA parameter
                for n, p in self.model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad_(True)
                yield self.model

    def params(self, role: str):
        if self.mode == "full":
            return list(self.models[role].parameters())
        return [p for n, p in self.model.named_parameters() if f".{role}." in n and "lora_" in n]

    def add_listener(self, name: str):
        """E4 partner pool. LoRA only: extra full copies would not fit in memory."""
        assert self.mode == "lora", "listener pools need finetune=lora"
        self.model.add_adapter(name, lora_config(self.cfg))
        self.listeners.append(name)

    def reset(self, role: str):
        """Put an agent back to the base model (E4 listener reset)."""
        if self.mode == "full":
            self.models[role].load_state_dict(self.ref.state_dict())
            return
        for mod in self.model.modules():
            if hasattr(mod, "reset_lora_parameters") and role in getattr(mod, "lora_A", {}):
                mod.reset_lora_parameters(role, True)  # lora_B = 0: adapter is a no-op again

    def train(self, on: bool = True):
        for m in ([self.model] if self.mode == "lora" else self.models.values()):
            m.train(on)

    def save(self, path: str):
        if self.mode == "lora":
            self.model.save_pretrained(path)  # every adapter, ~35 MB each
        else:
            for r, m in self.models.items():
                m.save_pretrained(f"{path}/{r}")


def make_optimizer(params, lr: float, cfg):
    if cfg["optim"] == "adamw8bit":  # halves optimizer memory; needed only for full FT if tight
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


# ---------------------------------------------------------------- speaker

@torch.no_grad()
def speak(team: Team, games: list[Game], max_tokens: int, n_samples: int = 1,
          temperature: float = 1.0, greedy: bool = False, role: str | None = SPEAKER):
    """Sample `n_samples` messages per game. Returns (prompt_ids, message_ids, texts),
    each a list of length len(games) * n_samples, grouped game-major.
    role=None uses the base model (zero-shot / reference speaker)."""
    tok = team.tok
    prompts = [chat(tok, speaker_prompt(g, max_tokens)) for g in games]
    enc = tok(prompts, return_tensors="pt", padding=True).to(team.device)
    # override Qwen's generation_config defaults (top_k=20, top_p=0.8, repetition_penalty)
    # so the sampling distribution is exactly the policy used in the log-probs
    kwargs = dict(max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id,
                  num_return_sequences=n_samples, repetition_penalty=1.0)
    if greedy:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    with team.ctx(role) as m:
        out = m.generate(**enc, **kwargs)
    gen = out[:, enc["input_ids"].shape[1]:]
    prompt_ids, msg_ids, texts = [], [], []
    for i in range(gen.shape[0]):
        p = enc["input_ids"][i // n_samples]
        p = p[enc["attention_mask"][i // n_samples].bool()]
        msg = gen[i]
        # cut at first eos/pad, keep the eos so the model learns to stop
        stop = ((msg == tok.eos_token_id) | (msg == tok.pad_token_id)).nonzero()
        if len(stop):
            msg = msg[: stop[0].item() + 1]
        prompt_ids.append(p)
        msg_ids.append(msg)
        texts.append(tok.decode(msg, skip_special_tokens=True))
    return prompt_ids, msg_ids, texts


def sequence_logprobs(team: Team, prompt_ids, msg_ids, role: str | None, micro_bs: int = 64):
    """Per-token log-probs of each message under the speaker (role) or reference (None).
    Returns a list of 1-D fp32 tensors (gradient flows when role is set and grad is enabled).
    With grad on, call it on one micro-batch at a time and backward before the next one:
    full-vocabulary logits for 256 messages do not fit in memory at once."""
    pad = team.tok.pad_token_id
    out = []
    for s in range(0, len(prompt_ids), micro_bs):
        P, M = prompt_ids[s:s + micro_bs], msg_ids[s:s + micro_bs]
        seqs = [torch.cat([p, m]) for p, m in zip(P, M)]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), pad, device=team.device)
        att = torch.zeros_like(ids)
        for i, x in enumerate(seqs):  # left pad
            ids[i, L - len(x):] = x
            att[i, L - len(x):] = 1
        # left padding puts every message at the end, so only the last K positions are needed
        # (vocab is 151k: full-sequence logits would dominate memory)
        K = max(len(m) for m in M) + 1
        with team.ctx(role) as m:
            logits = m(input_ids=ids, attention_mask=att, logits_to_keep=K).logits
        logits = logits[:, :-1].float()
        tgt = ids[:, L - K + 1:, None]
        logp = logits.gather(-1, tgt)[..., 0] - logits.logsumexp(-1)
        for i, msg in enumerate(M):
            out.append(logp[i, K - 1 - len(msg):])  # positions predicting the message tokens
    return out


# ---------------------------------------------------------------- listener

def letter_ids(tok, n: int) -> list[int]:
    ids = [tok.convert_tokens_to_ids(c) for c in LETTERS[:n]]
    assert all(i is not None and i != tok.unk_token_id for i in ids), "letters must be single tokens"
    return ids


def listen_logits(team: Team, games: list[Game], messages: list[str], role: str | None = LISTENER,
                  micro_bs: int = 64, reader=None):
    """fp32 logits over the candidate letters, shape (len(games), D+1).
    `reader=(model, tok)` evaluates an outside frozen reader instead of a team member."""
    tok = reader[1] if reader is not None else team.tok
    lids = letter_ids(tok, len(games[0].candidates))
    rows = []
    for s in range(0, len(games), micro_bs):
        prompts = [chat(tok, listener_prompt(g, m))
                   for g, m in zip(games[s:s + micro_bs], messages[s:s + micro_bs])]
        enc = tok(prompts, return_tensors="pt", padding=True).to(team.device)
        if reader is not None:
            with team.amp():
                logits = reader[0](**enc, logits_to_keep=1).logits[:, -1]
        else:
            with team.ctx(role) as m:
                logits = m(**enc, logits_to_keep=1).logits[:, -1]
        rows.append(logits[:, lids].float())
    return torch.cat(rows)


def load_reader(name: str, cfg, device=None):
    """Outside frozen reader (e.g. a larger Qwen) that never trains with the speaker."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return load_causal_lm(name, cfg, device).eval(), load_tokenizer(name)
