# Novelty check — 2026-10-29

Question: has anyone already RL-trained two pretrained LLMs (speaker + listener) on a
communication game and measured whether their messages drift away from readable language?

**Verdict: not found.** Every nearby work is missing at least one ingredient:
(a) pretrained LLMs, (b) both agents updated by RL, (c) drift/readability measured with an
outside reader.

| Work | (a) LLMs | (b) RL co-training | (c) drift measured | How it differs |
|---|---|---|---|---|
| Sullivan & Koller, *On Language Drift during RLVR Post-Training* (arXiv 2610.02015, Oct 2026) | yes | single model | yes (legibility) | Closest; single-agent reasoning. Gives our hypothesis: drift appears when the task can't be drawn out of the base model. **Must cite.** |
| Anand et al., *Communication between Frozen LLMs via Prompt Optimization in a Referential Game* (arXiv 2609.31989, Sep 2026) | yes | no: weights frozen, prompts optimized | no | Same game, but no weight updates. A good contrast: "prompt-level vs weight-level protocols". |
| Talebirad et al., *From Signals to Structure: Memory Architecture and Language Emergence in LLM Agents* (arXiv 2607.00233, Jun 2026) | yes | no: in-context and memory | conventions, not drift | In-context Lewis game. |
| Wright & Suhr, *Learning to Refer from Estimated Listener Gaze* (arXiv 2609.14207, COLM 2026) | VLM speaker | speaker only; fixed listener | reports that without KL messages collapse to 1–2 tokens | Useful evidence for our E2 (KL) prediction. |
| Clark, Martin & Mahowald, *Why is AI-generated language so hard to understand?* (arXiv 2610.29162, Oct 2026) | yes | no (length-penalty fine-tune) | psycholinguistic difficulty | Supports the motivation: optimization pressure makes text less human-like. |
| LoGIC (arXiv 2507.08610) | VLM | yes, GRPO speaker+listener | no | Uses the game to improve captioning, not to study drift. |
| MAGRPO, *LLM Collaboration with MARL* (arXiv 2508.04652) | yes | yes | no | Writing/coding cooperation; doesn't look at message readability. |
| LangGround (NeurIPS 2024, arXiv 2409.17348) | uses LLM data | small MARL nets | interpretability | Grounds small agents' messages in LLM text; the reverse direction. |
| Lazaridou et al. 2020 (2005.07064); Lee et al. 2019 (1909.04499); Lu et al. 2020, Seeded Iterated Learning (2003.12694) | small LSTMs / seq2seq | yes | yes | The classic drift results, before LLMs. We test whether they still hold at LLM scale. |
| Motwani et al. 2024 (2402.07510); Rippin et al. 2026 (2606.28425); Codetta (2609.28900) | yes | no | covert channels | Steganography by prompting/tool use. Our private-code gap is the RL-emergent version of this. |

## What this means for the paper
- Frame the contribution as *the first controlled measurement of drift between RL-co-trained
  pretrained LLMs*, testing Sullivan & Koller's single-agent finding in the multi-agent case.
- Anand et al. (frozen + prompt optimization) and our setup (weights trained) are natural
  contrasts; mention it in related work.
- Risk: the field moves fast (3 of the papers above are from the last 2 months). Re-run this
  search before submitting (search terms: "language drift" LLM multi-agent RL; referential
  game GRPO; emergent communication LLM fine-tuning).
