# Qwen3-Coder-Next: how much context to serve

Measured on 2026-10-21 on one H200 (RunPod secure), `UD-Q4_K_XL`, running
`tang-llm serve --ctx 131072 --batch 8`. The node image's `CTX` env var sets `--ctx`.

| check | result |
|---|---|
| `/v1/models` `max_model_len` | 131072 (it reports `--ctx`) |
| VRAM with all 8 batch slots allocated | 76.3 GB of 143.8 GB |
| needle retrieval at 53.9k tokens | correct |
| needle retrieval at 108.2k tokens | correct |
| cold prefill at 53.9k | 121 s (445 tok/s) |
| cold prefill at 108.2k | 375 s (288 tok/s) |
| decode just after (short samples) | ~100 tok/s at 54k, ~60 tok/s at 108k |

Only the 12 full-attention layers keep a K/V cache, which is about 25 KB a token at bf16. So
8 slots at 128k fit next to the ~50 GB of weights.

## Recommendation: `CTX=65536` for now

A cold prompt near 128k holds its slot for about 6 minutes of prefill. The scheduler gives
prefill half the GPU while other streams decode, so that one prompt also halves everyone
else's decode for the whole stretch. At 64k, cold prefill stays around 2.5 minutes. Agent
loops reuse their prefix, so only the first turn of a session pays the full prefill.

Raise to `CTX=131072` once wide prefill brings a cold 108k prompt under about a minute.
Memory and retrieval already hold at 128k.
