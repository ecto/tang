# Retention: `--kv-ttl`

`tang-llm serve --kv-ttl <secs>` (or `TANG_KV_TTL=<secs>`) bounds how long anything computed
from a prompt outlives its last use. Off by default (`0` or `off`): caches keep entries until
their room is needed, as before.

With a TTL, an entry unused for that long is gone:

| Where | What | Cleared |
| --- | --- | --- |
| KV block pool (`blocks.rs`) | sealed 256-position blocks no conversation holds | freed |
| Engine slots (`engine.rs`) | parked conversations, the idle active one, their `prompt_cache_key` | dropped |
| Disk tier (`kvstore.rs`, `~/.cache/tang/kv/`) | `blocks/<hash>.kvb`, `<key>.tail`, old `.kv`/`.json` | files deleted |
| Draft store (`draft.rs`, `~/.cache/tang/drafts/<model>.tok`) | earlier completions for suffix drafts | dropped, file rewritten (deleted when empty) |
| Draft model | its KV cache and pool | dropped once the server is idle past the TTL |
| Flash backend (`flash/serve.rs`) | saved `RunningState`s in RAM, remembered replies for splicing, the live sequence | dropped / reset |
| Flash backend disk (`*.state`, including the pinned system-prompt state) | | files deleted |

"Use" refreshes the clock: attaching or sealing a block, picking a conversation, reading or
rewriting a disk file (its modification time is the clock), restoring a saved state. A draft
store entry counts from when it was added. Requests that are running are never touched.

Two mechanisms, so nothing past the TTL is served and nothing lingers:

- **Lazily**, on lookup: an expired block, file, saved state or conversation is skipped (a
  disk file is deleted right there).
- **A sweep** in the server's worker, every tenth of the TTL (between 1 s and 1 min), also
  while idle: frees what's expired in memory and deletes expired files. `/node` is refreshed
  after each sweep, and its `kv.disk_blocks` leaves out files past the TTL even before one.

Not covered: `--dump-ids` (an explicit debugging option that writes prompts on purpose) and the
Flash backend's `routing.bin` (aggregate expert counts, not per prompt). Logs carry token
counts and timings, not text.

Zest runs with `--kv-ttl 3600`.
