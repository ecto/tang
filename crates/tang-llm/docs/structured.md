# Constrained chat output

`/v1/chat/completions` accepts `response_format` with type `text`, `json_object` or
`json_schema`. Schema requests use `json_schema.schema`; name and strict fields are
accepted metadata. The supplied schema is enforced during token sampling through
llguidance 1.9.1, using the loaded local Hugging Face tokenizer. No tokenizer or
schema is fetched from a remote service. Qwen ByteLevel and Gemma ByteFallback
formats were exercised against real local weights.

Schemas are at most 64 KiB and compile before queueing. Invalid format/schema
requests return HTTP 400. Stop strings cannot be combined with constrained output.
Reasoning is disabled for its prompt and speculative drafts are disabled for the
request. EOS tokens remain unavailable until the grammar accepts; regular token
and context limits still apply, so a length-limited response can be an incomplete
JSON prefix. Require `finish_reason: stop` for a completed result.

Each model builds and reuses a tokenizer/grammar factory; each request owns its
matcher. Unconstrained requests retain ordinary reasoning, stop and speculation
behavior. Models advertise `json_object` and `json_schema` capabilities.

```sh
python3 crates/tang-llm/scripts/test_structured_api.py \
  --base-url http://127.0.0.1:18915/v1 \
  --model /path/to/local/model \
  --output /tmp/new-structured-api-test
```

Validation: a real Qwen3-4B checkpoint returns required constant fields and numeric
bounds despite a conflicting prompt; streaming returns valid JSON; invalid type,
invalid schema and oversized schema return HTTP 400. Constrained requests use zero
drafts with global speculation enabled. The real Frog agent executes Image, Write,
Screenshot, Compare and subsequent Edit revisions without malformed JSON. Its visual
quality remains poor and repeated ineffective edits trigger Frog's loop detector;
format correctness does not establish task completion.

Inter-token indentation is limited to eight whitespace characters. This prevents
a model from repeatedly generating whitespace before a required value until its
token budget expires. String contents retain ordinary JSON whitespace and escapes.

The [llguidance Rust example](https://github.com/guidance-ai/llguidance/blob/main/sample_parser/src/minimal.rs)
shows the underlying mask/sample/consume API.
