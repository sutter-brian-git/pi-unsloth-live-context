# pi-unsloth-live-context

A [pi](https://github.com/badlogic/pi-mono) coding-agent extension that keeps the
`contextWindow` of your local **Unsloth Dynamic Studio** model in sync with the
*live* `llama-server` context — which shrinks and grows dynamically based on how
much VRAM is free.

## The problem

`models.json` can only store a single static `contextWindow`. For a local model
that's usually the **native/max** window (e.g. `262144`). But Unsloth Dynamic
Studio runs `llama-server` with a much smaller, *fluctuating* `-c`/`n_ctx`
(e.g. `115712`) so the model fits alongside whatever else is using your GPU.

If pi believes it has more context than the server actually does, two things go
wrong:

- The footer context % and auto-compaction thresholds are wrong, so pi won't
  compact when it should.
- Prompts eventually exceed the real window and requests **hard-fail** with a
  `400` instead of degrading gracefully.

## What this extension does

**On session start and on model select** (passive — never forces a load, so
starting pi doesn't evict whatever you have loaded in Studio):

1. **Probes** `GET {studio}/v1/models` (the same base URL pi already uses) and
   reads the live `context_length` of the loaded model matching your active pi
   model id (falling back to the single loaded model, since studio serves one at
   a time). Only entries with `loaded: true` are trusted.
2. **Persists** that value into `models.json` (atomic write) so every future
   session — new, resumed, or in another pi process — starts correct.
3. **Reloads** the in-memory model catalog and re-anchors the current session's
   active model via `pi.setModel`, so compaction thresholds, the footer context
   %, and overflow detection use the live value immediately — no `/model` dance,
   no restart.

**Before every agent run** (the submission gate):

4. If the model isn't resident yet, it **triggers the Studio auto-switch load**
   with a minimal `/v1/chat/completions` request (using the full quant-suffixed
   pi model id so Studio applies your saved per-model settings) and **polls until
   the model reports its live `context_length`**. Only then does it let pi submit
   the real prompt.
5. **Re-probes** for mid-session VRAM changes (studio restarting `llama-server`
   with a smaller `-c`) and re-anchors if the value moved.
6. **Compacts before submission if needed.** Since pi 0.86.x the agent loop only
   runs its auto-compaction check *between* turns — the first LLM call of a
   prompt is submitted without any compaction gate, so a resumed session that no
   longer fits the live window hard-fails with `Message too long`. After
   re-anchoring, the extension checks pi's own context estimate and, if it
   exceeds the live window minus a reserve margin (max of 20k or 10% of the
   window), triggers pi's standard compaction (`ctx.compact()`) and **waits for
   the compaction entry to land** before letting the prompt through.

The handlers **await** all of this — pi awaits extension handlers on these
events, so nothing is submitted to studio before the live context is known. (pi
wires `ctx.compact()` as fire-and-forget with `onComplete`/`onError` callbacks,
so the extension waits on those plus a session-branch poll.) Worst-case latency
added per prompt: the load timeout when Studio has to cold-load, or the
compaction wait when it summarizes; in sync it's a ~1–5 ms local HTTP GET.

When the live value changes you'll see a status line in the footer:
`unsloth-ctx: live context 111,872 tok`.

## Safety

- Only entries with `loaded: true` are trusted. If studio is offline or
  mid-restart, the probe returns `null` and your configured value is kept — it
  never clobbers a good value with a bad one.
- A cold load that doesn't finish within the load timeout falls back to the
  configured value (pi submits; studio handles it) rather than blocking forever.
- If pre-submission compaction fails or times out, the extension logs why and
  submits anyway — you'll see pi's own `Message too long` error plus a clear
  hint (`consider /new or freeing VRAM`) on stderr. Re-prompting retries.
- The load trigger is a single minimal chat completion; Studio serializes
  auto-switch swaps with an internal lock, so it can't race your real request.
- The `models.json` rewrite is atomic (write to a temp file, then rename) and
  only touches the matching model's `contextWindow` field; everything else in
  the file is preserved byte-for-byte via a JSON round-trip.
- All sync work is wrapped in try/catch with a re-entrancy guard, so a failing
  probe can never break your session.

## Install

Requires a recent pi that exposes `ctx.modelRegistry.refresh()`,
`pi.setModel()`, and (for the pre-submission compaction gate) `ctx.compact()` /
`ctx.getContextUsage()` to extensions. Older versions degrade gracefully:
sync/re-anchor still work, but an oversized prompt can still hit Studio's
`Message too long` wall.

```sh
git clone https://github.com/sutter-brian-git/pi-unsloth-live-context \
  ~/.pi/agent/extensions/unsloth-live-context
```

Then start pi (or run `/reload` in a running session). No build step — pi loads
the TypeScript directly.

> **Note:** an already-running pi process only picks up a newly installed
> extension on the next start or after `/reload`.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `UNSLOTH_STUDIO_URL` | — (falls back to the provider `baseUrl` in `models.json`, then `http://127.0.0.1:8888/v1`) | Authoritative override for the OpenAI-compatible base URL to probe. The API key always comes from `models.json`. |
| `PI_LIVE_CONTEXT_PROVIDER` | `unsloth` | The provider id in `models.json` to keep in sync. |
| `PI_LIVE_CONTEXT_LOAD_TIMEOUT` | `120` (seconds) | How long to wait for a cold model load before keeping the configured value. |
| `PI_LIVE_CONTEXT_COMPACT_WAIT` | `600` (seconds) | How long to wait for a triggered pre-submission compaction before submitting anyway. |

By default the studio base URL is read from your `models.json` provider entry,
so it automatically follows whatever URL pi actually talks to.

## How model matching works

pi model ids can carry a variant suffix (e.g.
`cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF:Qwen3.8-27B-iMatrix-NVFP4-MTP`)
while studio reports the base GGUF name. The extension matches on the full id or
the part before `:`; if nothing matches and exactly one model is loaded, it uses
that one.
