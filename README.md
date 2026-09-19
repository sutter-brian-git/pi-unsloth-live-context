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

On session start, on model select, and before every agent run it:

1. **Probes** `GET {studio}/v1/models` (the same base URL pi already uses) and
   reads the live `context_length` of the loaded model matching your active pi
   model id (falling back to the single loaded model, since studio serves one at
   a time).
2. **Persists** that value into `models.json` (atomic write) so every future
   session — new, resumed, or in another pi process — starts correct.
3. **Reloads** the in-memory model catalog and re-anchors the current session's
   active model via `pi.setModel`, so compaction thresholds, the footer context
   %, and overflow detection use the live value immediately — no `/model` dance,
   no restart.
4. **Re-probes before every LLM call**, so a mid-session VRAM change (studio
   restarting `llama-server` with a smaller `-c`) is picked up *before* the next
   request instead of surfacing as a hard error.

The handlers **await** the sync — pi awaits extension handlers on these events,
so the agent loop is blocked until the model is re-anchored. That ordering
matters: pi's auto-compaction check runs right after `before_agent_start`, so a
fire-and-forget sync would race it and could compact against a stale (lower)
limit even though the live server has grown. Worst-case latency added per turn
is the probe timeout (~2.5 s) when studio is unreachable; in sync, it's a
~1–5 ms local HTTP GET.

When the live value changes you'll see a status line in the footer:
`unsloth-ctx: live context 111,872 tok`.

## Safety

- Only entries with `loaded: true` are trusted. If studio is offline or
  mid-restart, the probe returns `null` and your configured value is kept — it
  never clobbers a good value with a bad one.
- The `models.json` rewrite is atomic (write to a temp file, then rename) and
  only touches the matching model's `contextWindow` field; everything else in
  the file is preserved byte-for-byte via a JSON round-trip.
- All sync work is wrapped in try/catch with a re-entrancy guard, so a failing
  probe can never break your session or block a request.

## Install

Requires a recent pi that exposes `ctx.modelRegistry.refresh()` and
`pi.setModel()` to extensions.

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
| `UNSLOTH_STUDIO_URL` | `http://127.0.0.1:8888/v1` | OpenAI-compatible base URL to probe, used only if `models.json` doesn't already define a `baseUrl` for the provider. |
| `PI_LIVE_CONTEXT_PROVIDER` | `unsloth` | The provider id in `models.json` to keep in sync. |

The studio base URL is read from your `models.json` provider entry when present,
so it automatically follows whatever URL pi actually talks to.

## How model matching works

pi model ids can carry a variant suffix (e.g.
`cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF:Qwen3.8-27B-iMatrix-NVFP4-MTP`)
while studio reports the base GGUF name. The extension matches on the full id or
the part before `:`; if nothing matches and exactly one model is loaded, it uses
that one.
