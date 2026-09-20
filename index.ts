import { getAgentDir } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";

/**
 * unsloth-live-context
 * --------------------
 * Keeps Pi's `contextWindow` for the local Unsloth Studio provider in sync with
 * the *live* llama-server context, which shrinks and grows dynamically based on
 * how much VRAM is free.
 *
 * models.json can only store one static value, and Studio enforces the LOADED
 * context window against every incoming prompt (a bigger prompt gets a hard
 * 400 "Message too long: N tokens exceeds the M-token context window"). So pi
 * must never believe it has more context than the server will actually serve.
 *
 * How it works:
 *   1. Probes GET {studio}/v1/models (the same base URL pi uses) and reads the
 *      live `context_length` of the loaded model matching the active pi model
 *      id (falls back to the single loaded model, since studio serves one at a
 *      time). Only entries with `loaded: true` are trusted — unloaded models
 *      report no context at all.
 *   2. Before every agent run, if the model is not resident yet, it triggers
 *      the Studio auto-switch load with a minimal /v1/chat/completions request
 *      (using the full quant-suffixed pi model id so Studio applies the saved
 *      per-model settings) and polls until the model reports its live
 *      context_length. Only then does it let pi submit the real prompt.
 *   3. Persists the value into models.json (atomic write) so every future
 *      session — new, resumed, or in another pi process — starts correct.
 *   4. Reloads the in-memory model catalog and re-anchors this session's active
 *      model via pi.setModel, so compaction thresholds, the footer context %,
 *      and overflow detection use the live value immediately.
 *   5. Compacts if needed: since pi 0.86.x the agent loop only runs its
 *      auto-compaction check *between* turns — the first LLM call of a prompt
 *      is never gated. So after re-anchoring, if pi's own context estimate
 *      exceeds the live window minus a reserve margin, the extension triggers
 *      pi's standard compaction (ctx.compact()) before submission. That makes
 *      "Message too long" impossible: whatever pi sends next fits what studio
 *      will serve.
 *
 * The handlers await all of this (pi awaits extension handlers on these
 * events), which is deliberate: worst-case added latency per prompt is the
 * load timeout when Studio has to cold-load; in sync it's a ~1–5 ms local
 * HTTP GET.
 *
 * Safety: if studio is unreachable the probe fails fast and the configured
 * value is kept; a load that doesn't finish within the timeout falls back to
 * the configured value (pi submits, studio handles it — the old behavior).
 * The models.json rewrite is atomic (tmp + rename) and only touches the
 * matching model's contextWindow field. All sync work is wrapped in try/catch
 * with a re-entrancy guard, so a failing probe can never break a session.
 */

/** Provider in models.json to keep in sync. Default: the Unsloth Studio provider. */
const PROVIDER_ID = process.env.PI_LIVE_CONTEXT_PROVIDER ?? "unsloth";
const MODELS_JSON = path.join(getAgentDir(), "models.json");
const FALLBACK_BASE_URL = process.env.UNSLOTH_STUDIO_URL ?? "http://127.0.0.1:8888/v1";
const PROBE_TIMEOUT_MS = 2500;
/** How long to wait for a cold model load before giving up and keeping the configured value. */
const LOAD_TIMEOUT_MS = Math.max(5, Number(process.env.PI_LIVE_CONTEXT_LOAD_TIMEOUT ?? 120)) * 1000;
const POLL_INTERVAL_MS = 1500;
/** Reserve margin before compacting: max(20k, 10% of the live window). */
const RESERVE_MIN = 20_000;
const RESERVE_FRACTION = 0.1;
/** How long to wait for a triggered compaction to finish before submitting anyway.
 * Summarizing a ~150k-token branch on a local model takes several minutes; keep this generous. */
const COMPACT_WAIT_MS = Math.max(30, Number(process.env.PI_LIVE_CONTEXT_COMPACT_WAIT ?? 600)) * 1000;

interface StudioModelInfo {
	id?: string;
	loaded?: boolean;
	context_length?: number;
}

type ProbeResult =
	| { kind: "unreachable" } // studio offline / erroring — keep configured value, fail fast
	| { kind: "loaded"; context: number }
	| { kind: "not_loaded" }; // studio is up but the model isn't resident yet

/**
 * Probe endpoint: UNSLOTH_STUDIO_URL (if set) wins, then the provider's baseUrl
 * from models.json, then the default. The apiKey always comes from models.json.
 */
function studioEndpoint(): { url: string; apiKey?: string } {
	let baseUrl: string | undefined;
	let apiKey: string | undefined;
	try {
		const doc = JSON.parse(fs.readFileSync(MODELS_JSON, "utf8"));
		const p = (doc as any)?.providers?.[PROVIDER_ID];
		if (typeof p?.baseUrl === "string" && p.baseUrl.startsWith("http")) baseUrl = p.baseUrl.replace(/\/+$/, "");
		if (typeof p?.apiKey === "string") apiKey = p.apiKey;
	} catch {
		// fall through to default
	}
	return { url: process.env.UNSLOTH_STUDIO_URL?.replace(/\/+$/, "") ?? baseUrl ?? FALLBACK_BASE_URL, apiKey };
}

async function fetchStudioModels(url: string, apiKey?: string): Promise<StudioModelInfo[] | null> {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
	try {
		const headers: Record<string, string> = {};
		if (apiKey) headers["authorization"] = `Bearer ${apiKey}`;
		const res = await fetch(`${url}/models`, { headers, signal: controller.signal });
		if (!res.ok) return null;
		const body = (await res.json()) as { data?: StudioModelInfo[] };
		return Array.isArray(body.data) ? body.data : [];
	} catch {
		return null;
	} finally {
		clearTimeout(timer);
	}
}

async function probeStudio(modelId: string): Promise<ProbeResult> {
	const { url, apiKey } = studioEndpoint();
	const models = await fetchStudioModels(url, apiKey);
	if (models === null) return { kind: "unreachable" };
	const loaded = models.filter(
		(m) => m.loaded === true && typeof m.context_length === "number" && m.context_length > 0,
	);
	if (loaded.length === 0) return { kind: "not_loaded" };
	const baseId = modelId.split(":")[0];
	const match = loaded.find((m) => m.id === modelId || m.id === baseId);
	const ctx = (match ?? (loaded.length === 1 ? loaded[0] : undefined))?.context_length;
	return typeof ctx === "number" && ctx > 0 ? { kind: "loaded", context: ctx } : { kind: "not_loaded" };
}

/** Passive probe: live context if the model is resident, else null (keep configured value). */
async function probeLiveContext(modelId: string): Promise<number | null> {
	const p = await probeStudio(modelId);
	return p.kind === "loaded" ? p.context : null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Like probeLiveContext, but if the model isn't resident yet it triggers the
 * Studio auto-switch load and waits (up to LOAD_TIMEOUT_MS) for the live
 * context to appear. Returns null when the value can't be determined.
 */
async function ensureLoadedContext(modelId: string): Promise<number | null> {
	const first = await probeStudio(modelId);
	if (first.kind === "loaded") return first.context;
	if (first.kind === "unreachable") return null;

	// Studio is up but our model isn't resident. Trigger the auto-switch load
	// with a minimal request. The full quant-suffixed id matters: bare ids load
	// with app defaults, the suffixed form loads with saved per-model settings.
	const { url, apiKey } = studioEndpoint();
	const headers: Record<string, string> = { "content-type": "application/json" };
	if (apiKey) headers["authorization"] = `Bearer ${apiKey}`;
	let loadInFlight = true;
	try {
		const res = await fetch(`${url}/chat/completions`, {
			method: "POST",
			headers,
			body: JSON.stringify({
				model: modelId,
				messages: [{ role: "user", content: "." }],
				max_tokens: 1,
				stream: false,
			}),
			signal: AbortSignal.timeout(LOAD_TIMEOUT_MS),
		});
		loadInFlight = res.ok; // explicit refusal (404/401/…) → polling is pointless
		await res.arrayBuffer().catch(() => {});
	} catch {
		loadInFlight = true; // timeout/network blip → the load may still be progressing
	}
	if (!loadInFlight) return null;

	const deadline = Date.now() + LOAD_TIMEOUT_MS;
	while (Date.now() < deadline) {
		await sleep(POLL_INTERVAL_MS);
		const p = await probeStudio(modelId);
		if (p.kind === "loaded") return p.context;
		if (p.kind === "unreachable") return null; // studio died mid-load — keep configured value
	}
	return null; // timed out — keep configured value; pi submits and studio handles it
}

/** Persists contextWindow into models.json (atomic). True if in sync after call. */
function persistContextWindow(modelId: string, contextWindow: number): boolean {
	let doc: unknown;
	try {
		doc = JSON.parse(fs.readFileSync(MODELS_JSON, "utf8"));
	} catch {
		return false;
	}
	const models = (doc as any)?.providers?.[PROVIDER_ID]?.models;
	const model = Array.isArray(models) ? models.find((m: any) => m?.id === modelId) : undefined;
	if (!model) return false;
	if (model.contextWindow === contextWindow) return true;
	model.contextWindow = contextWindow;
	const tmp = `${MODELS_JSON}.tmp.${process.pid}`;
	try {
		fs.writeFileSync(tmp, JSON.stringify(doc, null, 2) + "\n", "utf8");
		fs.renameSync(tmp, MODELS_JSON);
		return true;
	} catch {
		try {
			fs.unlinkSync(tmp);
		} catch {
			// ignore cleanup failure
		}
		return false;
	}
}

export default function unslothLiveContext(pi: ExtensionAPI): void {
	let syncing = false;

	const syncActiveModel = async (ctx: ExtensionContext, ensureLoaded = false) => {
		if (syncing) return;
		const model = ctx.model;
		if (!model || model.provider !== PROVIDER_ID) return;
		syncing = true;
		try {
			const live = ensureLoaded ? await ensureLoadedContext(model.id) : await probeLiveContext(model.id);
			if (live === null || model.contextWindow === live) return;

			// 1. Persist for future sessions (new / resumed / other processes).
			persistContextWindow(model.id, live);
			// 2. Reload the in-memory catalog from models.json.
			await ctx.modelRegistry.refresh({ allowNetwork: false });
			// 3. Re-anchor this session's active model object.
			const fresh = ctx.modelRegistry.find(model.provider, model.id);
			if (fresh && fresh.contextWindow !== ctx.model?.contextWindow) {
				await pi.setModel(fresh);
			}
			// Interactive-only; absent in print/rpc modes.
			(pi as any).setStatus?.("unsloth-ctx", `live context ${live.toLocaleString("en-US")} tok`);
		} catch (err) {
			console.error(`[unsloth-live-context] sync failed: ${err instanceof Error ? err.message : String(err)}`);
		} finally {
			syncing = false;
		}
	};

	/**
	 * Pi 0.86.x only auto-compacts *between* turns — the first LLM call of a
	 * prompt is submitted without a compaction gate. So after re-anchoring to
	 * the live window, check pi's own context estimate and trigger pi's
	 * standard compaction if it would overflow what studio will serve.
	 */
	const compactIfOversized = async (ctx: ExtensionContext) => {
		try {
			const usage = (ctx as any).getContextUsage?.() as
				| { tokens?: number | null; contextWindow?: number }
				| undefined;
			if (!usage || usage.tokens == null || !usage.contextWindow || usage.contextWindow <= 0) return;
			const reserve = Math.max(RESERVE_MIN, Math.floor(usage.contextWindow * RESERVE_FRACTION));
			if (usage.tokens <= usage.contextWindow - reserve) return;
			console.error(
				`[unsloth-live-context] context ${usage.tokens.toLocaleString("en-US")} tok exceeds live window ` +
					`${usage.contextWindow.toLocaleString("en-US")} − ${reserve.toLocaleString("en-US")} reserve — compacting before submission`,
			);
			// pi wires ctx.compact() as fire-and-forget (its binding starts the session
			// compaction in a background IIFE and returns immediately, reporting via
			// onComplete/onError), so kick it off and wait for completion before
			// returning. Our handler is awaited by prompt(), so pi cannot submit until
			// the context has actually shrunk. Branch-entry polling covers bindings
			// that ignore the callbacks.
			const getBranch = () => (ctx.sessionManager as any).getBranch?.() as Array<{ type?: string }> | undefined;
			const branchBefore = getBranch()?.length ?? 0;
			await new Promise<void>((resolve) => {
				let done = false;
				const finish = (note?: string) => {
					if (done) return;
					done = true;
					clearInterval(poll);
					clearTimeout(timeout);
					if (note) console.error(`[unsloth-live-context] ${note}`);
					resolve();
				};
				const poll = setInterval(() => {
					const branch = getBranch();
					if (
						branch &&
						branch.length > branchBefore &&
						branch.slice(branchBefore).some((e) => e.type === "compaction")
					) {
						return finish(); // compaction landed - context has shrunk
					}
					const u2 = (ctx as any).getContextUsage?.() as { tokens?: number | null } | undefined;
					if (u2?.tokens != null && u2.tokens <= usage.contextWindow - reserve) return finish(); // shrank anyway
				}, 500);
				const timeout = setTimeout(
					() => finish("compaction did not finish in time - submitting as-is"),
					COMPACT_WAIT_MS,
				);
				try {
					(ctx as any).compact?.({
						onComplete: () => finish(),
						onError: (err: Error) =>
							finish(
								`compaction failed (${err.message}) - submitting as-is; consider /new or freeing VRAM`,
							),
					});
				} catch {
					finish();
				}
			});
		} catch (err) {
			console.error(
				`[unsloth-live-context] pre-submission compaction failed: ${err instanceof Error ? err.message : String(err)}`,
			);
		}
	};

	// NOTE: pi awaits extension handlers on these events, so returning a promise
	// blocks the session/agent loop until the work finishes. That is deliberate:
	// nothing may be submitted to studio before the live context is known.
	// syncActiveModel never rejects (all errors are caught inside), so awaiting
	// it can only delay a prompt by the load timeout when studio must cold-load.

	// New sessions, /resume, /new, /fork, /reload — all fire session_start.
	// Passive: re-anchor if the model is already resident; never force a load,
	// so starting pi doesn't evict whatever the user has loaded in Studio.
	pi.on("session_start", async (_event, ctx) => {
		await syncActiveModel(ctx);
	});
	// Model switches (including "restore" on resume). Also passive.
	pi.on("model_select", async (event, ctx) => {
		if (event.model.provider === PROVIDER_ID) await syncActiveModel(ctx);
	});
	// The submission gate: wait for the model to be resident with a known live
	// context, then compact if the conversation no longer fits.
	pi.on("before_agent_start", async (_event, ctx) => {
		await syncActiveModel(ctx, true);
		await compactIfOversized(ctx);
	});
}
