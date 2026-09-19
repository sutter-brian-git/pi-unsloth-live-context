import { getAgentDir } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as path from "node:path";

/**
 * unsloth-live-context
 * --------------------
 * Keeps Pi's `contextWindow` for the local Unsloth Studio provider in sync with
 * the *live* llama-server context, which shrinks dynamically based on free VRAM.
 *
 * models.json only knows the native/max window (e.g. 262144), but the running
 * server may be serving far less (e.g. 115712). If Pi believes it has more
 * context than the server actually does, prompts overflow and requests fail.
 *
 * How it works:
 *   1. Probes GET {studio}/v1/models (the same base URL pi uses) and reads
 *      `context_length` of the loaded model matching the active pi model id
 *      (falls back to the single loaded model, since studio serves one at a time).
 *   2. If it differs from models.json, rewrites that file so every future
 *      session — new, resumed, or in another pi process — starts correct.
 *   3. Reloads the in-memory model catalog (ctx.modelRegistry.refresh) and
 *      re-anchors this session's active model via pi.setModel, so compaction
 *      thresholds, the footer context %, and overflow detection use the live
 *      value immediately.
 *   4. Re-probes before every agent run, so a mid-session VRAM change (studio
 *      restarting llama-server with a smaller -c) is picked up *before* the
 *      next LLM call instead of surfacing as a hard error.
 *
 * Safety: only entries with `loaded: true` are trusted; if studio is offline or
 * restarting, the probe returns null and the configured value is kept. The
 * models.json rewrite is atomic (tmp + rename) and only touches the matching
 * model's contextWindow field.
 */

/** Provider in models.json to keep in sync. Default: the Unsloth Studio provider. */
const PROVIDER_ID = process.env.PI_LIVE_CONTEXT_PROVIDER ?? "unsloth";
const MODELS_JSON = path.join(getAgentDir(), "models.json");
const FALLBACK_BASE_URL = process.env.UNSLOTH_STUDIO_URL ?? "http://127.0.0.1:8888/v1";
const PROBE_TIMEOUT_MS = 2500;

interface StudioModelInfo {
	id?: string;
	loaded?: boolean;
	context_length?: number;
}

function studioBaseUrl(): string {
	try {
		const doc = JSON.parse(fs.readFileSync(MODELS_JSON, "utf8"));
		const baseUrl = doc?.providers?.[PROVIDER_ID]?.baseUrl;
		if (typeof baseUrl === "string" && baseUrl.startsWith("http")) {
			return baseUrl.replace(/\/+$/, "");
		}
	} catch {
		// fall through to default
	}
	return FALLBACK_BASE_URL;
}

/** Returns the live context_length for the given pi model id, or null if unknown. */
async function probeLiveContext(modelId: string): Promise<number | null> {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
	try {
		const res = await fetch(`${studioBaseUrl()}/models`, { signal: controller.signal });
		if (!res.ok) return null;
		const body = (await res.json()) as { data?: StudioModelInfo[] };
		const models = Array.isArray(body.data) ? body.data : [];
		const loaded = models.filter(
			(m) => m.loaded === true && typeof m.context_length === "number" && m.context_length > 0,
		);
		if (loaded.length === 0) return null;
		const baseId = modelId.split(":")[0];
		const match = loaded.find((m) => m.id === modelId || m.id === baseId);
		return (match ?? (loaded.length === 1 ? loaded[0] : undefined))?.context_length ?? null;
	} catch {
		return null; // studio offline / restarting — keep the configured value
	} finally {
		clearTimeout(timer);
	}
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

	const syncActiveModel = async (ctx: ExtensionContext) => {
		if (syncing) return;
		const model = ctx.model;
		if (!model || model.provider !== PROVIDER_ID) return;
		syncing = true;
		try {
			const live = await probeLiveContext(model.id);
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

	// New sessions, /resume, /new, /fork, /reload — all fire session_start.
	pi.on("session_start", (_event, ctx) => {
		void syncActiveModel(ctx);
	});
	// Model switches (including "restore" on resume).
	pi.on("model_select", (event, ctx) => {
		if (event.model.provider === PROVIDER_ID) void syncActiveModel(ctx);
	});
	// Catch mid-session VRAM changes before the next LLM call.
	pi.on("before_agent_start", (_event, ctx) => {
		void syncActiveModel(ctx);
	});
}
