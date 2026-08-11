#!/usr/bin/env node
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { fileURLToPath } from "node:url";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";

const DEFAULT_PROJECT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_STATE_DIR = path.join(os.homedir(), ".openclaw");
const DEFAULT_TMP_DIR = process.env.ARTHA_OPENCLAW_TMP_DIR || path.join(DEFAULT_STATE_DIR, "workspace", "tmp");
const DEFAULT_TIMEOUT_MS = 120000;

function parseArgs(argv) {
  const args = {
    projectDir: DEFAULT_PROJECT_DIR,
    stateDir: process.env.OPENCLAW_STATE_DIR || DEFAULT_STATE_DIR,
    openclawConfig: process.env.OPENCLAW_CONFIG_PATH || path.join(DEFAULT_STATE_DIR, "openclaw.json"),
    tmpDir: DEFAULT_TMP_DIR,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    callbackData: "",
    telegram: false,
    lockFile: "/tmp/artha-robinhood-manual-action.lock",
    lockWaitMs: 90000,
    oauthFile: null,
    mcpUrl: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`Missing value for ${arg}`);
      return argv[i];
    };
    if (arg === "--callback-data") args.callbackData = next();
    else if (arg === "--project-dir") args.projectDir = next();
    else if (arg === "--state-dir") args.stateDir = next();
    else if (arg === "--openclaw-config") args.openclawConfig = next();
    else if (arg === "--tmp-dir") args.tmpDir = next();
    else if (arg === "--timeout-ms") args.timeoutMs = Number(next());
    else if (arg === "--lock-file") args.lockFile = next();
    else if (arg === "--lock-wait-ms") args.lockWaitMs = Number(next());
    else if (arg === "--oauth-file") args.oauthFile = next();
    else if (arg === "--mcp-url") args.mcpUrl = next();
    else if (arg === "--telegram") args.telegram = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.callbackData) throw new Error("Missing --callback-data");
  args.projectDir = path.resolve(expandHome(args.projectDir));
  args.stateDir = path.resolve(expandHome(args.stateDir));
  args.openclawConfig = path.resolve(expandHome(args.openclawConfig));
  args.tmpDir = path.resolve(expandHome(args.tmpDir));
  args.lockFile = path.resolve(expandHome(args.lockFile));
  args.python = path.join(args.projectDir, ".venv", "bin", "python");
  return args;
}

function expandHome(value) {
  if (!value) return value;
  if (value === "~") return os.homedir();
  if (value.startsWith("~/")) return path.join(os.homedir(), value.slice(2));
  return value;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function atomicWriteJson(file, payload) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp-${process.pid}-${randomBytes(4).toString("hex")}`;
  fs.writeFileSync(tmp, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, file);
}

function resolveConfig(args) {
  const openclaw = fs.existsSync(args.openclawConfig) ? readJson(args.openclawConfig) : {};
  const server = openclaw?.mcp?.servers?.["robinhood-trading"] || {};
  return {
    mcpUrl: args.mcpUrl || server.url || "https://agent.robinhood.com/mcp/trading",
    oauthFile: args.oauthFile || findOauthFile(args.stateDir),
  };
}

function findOauthFile(stateDir) {
  const dir = path.join(stateDir, "mcp-oauth");
  const candidates = fs.existsSync(dir)
    ? fs.readdirSync(dir)
      .filter((name) => /^robinhood-trading-.*\.json$/.test(name))
      .map((name) => path.join(dir, name))
      .filter((file) => {
        try {
          return Boolean(readJson(file)?.tokens?.access_token);
        } catch {
          return false;
        }
      })
    : [];
  candidates.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  if (!candidates.length) throw new Error(`No authorized OpenClaw Robinhood OAuth file found under ${dir}`);
  return candidates[0];
}

function createOAuthProvider(oauthFile) {
  let payload = readJson(oauthFile);
  const save = () => atomicWriteJson(oauthFile, payload);
  return {
    clientMetadata: {
      client_name: "Artha Robinhood Review Runner",
      redirect_uris: ["http://127.0.0.1:8989/oauth/callback"],
      token_endpoint_auth_method: "none",
      scope: "internal",
    },
    redirectUrl: "http://127.0.0.1:8989/oauth/callback",
    state: async () => payload.state,
    tokens: async () => payload.tokens,
    saveTokens: async (tokens) => {
      payload.tokens = { ...(payload.tokens || {}), ...tokens, refresh_token: tokens.refresh_token || payload?.tokens?.refresh_token };
      save();
    },
    clientInformation: async () => payload.clientInformation,
    saveClientInformation: async (clientInformation) => {
      payload.clientInformation = clientInformation;
      save();
    },
    discoveryState: async () => payload.discoveryState,
    saveDiscoveryState: async (discoveryState) => {
      payload.discoveryState = discoveryState;
      save();
    },
    saveCodeVerifier: async (codeVerifier) => {
      payload.codeVerifier = codeVerifier;
      save();
    },
    invalidateCredentials: async (scope) => {
      if (scope === "tokens") payload.tokens = {};
      if (scope === "all") {
        payload.tokens = {};
        payload.clientInformation = undefined;
      }
      save();
    },
    redirectToAuthorization: async (authorizationUrl) => {
      payload.lastAuthorizationUrl = authorizationUrl.toString();
      save();
      throw new Error("Robinhood OAuth reauthorization required: run `openclaw mcp login robinhood-trading`.");
    },
  };
}

async function withTimeout(promise, timeoutMs, label) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function callTool(client, name, args, timeoutMs) {
  return withTimeout(client.callTool({ name, arguments: args }, CallToolResultSchema), timeoutMs, name);
}

async function runSnapshotSync(args) {
  await runProcess("/opt/homebrew/bin/node", ["scripts/robinhood_snapshot_sync.mjs", "--market-hours-only", "--quiet"], {
    cwd: args.projectDir,
    timeoutMs: 180000,
  });
}

function runProcess(command, processArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, processArgs, {
      cwd: options.cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    const timer = options.timeoutMs
      ? setTimeout(() => {
        child.kill("SIGTERM");
        setTimeout(() => child.kill("SIGKILL"), 5000).unref();
      }, options.timeoutMs)
      : null;
    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (timer) clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`${command} ${processArgs.join(" ")} failed rc=${code} signal=${signal || ""} stdout=${stdout.slice(0, 1000)} stderr=${stderr.slice(0, 1000)}`));
        return;
      }
      resolve({ code, signal, stdout, stderr });
    });
  });
}

async function acquireLock(lockFile, waitMs) {
  const staleMs = 10 * 60 * 1000;
  const started = Date.now();
  fs.mkdirSync(path.dirname(lockFile), { recursive: true });
  const pidAlive = (pid) => {
    const value = Number(pid);
    if (!Number.isInteger(value) || value <= 0) return false;
    try {
      process.kill(value, 0);
      return true;
    } catch (error) {
      return error?.code === "EPERM";
    }
  };
  while (true) {
    try {
      const fd = fs.openSync(lockFile, "wx", 0o600);
      fs.writeFileSync(fd, `${JSON.stringify({ pid: process.pid, acquired_at: new Date().toISOString() })}\n`, "utf8");
      return () => {
        try { fs.closeSync(fd); } catch {}
        try { fs.unlinkSync(lockFile); } catch {}
      };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      let stale = false;
      let deadOwner = false;
      try {
        const state = JSON.parse(fs.readFileSync(lockFile, "utf8"));
        stale = Date.now() - fs.statSync(lockFile).mtimeMs > staleMs;
        deadOwner = Boolean(state.pid) && !pidAlive(state.pid);
      } catch {
        stale = true;
      }
      if (deadOwner || stale) {
        try {
          fs.unlinkSync(lockFile);
          continue;
        } catch {}
      }
      if (Date.now() - started >= waitMs) return null;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
}

function extractToolData(payload) {
  if (payload?.structuredContent?.data && typeof payload.structuredContent.data === "object") return payload.structuredContent.data;
  if (payload?.data && typeof payload.data === "object") return payload.data;
  for (const item of payload?.content || []) {
    if (item?.type !== "text") continue;
    try {
      const parsed = JSON.parse(item.text || "{}");
      if (parsed?.data && typeof parsed.data === "object") return parsed.data;
      if (parsed && typeof parsed === "object") return parsed;
    } catch {}
  }
  return {};
}

function orderFromToolPayload(payload) {
  if (payload?.order && typeof payload.order === "object") return payload.order;
  const data = extractToolData(payload);
  if (data?.order && typeof data.order === "object") return data.order;
  return data && typeof data === "object" ? data : {};
}

function argsMatchForPlace(reviewArgs, placeArgs) {
  const keys = ["account_number", "symbol", "side", "type", "quantity", "dollar_amount", "limit_price", "stop_price", "time_in_force", "market_hours"];
  const mismatches = [];
  for (const key of keys) {
    const left = reviewArgs?.[key] == null ? "" : String(reviewArgs[key]);
    const right = placeArgs?.[key] == null ? "" : String(placeArgs[key]);
    if (left !== right) mismatches.push(`${key}: review=${left || "-"} place=${right || "-"}`);
  }
  return { ok: mismatches.length === 0, mismatches };
}

function runArtha(args, commandArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(args.python, ["run.py", ...commandArgs], {
      cwd: args.projectDir,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("close", (code) => {
      const parsed = parseJsonPrefix(stdout);
      if (code !== 0 && options.requireZero !== false) {
        reject(new Error(`Artha command failed (${code}): ${stderr || stdout}`));
        return;
      }
      resolve({ code, stdout, stderr, parsed });
    });
  });
}

function parseJsonPrefix(text) {
  const start = String(text || "").indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (char === "\\") {
      escape = true;
      continue;
    }
    if (char === "\"") {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (char === "{") depth += 1;
    else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(start, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const release = await acquireLock(args.lockFile, args.lockWaitMs);
  if (!release) {
    console.log(JSON.stringify({ success: false, status: "LOCKED", message: "Another manual Robinhood action is already running." }, null, 2));
    process.exit(1);
  }
  try {
  if (args.callbackData.startsWith("artha:place:")) await runSnapshotSync(args);
  const op = (await runArtha(args, ["robinhood-action", args.callbackData])).parsed;
  if (!op?.success || op.operation !== "tradability_then_review_equity_order") {
    if (!op?.success || op.operation !== "tradability_then_review_then_place_equity_order") {
      console.log(JSON.stringify({ success: false, stage: "resolve", operation: op }, null, 2));
      process.exit(1);
    }
  }

  const { mcpUrl, oauthFile } = resolveConfig(args);
  const transport = new StreamableHTTPClientTransport(new URL(mcpUrl), {
    authProvider: createOAuthProvider(oauthFile),
  });
  const client = new Client({ name: "artha-robinhood-review-runner", version: "1.0.0" });
  await client.connect(transport);

  try {
    const prefix = path.join(args.tmpDir, `artha_manual_${op.action_id}`);
    const tradability = await callTool(client, "get_equity_tradability", op.tradability_mcp_args, args.timeoutMs);
    const review = await callTool(client, "review_equity_order", op.review_mcp_args, args.timeoutMs);
    const tradabilityFile = `${prefix}_tradability.json`;
    const reviewFile = `${prefix}_review.json`;
    atomicWriteJson(tradabilityFile, tradability);
    atomicWriteJson(reviewFile, review);
    const recordArgs = ["robinhood-record-review", op.action_id, "--tradability-file", tradabilityFile, "--review-file", reviewFile];
    if (args.telegram) recordArgs.push("--telegram");
    const recorded = await runArtha(args, recordArgs, { requireZero: false });
    if (op.operation === "tradability_then_review_equity_order") {
    console.log(JSON.stringify({
      success: recorded.parsed?.status === "review_clear",
      action_id: op.action_id,
      symbol: op.review_mcp_args?.symbol,
      amount: op.review_mcp_args?.dollar_amount || op.review_mcp_args?.quantity,
      record_exit_code: recorded.code,
      recorded: recorded.parsed,
      files: { tradability: tradabilityFile, review: reviewFile },
    }, null, 2));
    process.exit(recorded.parsed?.status === "review_clear" ? 0 : 1);
    }
    if (recorded.parsed?.status !== "review_clear") {
      console.log(JSON.stringify({ success: false, stage: "second_review", action_id: op.action_id, recorded: recorded.parsed }, null, 2));
      process.exit(1);
    }

    const final = await runArtha(args, ["robinhood-final-clearance", op.action_id], { requireZero: false });
    if (!final.parsed?.allow_place) {
      console.log(JSON.stringify({ success: false, stage: "final_clearance", action_id: op.action_id, final: final.parsed }, null, 2));
      process.exit(1);
    }

    const placeOp = (await runArtha(args, ["robinhood-action", args.callbackData])).parsed;
    if (!placeOp?.success || placeOp.operation !== "tradability_then_review_then_place_equity_order" || !placeOp.place_mcp_args) {
      console.log(JSON.stringify({ success: false, stage: "final_place_args", action_id: op.action_id, place_operation: placeOp }, null, 2));
      process.exit(1);
    }
    const match = argsMatchForPlace(op.review_mcp_args, placeOp.place_mcp_args);
    if (!match.ok) {
      console.log(JSON.stringify({ success: false, stage: "place_args_match", action_id: op.action_id, mismatches: match.mismatches }, null, 2));
      process.exit(1);
    }

    const place = await callTool(client, "place_equity_order", placeOp.place_mcp_args, args.timeoutMs);
    const placeFile = `${prefix}_place.json`;
    atomicWriteJson(placeFile, place);
    const submission = await runArtha(args, ["robinhood-record-submission", op.action_id, "--file", placeFile], { requireZero: false });
    await runSnapshotSync(args);
    const order = orderFromToolPayload(place);
    console.log(JSON.stringify({
      success: submission.parsed?.status === "PASS",
      action_id: op.action_id,
      symbol: placeOp.place_mcp_args?.symbol,
      amount: placeOp.place_mcp_args?.dollar_amount || placeOp.place_mcp_args?.quantity,
      order,
      submission: submission.parsed,
      files: { tradability: tradabilityFile, review: reviewFile, place: placeFile },
    }, null, 2));
    process.exit(submission.parsed?.status === "PASS" ? 0 : 1);
  } finally {
    await client.close();
  }
  } finally {
    release();
  }
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
