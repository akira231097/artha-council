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
const DEFAULT_CHAT_ID = process.env.TELEGRAM_CHAT_ID || "";
const DEFAULT_LOCK_FILE = "/tmp/artha-robinhood-auto-trade.lock";
const DEFAULT_TIMEOUT_MS = 120000;

function parseArgs(argv) {
  const args = {
    projectDir: DEFAULT_PROJECT_DIR,
    stateDir: process.env.OPENCLAW_STATE_DIR || DEFAULT_STATE_DIR,
    openclawConfig: process.env.OPENCLAW_CONFIG_PATH || path.join(DEFAULT_STATE_DIR, "openclaw.json"),
    tmpDir: DEFAULT_TMP_DIR,
    lockFile: DEFAULT_LOCK_FILE,
    lockWaitMs: 90000,
    maxActions: 5,
    marketHoursOnly: false,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    telegramChatId: DEFAULT_CHAT_ID,
    oauthFile: null,
    mcpUrl: null,
    accountNumber: null,
    accountSuffix: null,
    quiet: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`Missing value for ${arg}`);
      return argv[i];
    };
    if (arg === "--project-dir") args.projectDir = next();
    else if (arg === "--state-dir") args.stateDir = next();
    else if (arg === "--openclaw-config") args.openclawConfig = next();
    else if (arg === "--tmp-dir") args.tmpDir = next();
    else if (arg === "--lock-file") args.lockFile = next();
    else if (arg === "--lock-wait-ms") args.lockWaitMs = Number(next());
    else if (arg === "--max-actions") args.maxActions = Number(next());
    else if (arg === "--timeout-ms") args.timeoutMs = Number(next());
    else if (arg === "--telegram-chat-id") args.telegramChatId = next();
    else if (arg === "--oauth-file") args.oauthFile = next();
    else if (arg === "--mcp-url") args.mcpUrl = next();
    else if (arg === "--account-number") args.accountNumber = next();
    else if (arg === "--account-suffix") args.accountSuffix = next();
    else if (arg === "--market-hours-only") args.marketHoursOnly = true;
    else if (arg === "--quiet") args.quiet = true;
    else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  args.projectDir = path.resolve(expandHome(args.projectDir));
  args.stateDir = path.resolve(expandHome(args.stateDir));
  args.openclawConfig = path.resolve(expandHome(args.openclawConfig));
  args.tmpDir = path.resolve(expandHome(args.tmpDir));
  args.python = path.join(args.projectDir, ".venv", "bin", "python");
  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/robinhood_auto_trade_runner.mjs [options]

Deterministic Robinhood MCP runner for Artha queued auto_buy/auto_sell actions.

Options:
  --market-hours-only              Exit 0 outside Mon-Fri 08:30-14:59 America/Chicago
  --max-actions <n>                Process at most n queued actions (default: 5)
  --telegram-chat-id <id>          Telegram chat for non-idle success/block alerts
  --oauth-file <path>              OpenClaw OAuth file for robinhood-trading
  --mcp-url <url>                  Robinhood MCP streamable HTTP URL
`);
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

function readDotenv(file) {
  const values = {};
  if (!fs.existsSync(file)) return values;
  for (const rawLine of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const idx = line.indexOf("=");
    if (idx <= 0) continue;
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    values[key] = value;
  }
  return values;
}

function resolveConfig(args) {
  const env = readDotenv(path.join(args.projectDir, ".env"));
  const openclaw = fs.existsSync(args.openclawConfig) ? readJson(args.openclawConfig) : {};
  const server = openclaw?.mcp?.servers?.["robinhood-trading"] || {};
  const mcpUrl = args.mcpUrl || server.url || "https://agent.robinhood.com/mcp/trading";
  const accountNumber = args.accountNumber || process.env.ARTHA_ROBINHOOD_AGENTIC_ACCOUNT_NUMBER || env.ARTHA_ROBINHOOD_AGENTIC_ACCOUNT_NUMBER || "";
  const accountSuffix = args.accountSuffix || (accountNumber ? accountNumber.slice(-4) : "0195");
  const oauthFile = args.oauthFile || findOauthFile(args.stateDir);
  return { mcpUrl, accountNumber, accountSuffix, oauthFile };
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
  if (!candidates.length) {
    throw new Error(`No authorized OpenClaw Robinhood OAuth file found under ${dir}`);
  }
  return candidates[0];
}

function createOAuthProvider(oauthFile) {
  let payload = readJson(oauthFile);
  const save = () => atomicWriteJson(oauthFile, payload);
  return {
    clientMetadata: {
      client_name: "Artha Robinhood Auto Trade Runner",
      redirect_uris: ["http://127.0.0.1:8989/oauth/callback"],
      token_endpoint_auth_method: "none",
      scope: "internal",
    },
    redirectUrl: "http://127.0.0.1:8989/oauth/callback",
    state: async () => payload.state,
    tokens: async () => payload.tokens,
    saveTokens: async (tokens) => {
      payload.tokens = {
        ...(payload.tokens || {}),
        ...tokens,
        refresh_token: tokens.refresh_token || payload?.tokens?.refresh_token,
      };
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

function chicagoMarketWindow() {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Chicago",
      weekday: "short",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).formatToParts(new Date()).map((part) => [part.type, part.value])
  );
  const weekday = parts.weekday;
  const hour = Number(parts.hour);
  const minute = Number(parts.minute);
  const marketDay = ["Mon", "Tue", "Wed", "Thu", "Fri"].includes(weekday);
  const minutes = hour * 60 + minute;
  return {
    open: marketDay && minutes >= 8 * 60 + 30 && minutes < 15 * 60,
    local: `${weekday} ${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")} CT`,
  };
}

function extractToolData(result) {
  if (result?.structuredContent?.data && typeof result.structuredContent.data === "object") {
    return result.structuredContent.data;
  }
  for (const item of result?.content || []) {
    if (item?.type !== "text") continue;
    try {
      const parsed = JSON.parse(item.text || "{}");
      if (parsed?.data && typeof parsed.data === "object") return parsed.data;
      if (parsed && typeof parsed === "object") return parsed;
    } catch {
      continue;
    }
  }
  return {};
}

function orderFromToolPayload(payload) {
  if (payload?.order && typeof payload.order === "object") return payload.order;
  const data = extractToolData(payload);
  if (data?.order && typeof data.order === "object") return data.order;
  return data && typeof data === "object" ? data : {};
}

async function withTimeout(promise, timeoutMs, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

async function callTool(client, name, toolArgs, timeoutMs) {
  const result = await withTimeout(
    client.request({ method: "tools/call", params: { name, arguments: toolArgs } }, CallToolResultSchema),
    timeoutMs,
    `Robinhood MCP ${name}`
  );
  if (result?.isError) {
    throw new Error(`Robinhood MCP ${name} returned isError=true`);
  }
  return result;
}

function runProcess(command, args, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: { ...process.env, ...(options.env || {}) },
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
    child.on("close", (code, signal) => {
      if (timer) clearTimeout(timer);
      resolve({ code, signal, stdout, stderr });
    });
  });
}

function parseJsonOutput(proc, label) {
  const raw = String(proc.stdout || "").trim();
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error(`${label} did not return JSON rc=${proc.code} stdout=${raw.slice(0, 1200)} stderr=${String(proc.stderr || "").slice(0, 1000)}`);
  }
}

async function runArtha(args, commandArgs, options = {}) {
  const proc = await runProcess(args.python, ["run.py", ...commandArgs], {
    cwd: args.projectDir,
    timeoutMs: options.timeoutMs || 180000,
  });
  const parsed = parseJsonOutput(proc, `run.py ${commandArgs.join(" ")}`);
  if (options.requireZero !== false && proc.code !== 0) {
    throw new Error(`run.py ${commandArgs.join(" ")} failed rc=${proc.code} stdout=${proc.stdout.slice(0, 1200)} stderr=${proc.stderr.slice(0, 1000)}`);
  }
  return { proc, parsed };
}

async function runSnapshotSync(args) {
  const proc = await runProcess("/opt/homebrew/bin/node", ["scripts/robinhood_snapshot_sync.mjs", "--market-hours-only", "--quiet"], {
    cwd: args.projectDir,
    timeoutMs: 180000,
  });
  if (proc.code !== 0) {
    throw new Error(`snapshot sync failed rc=${proc.code} stdout=${proc.stdout.slice(0, 1200)} stderr=${proc.stderr.slice(0, 1200)}`);
  }
  return proc;
}

async function acquireLock(lockFile, waitMs) {
  const staleMs = 10 * 60 * 1000;
  const started = Date.now();
  fs.mkdirSync(path.dirname(lockFile), { recursive: true });
  const lockState = () => {
    try {
      const raw = fs.readFileSync(lockFile, "utf8");
      return JSON.parse(raw);
    } catch {
      return {};
    }
  };
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
      let hasOwnerPid = false;
      try {
        const state = lockState();
        stale = Date.now() - fs.statSync(lockFile).mtimeMs > staleMs;
        hasOwnerPid = Boolean(state.pid);
        deadOwner = hasOwnerPid && !pidAlive(state.pid);
      } catch {
        stale = true;
      }
      if (deadOwner || (stale && !hasOwnerPid)) {
        try {
          fs.unlinkSync(lockFile);
          continue;
        } catch {}
      }
      if (Date.now() - started >= waitMs) {
        return null;
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
}

async function sendTelegram(args, message) {
  const proc = await runProcess("openclaw", [
    "message", "send",
    "--channel", "telegram",
    "--target", args.telegramChatId,
    "--message", message,
    "--json",
  ], { cwd: args.projectDir, timeoutMs: 20000 });
  return proc.code === 0;
}

function sanitizeError(error) {
  return String(error?.message || error || "unknown error")
    .replace(/Bearer\s+[A-Za-z0-9._~+/-]+/g, "Bearer [redacted]")
    .replace(/access_token["']?\s*:\s*["'][^"']+/g, 'access_token":"[redacted]')
    .slice(0, 1000);
}

function argsMatchForPlace(reviewArgs, placeArgs) {
  const keys = ["account_number", "symbol", "side", "type", "quantity", "dollar_amount", "limit_price", "stop_price", "time_in_force", "market_hours"];
  const mismatches = [];
  for (const key of keys) {
    const left = reviewArgs?.[key] == null ? "" : String(reviewArgs[key]);
    const right = placeArgs?.[key] == null ? "" : String(placeArgs[key]);
    if (key === "ref_id") continue;
    if (left !== right) mismatches.push(`${key}: review=${left || "-"} place=${right || "-"}`);
  }
  return { ok: mismatches.length === 0, mismatches };
}

function actionLabel(op) {
  const review = op?.review_mcp_args || op?.place_mcp_args || {};
  const symbol = String(review.symbol || op?.ticker || "?").toUpperCase();
  const side = String(op?.side || review.side || "?").toUpperCase();
  const amount = review.dollar_amount ? `$${review.dollar_amount}` : (review.quantity ? `${review.quantity} shares` : "order");
  return { symbol, side, amount };
}

function blockMessage(label, actionId, reason) {
  return [
    "⚠️ attention — Artha auto-trade blocked before any order.",
    `• Action: ${label.side} ${label.symbol} ${label.amount}`,
    `• Reason: ${String(reason || "unknown block").slice(0, 600)}`,
    `• Artha action: ${actionId}`,
    "👉 You: nothing needed.",
  ].join("\n");
}

function successMessage(label, actionId, placeResponse, submission) {
  const order = orderFromToolPayload(placeResponse);
  const state = order.state || submission?.recorded_submission?.status || submission?.status || "submitted";
  const orderId = order.id || submission?.recorded_submission?.broker_order_id || "-";
  const avg = order.average_price || order.price || order.executed_price || "-";
  const qty = order.quantity || order.cumulative_quantity || order.executed_quantity || "-";
  return [
    `✅ done — ${label.symbol} ${label.side.toLowerCase()} order placed through Artha.`,
    `• Amount: ${label.amount}`,
    `• State: ${state}`,
    `• Filled qty/avg: ${qty} @ ${avg}`,
    `• Broker order: ${orderId}`,
    `• Artha action: ${actionId}`,
    "👉 You: nothing needed.",
  ].join("\n");
}

async function processOperation(args, client, op) {
  const actionId = String(op.action_id || "");
  const label = actionLabel(op);
  if (!op?.success) {
    return { status: "blocked", action_id: actionId, reason: op?.message || op?.operation || "operation not successful" };
  }
  if (op.operation !== "auto_tradability_review_then_place_equity_order") {
    return { status: "blocked", action_id: actionId, reason: `unsupported operation ${op.operation}` };
  }
  if (!actionId || !op.tradability_mcp_args || !op.review_mcp_args) {
    return { status: "blocked", action_id: actionId, reason: "missing action_id, tradability args, or review args" };
  }
  const symbol = String(op.review_mcp_args.symbol || (op.tradability_mcp_args.symbols || [])[0] || "").toUpperCase();
  if (!symbol) {
    return { status: "blocked", action_id: actionId, reason: "operation missing symbol" };
  }

  const prefix = path.join(args.tmpDir, `artha_auto_trade_${actionId}`);
  const quote = await callTool(client, "get_equity_quotes", { symbols: [symbol] }, args.timeoutMs);
  const tradability = await callTool(client, "get_equity_tradability", op.tradability_mcp_args, args.timeoutMs);
  const review = await callTool(client, "review_equity_order", op.review_mcp_args, args.timeoutMs);
  const quoteFile = `${prefix}_quote.json`;
  const tradabilityFile = `${prefix}_tradability.json`;
  const reviewFile = `${prefix}_review.json`;
  atomicWriteJson(quoteFile, quote);
  atomicWriteJson(tradabilityFile, tradability);
  atomicWriteJson(reviewFile, review);

  const clearance = await runArtha(
    args,
    ["robinhood-auto-buy-agentic-clearance", actionId, "--quote-file", quoteFile, "--tradability-file", tradabilityFile, "--review-file", reviewFile],
    { requireZero: false, timeoutMs: 300000 }
  );
  if (!clearance.parsed?.allow_place) {
    const reason = clearance.parsed?.message || clearance.parsed?.agentic_execution_officer?.reason || clearance.parsed?.status || "agentic clearance blocked";
    return { status: "blocked", action_id: actionId, label, reason };
  }

  const placeOp1 = (await runArtha(args, ["robinhood-auto-buy-action", actionId], { timeoutMs: 120000 })).parsed;
  if (placeOp1.operation !== "tradability_then_review_then_place_equity_order") {
    return { status: "blocked", action_id: actionId, label, reason: `expected place operation, got ${placeOp1.operation}` };
  }

  const secondTradability = await callTool(client, "get_equity_tradability", placeOp1.tradability_mcp_args, args.timeoutMs);
  const secondReview = await callTool(client, "review_equity_order", placeOp1.review_mcp_args, args.timeoutMs);
  const secondTradabilityFile = `${prefix}_second_tradability.json`;
  const secondReviewFile = `${prefix}_second_review.json`;
  atomicWriteJson(secondTradabilityFile, secondTradability);
  atomicWriteJson(secondReviewFile, secondReview);

  const recorded = await runArtha(
    args,
    ["robinhood-record-review", actionId, "--tradability-file", secondTradabilityFile, "--review-file", secondReviewFile],
    { requireZero: false, timeoutMs: 120000 }
  );
  if (recorded.parsed?.status !== "review_clear") {
    const reason = recorded.parsed?.message || recorded.parsed?.review_gate?.reasons?.join("; ") || "second Robinhood review blocked";
    return { status: "blocked", action_id: actionId, label, reason };
  }

  const final = await runArtha(args, ["robinhood-final-clearance", actionId], { requireZero: false, timeoutMs: 180000 });
  if (!final.parsed?.allow_place) {
    return { status: "blocked", action_id: actionId, label, reason: final.parsed?.message || "final clearance blocked" };
  }

  const placeOp2 = (await runArtha(args, ["robinhood-auto-buy-action", actionId], { timeoutMs: 120000 })).parsed;
  if (!placeOp2.place_mcp_args) {
    return { status: "blocked", action_id: actionId, label, reason: "final place args missing" };
  }
  const match = argsMatchForPlace(placeOp1.review_mcp_args, placeOp2.place_mcp_args);
  if (!match.ok) {
    return { status: "blocked", action_id: actionId, label, reason: `place args changed after final review: ${match.mismatches.join("; ")}` };
  }

  const place = await callTool(client, "place_equity_order", placeOp2.place_mcp_args, args.timeoutMs);
  const placeFile = `${prefix}_place.json`;
  atomicWriteJson(placeFile, place);
  const submission = await runArtha(args, ["robinhood-record-submission", actionId, "--file", placeFile], {
    requireZero: false,
    timeoutMs: 120000,
  });
  await runSnapshotSync(args);
  return {
    status: submission.parsed?.status === "PASS" ? "placed" : "blocked",
    action_id: actionId,
    label,
    place_response: place,
    submission: submission.parsed,
    reason: submission.parsed?.message || submission.parsed?.status,
  };
}

async function drainQueue(args) {
  const market = chicagoMarketWindow();
  if (args.marketHoursOnly && !market.open) {
    return { success: true, status: "AUTO_BUY_MARKET_CLOSED", market: market.local, results: [] };
  }

  const status = (await runArtha(args, ["robinhood-auto-buy-queue-status"], { timeoutMs: 60000 })).parsed;
  if (!Number(status.operation_count || 0)) {
    return { success: true, status: "AUTO_BUY_IDLE", results: [], queue: status };
  }

  await runSnapshotSync(args);
  const batch = (await runArtha(args, ["robinhood-auto-buy-action"], { timeoutMs: 120000 })).parsed;
  const operations = Array.isArray(batch.operations) ? batch.operations : [];
  if (!operations.length) {
    return { success: true, status: "AUTO_BUY_IDLE", results: [], queue: batch };
  }

  const config = resolveConfig(args);
  const oauthProvider = createOAuthProvider(config.oauthFile);
  const client = new Client({ name: "artha-robinhood-auto-trade-runner", version: "1.0.0" }, {});
  const transport = new StreamableHTTPClientTransport(new URL(config.mcpUrl), { authProvider: oauthProvider });
  const results = [];
  try {
    await withTimeout(client.connect(transport), args.timeoutMs, "Robinhood MCP connect");
    for (const op of operations.slice(0, Math.max(1, args.maxActions))) {
      const result = await processOperation(args, client, op);
      results.push(result);
      if (result.status === "blocked") {
        await sendTelegram(args, blockMessage(result.label || actionLabel(op), result.action_id || op.action_id, result.reason));
      } else if (result.status === "placed") {
        await sendTelegram(args, successMessage(result.label || actionLabel(op), result.action_id, result.place_response, result.submission));
      }
    }
  } finally {
    try { await client.close(); } catch {}
  }
  const placed = results.filter((item) => item.status === "placed").length;
  const blocked = results.filter((item) => item.status === "blocked").length;
  return {
    success: true,
    status: placed ? "AUTO_BUY_PLACED" : (blocked ? "AUTO_BUY_BLOCKED" : "AUTO_BUY_IDLE"),
    results,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const release = await acquireLock(args.lockFile, args.lockWaitMs);
  if (!release) {
    if (!args.quiet) {
      console.log(JSON.stringify({
        success: true,
        status: "AUTO_BUY_ALREADY_RUNNING",
        message: "Another Artha Robinhood runner still owns the lock; skipping this tick.",
      }, null, 2));
    }
    return;
  }
  try {
    const result = await drainQueue(args);
    if (!args.quiet) console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    const message = sanitizeError(error);
    await sendTelegram(args, [
      "⚠️ attention — Artha Robinhood runner failed; no order was placed.",
      `• Error: ${message}`,
      "• Impact: queued auto-trades stay blocked until the runner succeeds.",
      "👉 You: nothing needed.",
    ].join("\n"));
    console.error(JSON.stringify({ success: false, status: "FAIL", error: message }, null, 2));
    process.exit(1);
  } finally {
    release();
  }
}

main();
