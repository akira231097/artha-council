#!/usr/bin/env node
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";

const DEFAULT_STATE_DIR = path.join(os.homedir(), ".openclaw");
const DEFAULT_PROJECT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_TMP_DIR = process.env.ARTHA_OPENCLAW_TMP_DIR || path.join(DEFAULT_STATE_DIR, "workspace", "tmp");
const DEFAULT_CHAT_ID = process.env.TELEGRAM_CHAT_ID || "";
const DEFAULT_LOCK_FILE = "/tmp/artha-robinhood-snapshot-sync.lock";
const DEFAULT_RUNNER_LOCK_FILE = `${DEFAULT_LOCK_FILE}.runner`;
const DEFAULT_SDK_TIMEOUT_MS = 120000;

function parseArgs(argv) {
  const args = {
    projectDir: DEFAULT_PROJECT_DIR,
    stateDir: process.env.OPENCLAW_STATE_DIR || DEFAULT_STATE_DIR,
    openclawConfig: process.env.OPENCLAW_CONFIG_PATH || path.join(DEFAULT_STATE_DIR, "openclaw.json"),
    handoffFile: path.join(DEFAULT_TMP_DIR, "artha_robinhood_snapshot.json"),
    stateFile: path.join(DEFAULT_PROJECT_DIR, "data", "robinhood", "snapshot_sync_state.json"),
    python: null,
    lockFile: DEFAULT_LOCK_FILE,
    runnerLockFile: DEFAULT_RUNNER_LOCK_FILE,
    runnerLockWaitMs: 90000,
    maxHandoffAgeMinutes: 10,
    marketHoursOnly: false,
    telegramChatId: DEFAULT_CHAT_ID,
    failureAlertAfter: 2,
    failureAlertCooldownMinutes: 30,
    timeoutMs: DEFAULT_SDK_TIMEOUT_MS,
    skipImport: false,
    quiet: false,
    oauthFile: null,
    mcpUrl: null,
    accountNumber: null,
    accountSuffix: null,
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
    else if (arg === "--handoff-file") args.handoffFile = next();
    else if (arg === "--state-file") args.stateFile = next();
    else if (arg === "--python") args.python = next();
    else if (arg === "--lock-file") args.lockFile = next();
    else if (arg === "--runner-lock-file") args.runnerLockFile = next();
    else if (arg === "--runner-lock-wait-ms") args.runnerLockWaitMs = Number(next());
    else if (arg === "--max-handoff-age-minutes") args.maxHandoffAgeMinutes = Number(next());
    else if (arg === "--telegram-chat-id") args.telegramChatId = next();
    else if (arg === "--failure-alert-after") args.failureAlertAfter = Number(next());
    else if (arg === "--failure-alert-cooldown-minutes") args.failureAlertCooldownMinutes = Number(next());
    else if (arg === "--timeout-ms") args.timeoutMs = Number(next());
    else if (arg === "--oauth-file") args.oauthFile = next();
    else if (arg === "--mcp-url") args.mcpUrl = next();
    else if (arg === "--account-number") args.accountNumber = next();
    else if (arg === "--account-suffix") args.accountSuffix = next();
    else if (arg === "--market-hours-only") args.marketHoursOnly = true;
    else if (arg === "--skip-import") args.skipImport = true;
    else if (arg === "--quiet") args.quiet = true;
    else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  args.projectDir = path.resolve(args.projectDir);
  args.openclawConfig = expandHome(args.openclawConfig);
  args.stateDir = expandHome(args.stateDir);
  args.handoffFile = path.resolve(expandHome(args.handoffFile));
  args.stateFile = path.resolve(expandHome(args.stateFile));
  args.python = args.python || defaultPython(args.projectDir);
  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/robinhood_snapshot_sync.mjs [options]

Read-only deterministic Robinhood MCP snapshot sync for Artha.

Options:
  --market-hours-only              Exit 0 without broker calls outside Mon-Fri 08:00-15:59 America/Chicago
  --skip-import                    Fetch and write handoff only; do not run Artha importer
  --handoff-file <path>            Snapshot handoff path
  --state-file <path>              Reliability state path
  --runner-lock-file <path>        Full-run lock path to prevent concurrent handoff writes
  --account-number <number>        Expected Robinhood account number
  --account-suffix <last4>         Expected account suffix when account number is not supplied
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

function defaultPython(projectDir) {
  const venvPython = path.join(projectDir, ".venv", "bin", "python");
  return fs.existsSync(venvPython) ? venvPython : "python3";
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
  const accountSuffix = args.accountSuffix || (accountNumber ? accountNumber.slice(-4) : "");
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
          const payload = readJson(file);
          return Boolean(payload?.tokens?.access_token);
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
      client_name: "Artha Robinhood Snapshot Sync",
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

function chicagoMarketWindowOpen() {
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
  return {
    open: marketDay && hour >= 8 && hour <= 15,
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

function accountNumberOf(account) {
  return String(account?.account_number || account?.rhs_account_number || "").trim();
}

function selectAccount(accountsData, expectedNumber, expectedSuffix) {
  const accounts = Array.isArray(accountsData?.accounts) ? accountsData.accounts : [];
  const candidates = accounts.filter((account) => account && typeof account === "object");
  const exact = expectedNumber
    ? candidates.find((account) => accountNumberOf(account) === expectedNumber)
    : null;
  const suffix = expectedSuffix
    ? candidates.find((account) => accountNumberOf(account).endsWith(expectedSuffix) && account.agentic_allowed === true)
    : null;
  const agentic = candidates.find((account) => account.agentic_allowed === true);
  const selected = exact || suffix || agentic;
  if (!selected) {
    throw new Error(`Could not find an agentic Robinhood account ending ${expectedSuffix || "unknown"}.`);
  }
  if (selected.agentic_allowed !== true) {
    throw new Error("Selected Robinhood account is not agentic_allowed=true.");
  }
  const number = accountNumberOf(selected);
  if (!number) throw new Error("Selected Robinhood account is missing account_number.");
  if (expectedNumber && number !== expectedNumber) {
    throw new Error(`Selected Robinhood account does not match configured account ending ${expectedNumber.slice(-4)}.`);
  }
  if (expectedSuffix && !number.endsWith(expectedSuffix)) {
    throw new Error(`Selected Robinhood account does not end with ${expectedSuffix}.`);
  }
  return selected;
}

function maskAccount(accountNumber) {
  const value = String(accountNumber || "");
  if (value.length <= 4) return "****";
  return `${"*".repeat(Math.max(0, value.length - 4))}${value.slice(-4)}`;
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

export function summarizeImporterFailure(proc, parsed) {
  if (!parsed || typeof parsed !== "object") {
    const stderr = String(proc?.stderr || "").trim().slice(0, 500);
    const stdout = String(proc?.stdout || "").trim().slice(0, 500);
    return `rc=${proc?.code ?? "unknown"}; importer output was not valid JSON; stderr=${stderr || "empty"}; stdout=${stdout || "empty"}`;
  }
  const validation = parsed.validation && typeof parsed.validation === "object" ? parsed.validation : {};
  const snapshot = parsed.snapshot && typeof parsed.snapshot === "object" ? parsed.snapshot : {};
  const sync = parsed.sync && typeof parsed.sync === "object" ? parsed.sync : {};
  const orderSync = sync.order_sync && typeof sync.order_sync === "object" ? sync.order_sync : {};
  const unresolved = Array.isArray(sync.unresolved)
    ? sync.unresolved.slice(0, 6).map((item) => {
      const ticker = String(item?.ticker || "unknown");
      const reason = String(item?.reason || "unresolved broker position").replace(/\s+/g, " ").slice(0, 180);
      return `${ticker}: ${reason}`;
    })
    : [];
  const parts = [
    `rc=${proc?.code ?? "unknown"}`,
    `validation=${validation.status || "UNKNOWN"}`,
    `snapshot=${snapshot.status || "UNKNOWN"}`,
    `sync=${sync.status || "UNKNOWN"}`,
    `order_sync=${orderSync.status || "UNKNOWN"}`,
  ];
  if (sync.reason) parts.push(`sync_reason=${String(sync.reason).replace(/\s+/g, " ").slice(0, 300)}`);
  if (unresolved.length) parts.push(`unresolved=[${unresolved.join(" | ")}]`);
  if (Array.isArray(orderSync.unmatched) && orderSync.unmatched.length) {
    parts.push(`unmatched_orders=${orderSync.unmatched.length}`);
  }
  return parts.join("; ");
}

async function runImporter(args, runId, generatedAt) {
  const proc = await runProcess(
    args.python,
    [
      "run.py",
      "robinhood-snapshot-import",
      "--strict",
      "--expect-run-id",
      runId,
      "--min-generated-at",
      generatedAt,
      "--max-handoff-age-minutes",
      String(args.maxHandoffAgeMinutes),
      "--lock-file",
      args.lockFile,
      "--file",
      args.handoffFile,
    ],
    { cwd: args.projectDir, timeoutMs: 90000 }
  );
  let parsed = null;
  try {
    parsed = JSON.parse(proc.stdout);
  } catch {
    parsed = null;
  }
  if (proc.code !== 0 || !parsed?.success) {
    throw new Error(`Artha strict importer failed: ${summarizeImporterFailure(proc, parsed)}`);
  }
  return parsed;
}

function loadSyncState(stateFile) {
  try {
    return fs.existsSync(stateFile) ? readJson(stateFile) : {};
  } catch {
    return {};
  }
}

function updateState(stateFile, patch) {
  const current = loadSyncState(stateFile);
  const next = { ...current, ...patch, updated_at: new Date().toISOString() };
  atomicWriteJson(stateFile, next);
  return next;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function acquireRunnerLock(lockFile, waitMs) {
  const staleMs = 10 * 60 * 1000;
  const started = Date.now();
  fs.mkdirSync(path.dirname(lockFile), { recursive: true });

  while (true) {
    try {
      const fd = fs.openSync(lockFile, "wx", 0o600);
      fs.writeFileSync(fd, `${JSON.stringify({ pid: process.pid, acquired_at: new Date().toISOString() })}\n`, "utf8");
      return () => {
        try {
          fs.closeSync(fd);
        } catch {
          // Best-effort cleanup.
        }
        try {
          fs.unlinkSync(lockFile);
        } catch {
          // Best-effort cleanup.
        }
      };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      let stale = false;
      try {
        stale = Date.now() - fs.statSync(lockFile).mtimeMs > staleMs;
      } catch {
        stale = true;
      }
      if (stale) {
        try {
          fs.unlinkSync(lockFile);
          continue;
        } catch {
          // Another runner may have removed it first.
        }
      }
      if (Date.now() - started >= waitMs) {
        throw new Error(`Robinhood snapshot sync runner is already active; timed out waiting for ${lockFile}`);
      }
      await sleep(250);
    }
  }
}

function shouldAlertFailure(state, failureAlertAfter, cooldownMinutes) {
  const failures = Number(state.consecutive_failures || 0);
  if (failures < failureAlertAfter) return false;
  const lastAlert = state.last_alert_at ? Date.parse(state.last_alert_at) : 0;
  return !lastAlert || Date.now() - lastAlert >= cooldownMinutes * 60 * 1000;
}

function describeError(error) {
  const parts = [];
  const push = (label, value) => {
    if (value !== undefined && value !== null && String(value).trim()) {
      parts.push(`${label}=${String(value).trim()}`);
    }
  };

  push("name", error?.name);
  push("message", error?.message || error || "unknown error");
  push("code", error?.code);

  const cause = error?.cause;
  if (cause) {
    push("causeName", cause.name);
    push("causeCode", cause.code);
    push("causeMessage", cause.message);
    if (Array.isArray(cause.errors)) {
      for (const [idx, item] of cause.errors.slice(0, 4).entries()) {
        const detail = [
          item?.code,
          item?.syscall,
          item?.address ? `${item.address}:${item.port || ""}` : "",
          item?.message,
        ].filter(Boolean).join(" ");
        push(`causeError${idx + 1}`, detail);
      }
    }
  }

  const raw = parts.join(" | ");
  const hasRouteFailure = /EHOSTUNREACH|ENETUNREACH|No route to host/i.test(raw);
  const hint = hasRouteFailure
    ? " | networkHint=IPv4 route/router unreachable; check DHCP/default gateway before Robinhood OAuth/import"
    : "";

  return `${raw}${hint}`;
}

function sanitizeError(error) {
  return describeError(error)
    .replace(/Bearer\s+[A-Za-z0-9._~+/-]+/g, "Bearer [redacted]")
    .replace(/access_token["']?\s*:\s*["'][^"']+/g, 'access_token":"[redacted]')
    .slice(0, 1800);
}

async function sendFailureAlert(args, state, errorMessage) {
  const message = [
    "ARTHA SNAPSHOT SYNC FAILURE",
    "---------------------------",
    `Deterministic Robinhood snapshot sync failed ${state.consecutive_failures} times.`,
    `Error: ${errorMessage.slice(0, 900)}`,
    `Last success: ${state.last_success_at || "unknown"}`,
    "Impact: broker-dependent Artha Review/Place actions stay blocked if the snapshot becomes stale.",
  ].join("\n");
  const proc = await runProcess(
    "openclaw",
    ["message", "send", "--channel", "telegram", "--target", args.telegramChatId, "--message", message, "--json"],
    { cwd: args.projectDir, timeoutMs: 20000 }
  );
  return proc.code === 0;
}

async function syncSnapshot(args) {
  const config = resolveConfig(args);
  const syncStartedAt = new Date().toISOString();
  const runId = `artha-rh-direct-${syncStartedAt.replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z")}-${randomBytes(3).toString("hex")}`;
  const oauthProvider = createOAuthProvider(config.oauthFile);
  const client = new Client({ name: "artha-robinhood-snapshot-sync", version: "1.0.0" }, {});
  const transport = new StreamableHTTPClientTransport(new URL(config.mcpUrl), { authProvider: oauthProvider });
  const steps = [];
  try {
    await withTimeout(client.connect(transport), args.timeoutMs, "Robinhood MCP connect");
    steps.push("connect");
    const accountsResponse = await callTool(client, "get_accounts", {}, args.timeoutMs);
    steps.push("get_accounts");
    const accountsData = extractToolData(accountsResponse);
    const account = selectAccount(accountsData, config.accountNumber, config.accountSuffix);
    const accountNumber = accountNumberOf(account);
    const portfolioResponse = await callTool(client, "get_portfolio", { account_number: accountNumber }, args.timeoutMs);
    steps.push("get_portfolio");
    const positionsResponse = await callTool(client, "get_equity_positions", { account_number: accountNumber }, args.timeoutMs);
    steps.push("get_equity_positions");
    const ordersResponse = await callTool(client, "get_equity_orders", { account_number: accountNumber }, args.timeoutMs);
    steps.push("get_equity_orders");

    const portfolioData = extractToolData(portfolioResponse);
    const positionsData = extractToolData(positionsResponse);
    const ordersData = extractToolData(ordersResponse);
    const envelope = {
      snapshot_envelope_version: 2,
      run_id: runId,
      generated_at: syncStartedAt,
      source: "robinhood_mcp",
      sync_client: "scripts/robinhood_snapshot_sync.mjs",
      account,
      accounts: accountsData.accounts || [],
      portfolio: portfolioData.portfolio || portfolioData,
      positions: positionsData.positions || positionsData.results || [],
      orders: ordersData.orders || ordersData.results || [],
      accounts_response: accountsResponse,
      portfolio_response: portfolioResponse,
      positions_response: positionsResponse,
      orders_response: ordersResponse,
    };
    atomicWriteJson(args.handoffFile, envelope);
    steps.push("write_handoff");

    const imported = args.skipImport ? { success: true, skipped: true } : await runImporter(args, runId, syncStartedAt);
    steps.push(args.skipImport ? "skip_import" : "strict_import");
    const result = {
      success: true,
      status: "PASS",
      run_id: runId,
      generated_at: syncStartedAt,
      account: maskAccount(accountNumber),
      handoff_file: args.handoffFile,
      steps,
      position_count: Array.isArray(envelope.positions) ? envelope.positions.length : 0,
      order_count: Array.isArray(envelope.orders) ? envelope.orders.length : 0,
      imported,
    };
    updateState(args.stateFile, {
      last_status: "PASS",
      last_success_at: new Date().toISOString(),
      last_run_id: runId,
      last_position_count: result.position_count,
      consecutive_failures: 0,
      last_error: null,
    });
    return result;
  } finally {
    try {
      await client.close();
    } catch {
      // Best-effort close only.
    }
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.marketHoursOnly) {
    const market = chicagoMarketWindowOpen();
    if (!market.open) {
      const skipped = updateState(args.stateFile, {
        last_status: "SKIPPED_MARKET_CLOSED",
        last_skip_at: new Date().toISOString(),
        last_skip_reason: `Outside sync window (${market.local})`,
      });
      if (!args.quiet) {
        console.log(JSON.stringify({ success: true, status: "SKIPPED", reason: skipped.last_skip_reason }, null, 2));
      }
      return;
    }
  }

  try {
    const started = Date.now();
    const releaseRunnerLock = await acquireRunnerLock(args.runnerLockFile, args.runnerLockWaitMs);
    let result;
    try {
      result = await syncSnapshot(args);
    } finally {
      releaseRunnerLock();
    }
    result.duration_ms = Date.now() - started;
    if (!args.quiet) console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    const errorMessage = sanitizeError(error);
    const previous = loadSyncState(args.stateFile);
    const state = updateState(args.stateFile, {
      last_status: "FAIL",
      last_failure_at: new Date().toISOString(),
      consecutive_failures: Number(previous.consecutive_failures || 0) + 1,
      last_error: errorMessage,
    });
    let alertSent = false;
    if (shouldAlertFailure(state, args.failureAlertAfter, args.failureAlertCooldownMinutes)) {
      alertSent = await sendFailureAlert(args, state, errorMessage);
      if (alertSent) {
        updateState(args.stateFile, { last_alert_at: new Date().toISOString() });
      }
    }
    console.error(JSON.stringify({ success: false, status: "FAIL", error: errorMessage, alert_sent: alertSent }, null, 2));
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main();
}
