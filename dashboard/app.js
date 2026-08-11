"use strict";

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[char]));
const money = (value, digits = 2) => value == null || Number.isNaN(Number(value))
  ? "—"
  : `${Number(value) < 0 ? "-" : ""}$${Math.abs(Number(value)).toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits})}`;
const signedMoney = (value) => value == null || Number.isNaN(Number(value)) ? "—" : `${Number(value) >= 0 ? "+" : "-"}$${Math.abs(Number(value)).toFixed(2)}`;
const pct = (value, signed = true, digits = 2) => value == null || Number.isNaN(Number(value))
  ? "—"
  : `${signed && Number(value) > 0 ? "+" : ""}${Number(value).toFixed(digits)}%`;
const number = (value, digits = 0) => value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toLocaleString("en-US", {maximumFractionDigits: digits});
const toneForNumber = (value) => Number(value) > 0.001 ? "positive" : Number(value) < -0.001 ? "negative" : "neutral";
const dateLabel = (value) => {
  const stamp = Date.parse(value);
  return Number.isNaN(stamp) ? "Unknown" : new Date(stamp).toLocaleString([], {month: "short", day: "numeric", hour: "numeric", minute: "2-digit"});
};
const compactReason = (value) => String(value || "").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());

let DATA = null;
let lastSuccess = 0;
let chartRange = "1M";
let positionSort = "value";
let decisionFilter = "all";

function verdictTone(verdict) {
  const value = String(verdict || "").toUpperCase();
  if (["BUY", "STARTER", "STARTER_BUY", "TACTICAL_BUY", "ACCUMULATE", "ADD"].includes(value)) return "buy";
  if (["DEFER", "WATCH", "WAIT"].includes(value)) return "wait";
  if (["AVOID", "SELL", "EXIT"].includes(value)) return "avoid";
  return "hold";
}

function metricCard(name, value, detail, tone = "neutral") {
  return `<article class="metric-card">
    <div class="metric-name">${esc(name)}</div>
    <strong class="metric-value ${esc(tone)}">${esc(value)}</strong>
    <span class="metric-detail">${esc(detail)}</span>
  </article>`;
}

function statusCopy() {
  const alarms = DATA?.alarms || [];
  const critical = alarms.filter((item) => item.severity === "critical").length;
  const warnings = alarms.filter((item) => item.severity === "warning").length;
  if (critical) return {tone: "bad", label: `${critical} critical issue${critical === 1 ? "" : "s"}`};
  if (warnings) return {tone: "warn", label: `${warnings} warning${warnings === 1 ? "" : "s"}`};
  return {tone: "good", label: "All required systems passed"};
}

function activateView(view, updateHash = true) {
  const allowed = ["overview", "portfolio", "decisions", "process", "learning", "health"];
  const target = allowed.includes(view) ? view : "overview";
  document.querySelectorAll(".view-tab").forEach((button) => {
    const active = button.dataset.view === target;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".view").forEach((section) => {
    const active = section.id === `view-${target}`;
    section.hidden = !active;
    section.classList.toggle("active", active);
  });
  if (updateHash) history.replaceState(null, "", `#${target}`);
  window.scrollTo({top: 0, behavior: "instant"});
  if (target === "overview" && DATA) requestAnimationFrame(renderChart);
}

document.querySelectorAll(".view-tab").forEach((button) => {
  button.addEventListener("click", () => activateView(button.dataset.view));
});
window.addEventListener("hashchange", () => activateView(location.hash.slice(1), false));
window.addEventListener("resize", () => {
  if (DATA && !$("view-overview").hidden) renderChart();
});

async function refresh() {
  try {
    const response = await fetch("/api/dashboard", {cache: "no-store"});
    if (response.status === 401) throw new Error("Your saved dashboard access has expired. Open the private dashboard link again.");
    if (!response.ok) throw new Error(`Dashboard returned ${response.status}`);
    DATA = await response.json();
    if (DATA.schema_version !== 2) throw new Error("The dashboard service is still running an older data version.");
    lastSuccess = Date.now();
    $("connection-error").classList.remove("visible");
    render();
  } catch (error) {
    $("connection-error").textContent = `${error.message} The last successful information remains on screen.`;
    $("connection-error").classList.add("visible");
    $("live-dot").className = "live-dot bad";
    $("header-status").textContent = "Dashboard data unavailable";
  } finally {
    const delay = DATA?.market?.open ? 15000 : 60000;
    window.setTimeout(refresh, delay);
  }
}

function render() {
  renderHeader();
  renderOverview();
  renderPortfolio();
  renderDecisions();
  renderProcess();
  renderLearning();
  renderHealth();
}

function renderHeader() {
  const state = statusCopy();
  $("live-dot").className = `live-dot ${state.tone}`;
  $("header-status").textContent = state.label;
  $("header-market").textContent = `${DATA.market.phase} · ${DATA.market.ct_time}`;
  $("header-updated").textContent = `Updated ${new Date(DATA.generated_at).toLocaleTimeString([], {hour: "numeric", minute: "2-digit", second: "2-digit"})}`;
  $("footer-refresh").textContent = DATA.market.open ? "Refreshes every 15 seconds" : "Refreshes every minute";
}

function renderOverview() {
  const v = DATA.vitals;
  const p = DATA.performance;
  const critical = DATA.alarms.filter((item) => item.severity === "critical");
  const warnings = DATA.alarms.filter((item) => item.severity === "warning");
  if (critical.length) {
    $("overview-title").textContent = "ARTHA needs attention";
    $("overview-summary").textContent = "A required protection or data service is not currently proven. Trading gates remain fail-closed.";
  } else if (warnings.length) {
    $("overview-title").textContent = "ARTHA is running with a warning";
    $("overview-summary").textContent = "Core monitoring is active, but one recent check or trade needs inspection below.";
  } else {
    $("overview-title").textContent = "ARTHA is operating normally";
    $("overview-summary").textContent = `${p.headline || "Portfolio state is current"} ${v.position_count} holdings are under sell-side monitoring.`;
  }
  $("overview-stamp").innerHTML = `<strong>${esc(v.regime.replaceAll("_", " "))}</strong><br>${esc(DATA.market.phase)} · ${esc(v.broker_source)}`;
  renderAlarms();

  $("overview-metrics").innerHTML = [
    metricCard("Account value", money(v.total_value), `${money(v.cash)} cash · ${money(v.invested)} invested`),
    metricCard("Since funding", `${signedMoney(v.pnl_total)} (${pct(v.pnl_total_pct)})`, `Based on ${money(v.base)} contributed`, toneForNumber(v.pnl_total)),
    metricCard("Since previous saved day", `${signedMoney(v.pnl_today)} (${pct(v.pnl_today_pct)})`, v.pnl_baseline_label || "Previous snapshot", toneForNumber(v.pnl_today)),
    metricCard("Positions", `${v.position_count} / ${DATA.system.policy.max_positions}`, `${v.position_slots} slots remain`),
    metricCard("Buy capacity now", money(v.deployable_now), `${money(v.exposure_headroom)} below 90% ceiling`, v.deployable_now >= 10 ? "positive" : "attention")
  ].join("");
  $("money-source").textContent = `Value source: ${v.broker_source}. Last broker snapshot ${v.broker_age_minutes == null ? "unknown" : `${number(v.broker_age_minutes, 0)} minutes ago`}.`;

  const investedWidth = Math.max(0, Math.min(100, Number(v.invested_pct || 0)));
  const bufferWidth = Math.max(0, Math.min(100 - investedWidth, 90 - investedWidth));
  $("capital-invested").style.width = `${investedWidth}%`;
  $("capital-buffer").style.left = `${investedWidth}%`;
  $("capital-buffer").style.width = `${bufferWidth}%`;
  $("capital-used-label").textContent = `${money(v.invested)} · ${pct(v.invested_pct, false)}`;
  $("capital-free-label").textContent = money(v.deployable_now);
  $("buy-capacity-note").textContent = v.buys_paused
    ? `New buys are currently constrained: ${v.buy_pause_reasons.join(" ")} Sell protection remains active.`
    : `ARTHA may automatically buy within its remaining capacity when Council, Execution Officer, and Robinhood all pass. No approval tap is required.`;
  renderChartControls();
  renderChart();
  renderActivity();
  $("latest-decision").innerHTML = DATA.decisions.length ? journeyMarkup(DATA.decisions[0], true) : `<div class="empty">No recent Council decision.</div>`;
}

function renderAlarms() {
  const alarms = DATA.alarms || [];
  const band = $("attention-band");
  const critical = alarms.some((item) => item.severity === "critical");
  const warning = alarms.some((item) => item.severity === "warning");
  band.className = `attention-band${critical ? " critical" : warning ? " warning" : ""}`;
  $("alarm-count").textContent = alarms.length;
  $("attention-title").textContent = alarms.length ? `${alarms.length} active notice${alarms.length === 1 ? "" : "s"}` : "No active alarms";
  $("alarm-list").innerHTML = alarms.length
    ? alarms.map((item) => `<div class="alarm-item">
        <span class="alarm-severity ${esc(item.severity)}">${esc(item.severity)}</span>
        <div class="alarm-copy"><b>${esc(item.title)}</b><p>${esc(item.plain)}</p></div>
        <span class="alarm-source">${esc(item.source)}</span>
      </div>`).join("")
    : `<div class="all-clear"><span class="check-symbol">✓</span><span>The latest supervisor, broker, monitoring, and execution checks contain no unresolved warning.</span></div>`;
}

function renderChartControls() {
  const ranges = ["1W", "1M", "3M", "All"];
  $("chart-range").innerHTML = ranges.map((range) => `<button class="${range === chartRange ? "selected" : ""}" data-range="${range}">${range}</button>`).join("");
  $("chart-range").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    chartRange = button.dataset.range;
    renderChartControls();
    renderChart();
  }));
}

function renderChart() {
  const target = $("equity-chart");
  const raw = DATA?.equity?.points || [];
  const spans = {"1W": 7 * 864e5, "1M": 30 * 864e5, "3M": 90 * 864e5, "All": Infinity};
  const now = Date.now();
  let points = raw.map((item) => ({time: Date.parse(item.t), value: Number(item.v)}))
    .filter((item) => Number.isFinite(item.time) && Number.isFinite(item.value))
    .filter((item) => now - item.time <= spans[chartRange]);
  if (points.length < 2) {
    target.innerHTML = `<div class="empty">Not enough saved history in this range.</div>`;
    return;
  }
  const width = Math.max(320, target.clientWidth || 720);
  const height = width < 600 ? 220 : 260;
  const pad = {left: 48, right: 16, top: 18, bottom: 30};
  const times = points.map((item) => item.time);
  const values = points.map((item) => item.value);
  const base = Number(DATA.equity.base || 350);
  let min = Math.min(...values, base);
  let max = Math.max(...values, base);
  const yPad = Math.max((max - min) * 0.14, 1.2);
  min -= yPad;
  max += yPad;
  const firstTime = Math.min(...times);
  const lastTime = Math.max(...times);
  const x = (value) => pad.left + (width - pad.left - pad.right) * (value - firstTime) / Math.max(1, lastTime - firstTime);
  const y = (value) => pad.top + (height - pad.top - pad.bottom) * (1 - (value - min) / Math.max(1, max - min));
  const path = points.map((item, index) => `${index ? "L" : "M"}${x(item.time).toFixed(2)},${y(item.value).toFixed(2)}`).join(" ");
  const gridValues = [min + yPad, (min + max) / 2, max - yPad];
  const grid = gridValues.map((value) => `<line x1="${pad.left}" x2="${width - pad.right}" y1="${y(value)}" y2="${y(value)}" stroke="#d9ddd5" stroke-dasharray="3 5"/><text x="4" y="${y(value) + 4}" fill="#777d75" font-size="10">$${value.toFixed(0)}</text>`).join("");
  const baseY = y(base);
  const baseline = baseY >= pad.top && baseY <= height - pad.bottom
    ? `<line x1="${pad.left}" x2="${width - pad.right}" y1="${baseY}" y2="${baseY}" stroke="#8c9189" stroke-width="1.5" stroke-dasharray="6 5"/>`
    : "";
  const markers = (DATA.equity.markers || []).map((marker) => {
    const time = Date.parse(marker.t);
    if (!Number.isFinite(time) || time < firstTime || time > lastTime) return "";
    let nearest = points[0];
    points.forEach((point) => { if (Math.abs(point.time - time) < Math.abs(nearest.time - time)) nearest = point; });
    const cx = x(time), cy = y(nearest.value);
    return marker.side === "buy"
      ? `<path d="M${cx - 4},${cy + 10} L${cx + 4},${cy + 10} L${cx},${cy + 2} Z" fill="#16784a"><title>Buy ${esc(marker.ticker)} ${money(marker.notional)}</title></path>`
      : `<path d="M${cx - 4},${cy - 10} L${cx + 4},${cy - 10} L${cx},${cy - 2} Z" fill="#b3362d"><title>Sell ${esc(marker.ticker)}</title></path>`;
  }).join("");
  const labels = [firstTime, (firstTime + lastTime) / 2, lastTime].map((stamp) => `<text x="${x(stamp)}" y="${height - 8}" text-anchor="middle" fill="#777d75" font-size="10">${new Date(stamp).toLocaleDateString([], {month: "short", day: "numeric"})}</text>`).join("");
  const up = values.at(-1) >= base;
  target.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="ARTHA account value history">
    ${grid}${baseline}
    <path d="${path}" fill="none" stroke="${up ? "#16784a" : "#b3362d"}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${x(points.at(-1).time)}" cy="${y(points.at(-1).value)}" r="4" fill="${up ? "#16784a" : "#b3362d"}"/>
    ${markers}${labels}
  </svg>`;
  const excluded = DATA.equity.data_quality?.excluded_isolated_outliers || 0;
  $("chart-quality").textContent = excluded ? `${excluded} isolated accounting spike hidden from this chart` : "No chart anomalies detected";
}

function renderActivity() {
  const events = DATA.activity || [];
  $("today-activity").innerHTML = events.length
    ? events.slice(0, 12).map((event) => `<div class="timeline-row">
        <time class="timeline-time">${esc(event.time)}</time>
        <span class="timeline-dot ${esc(event.kind)}"></span>
        <div class="timeline-copy"><b>${esc(event.title)}</b><span>${esc(event.detail)}</span></div>
      </div>`).join("")
    : `<div class="empty">No activity recorded today.</div>`;
}

function journeyMarkup(item, compact = false) {
  const execution = item.execution || {};
  const councilDetail = item.score == null ? `Confidence ${item.confidence}/10` : `Score ${number(item.score)} · confidence ${item.confidence}/10`;
  const orderResult = execution.status === "filled"
    ? `${money(execution.notional)} filled${execution.fill_price ? ` near ${money(execution.fill_price, 4)}` : ""}`
    : execution.label || "No order";
  return `<article class="journey-card">
    <div class="journey-head">
      <span class="ticker">${esc(item.ticker)}</span>
      <div><span class="verdict ${verdictTone(item.verdict)}">${esc(item.verdict.replaceAll("_", " "))}</span><div class="journey-meta">${esc(item.when_label)}</div></div>
      <span class="journey-meta">${esc(councilDetail)}</span>
    </div>
    <div class="journey-steps">
      <div class="journey-step"><span class="step-label">1 · Council</span><span class="step-result">${esc(item.verdict.replaceAll("_", " "))}</span><span class="step-reason">${esc(item.why || "No explanation recorded.")}</span></div>
      <div class="journey-step"><span class="step-label">2 · Execution Officer</span><span class="step-result ${esc(execution.tone || "neutral")}">${esc(execution.label || "No execution")}</span><span class="step-reason">${esc(execution.reason || "")}</span></div>
      <div class="journey-step"><span class="step-label">3 · Robinhood outcome</span><span class="step-result">${esc(orderResult)}</span><span class="step-reason">${execution.quantity ? `${number(execution.quantity, 6)} shares` : item.buy_side ? "No completed fill recorded" : "Broker was not called"}</span></div>
    </div>
    ${compact ? "" : `<div class="journey-detail">Council votes: ${item.votes.length ? item.votes.map((vote) => `${esc(vote.role)} ${esc(vote.verdict)}`).join(" · ") : "not available"}${item.evidence_count ? ` · ${number(item.evidence_count)} evidence items` : ""}</div>`}
  </article>`;
}

function renderPortfolio() {
  const v = DATA.vitals;
  const p = DATA.performance;
  $("portfolio-stamp").innerHTML = `<strong>${v.position_count} positions monitored</strong><br>${v.broker_open_orders} open Robinhood orders`;
  $("portfolio-metrics").innerHTML = [
    metricCard("Invested", money(v.invested), pct(v.invested_pct, false)),
    metricCard("Open gain / loss", signedMoney(p.unrealized_pnl), "Across current holdings", toneForNumber(p.unrealized_pnl)),
    metricCard("Realized from closed episodes", signedMoney(p.realized_pnl), `${p.episodes.closed} completed episodes`, toneForNumber(p.realized_pnl)),
    metricCard("Largest holding", DATA.positions.length ? `${DATA.positions[0].ticker} · ${money(DATA.positions[0].market_value)}` : "—", DATA.positions.length ? pct(DATA.positions[0].weight_pct, false) : "No holdings")
  ].join("");
  renderSectors();
  renderPositionSort();
  renderPositions();
  renderSellReviews();
}

function renderSectors() {
  const sectors = DATA.sectors || [];
  if (!sectors.length) {
    $("sector-layout").innerHTML = `<div class="empty">No sector classifications are available.</div>`;
    return;
  }
  const cap = Number(DATA.system.policy.max_sector_pct || 30);
  const largest = sectors[0];
  const bars = sectors.map((sector) => {
    const width = Math.max(1, Math.min(100, Number(sector.pct_nav) / cap * 100));
    return `<div class="sector-row"><span class="sector-name">${esc(sector.sector)}</span><span class="bar-track"><i class="bar-fill ${sector.pct_nav >= cap - 2 ? "near" : ""}" style="width:${width}%"></i></span><span class="bar-value">${pct(sector.pct_nav, false, 1)}</span></div>`;
  }).join("");
  $("sector-layout").innerHTML = `<div class="sector-bars">${bars}</div><aside class="sector-explain"><span class="label">Largest sector</span><strong>${pct(largest.pct_nav, false, 1)}</strong><h3>${esc(largest.sector)}</h3><p>${pct(largest.pct_invested, false, 1)} of invested money. The hard ceiling is ${pct(cap, false, 0)} of the full account. ARTHA checks the projected post-buy amount; sells are never blocked by this limit.</p></aside>`;
}

function renderPositionSort() {
  $("position-sort").querySelectorAll("button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.sort === positionSort);
    button.onclick = () => { positionSort = button.dataset.sort; renderPositionSort(); renderPositions(); };
  });
}

function renderPositions() {
  const positions = [...(DATA.positions || [])];
  if (positionSort === "pnl") positions.sort((a, b) => b.pnl_pct - a.pnl_pct);
  else if (positionSort === "risk") positions.sort((a, b) => (a.stop_dist_pct ?? 999) - (b.stop_dist_pct ?? 999));
  else positions.sort((a, b) => b.market_value - a.market_value);
  const header = `<div class="position-header"><span>Holding</span><span>Value</span><span>Price</span><span>Safety net</span><span>Sell-side state</span></div>`;
  const rows = positions.map((item) => {
    const distance = item.stop_dist_pct;
    const width = distance == null ? 0 : Math.max(2, Math.min(100, distance / 18 * 100));
    const riskClass = distance == null || distance <= 0 ? "breached" : distance < 4 ? "near" : "";
    return `<div class="position-row">
      <div class="position-main"><strong>${esc(item.ticker)}</strong><span>${esc(item.position_type.replaceAll("_", " "))} · ${esc(item.sector)}${item.days_held == null ? "" : ` · ${item.days_held} days`}</span></div>
      <div><span class="position-number">${money(item.market_value)}</span><span class="cell-sub ${toneForNumber(item.pnl)}">${signedMoney(item.pnl)} · ${pct(item.pnl_pct)}</span></div>
      <div><span class="position-number">${money(item.price, 2)}</span><span class="cell-sub">Cost ${money(item.avg_cost, 2)}</span></div>
      <div><span class="position-number">${item.active_stop ? money(item.active_stop, 2) : "Missing"}</span><span class="cell-sub">${distance == null ? "No distance available" : `${pct(distance, false, 1)} below price`}${item.stop_locked_profit ? " · profit protected" : ""}</span><div class="risk-meter"><i class="${riskClass}" style="width:${width}%"></i></div></div>
      <div><span class="status-tag ${esc(item.status_tone)}">${esc(item.status)}</span><span class="cell-sub">Last council: ${esc(item.last_sell_action)}${item.last_sell_score == null ? "" : ` · score ${number(item.last_sell_score)}`}</span></div>
    </div>`;
  }).join("");
  $("positions-table").innerHTML = positions.length ? header + rows : `<div class="empty">ARTHA currently holds no stocks.</div>`;
}

function renderSellReviews() {
  const reviews = DATA.sell_reviews || [];
  $("sell-review-list").innerHTML = reviews.length ? reviews.slice(0, 12).map((item) => `<article class="review-card">
    <div class="review-score"><strong>${esc(item.ticker)}</strong>${item.sell_score == null ? "No score" : `Sell score ${number(item.sell_score)}`}</div>
    <div class="review-copy"><b><span class="verdict ${verdictTone(item.action)}">${esc(item.action)}</span> after ${esc(item.trigger)}</b><p>${esc(item.reason)}</p><p>${esc(item.when_label)}</p></div>
    <div class="review-execution"><span class="status-tag ${item.execution?.tone === "critical" ? "critical" : item.execution?.tone === "attention" ? "attention" : ""}">${esc(item.execution?.label || "No order")}</span><p>${esc(item.execution?.reason || "")}</p></div>
  </article>`).join("") : `<div class="empty">No sell-council reviews recorded.</div>`;
}

function renderDecisions() {
  const rows = DATA.decisions || [];
  const bought = rows.filter((item) => item.execution?.status === "filled").length;
  const buyIdeas = rows.filter((item) => item.buy_side).length;
  const blocked = rows.filter((item) => ["blocked", "review_blocked", "failed"].includes(item.execution?.status)).length;
  $("decision-summary").innerHTML = [
    metricCard("Recent decisions shown", number(rows.length), "Newest first"),
    metricCard("Buy-side verdicts", number(buyIdeas), "Council liked the setup"),
    metricCard("Automatic fills", number(bought), "Robinhood completed the order", bought ? "positive" : "neutral"),
    metricCard("Execution blocks", number(blocked), "A live broker gate stopped the order", blocked ? "attention" : "positive")
  ].join("");
  $("decision-filter").querySelectorAll("button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.filter === decisionFilter);
    button.onclick = () => { decisionFilter = button.dataset.filter; renderDecisions(); };
  });
  const filtered = rows.filter((item) => {
    if (decisionFilter === "bought") return item.execution?.status === "filled";
    if (decisionFilter === "buy-side") return item.buy_side;
    if (decisionFilter === "wait") return !item.buy_side;
    if (decisionFilter === "blocked") return ["blocked", "review_blocked", "failed"].includes(item.execution?.status);
    return true;
  });
  $("decision-list").innerHTML = filtered.length ? filtered.map(decisionDetails).join("") : `<div class="empty">No decisions match this filter.</div>`;
}

function decisionDetails(item) {
  const execution = item.execution || {};
  const order = execution.status === "filled" ? `${money(execution.notional)} filled` : execution.label;
  return `<details class="decision-card">
    <summary class="decision-top">
      <div><strong class="ticker">${esc(item.ticker)}</strong><span class="verdict ${verdictTone(item.verdict)}">${esc(item.verdict.replaceAll("_", " "))}</span></div>
      <div class="decision-company"><strong>${item.score == null ? "Council decision" : `Opportunity score ${number(item.score)}/100`}</strong><span>${esc(item.when_label)} · Confidence ${item.confidence}/10</span></div>
      <div class="decision-action"><b class="${esc(execution.tone || "neutral")}">${esc(order || "No execution")}</b><span>${esc(execution.reason || "")}</span></div>
      <span class="chevron">›</span>
    </summary>
    <div class="decision-expanded">
      <div class="decision-chain">
        <div class="chain-cell"><span class="step-label">Investment decision</span><b>${esc(item.verdict.replaceAll("_", " "))}${item.score == null ? "" : ` · ${number(item.score)}/100`}</b><p>${esc(item.why)}</p></div>
        <div class="chain-cell"><span class="step-label">Execution decision</span><b class="${esc(execution.tone || "neutral")}">${esc(execution.label || "No execution")}</b><p>${esc(execution.reason || "")}</p></div>
        <div class="chain-cell"><span class="step-label">Broker result</span><b>${execution.status === "filled" ? `${money(execution.notional)} bought` : execution.status === "not_requested" ? "Robinhood not called" : compactReason(execution.status)}</b><p>${execution.fill_price ? `${number(execution.quantity, 6)} shares at ${money(execution.fill_price, 4)}` : execution.reference_price ? `Reference or cap ${money(execution.reference_price, 2)}` : "No completed fill recorded."}</p></div>
      </div>
      <div class="vote-row">${item.votes.map((vote) => `<span class="vote">${esc(vote.role)}: ${esc(vote.verdict)} · ${vote.confidence}/10</span>`).join("")}${item.evidence_count ? `<span class="vote">${item.evidence_count} evidence items</span>` : ""}</div>
    </div>
  </details>`;
}

function renderProcess() {
  const pipeline = DATA.pipeline || {};
  $("pipeline-stamp").innerHTML = pipeline.created_at ? `<strong>Latest Scout run</strong><br>${esc(dateLabel(pipeline.created_at))}` : `<strong>No recent Scout timestamp</strong>`;
  $("process-flow").innerHTML = (pipeline.stages || []).map((stage) => `<article class="process-stage ${esc(stage.status)}"><span class="stage-number">${stage.number}</span><h3>${esc(stage.name)}</h3><strong class="stage-metric">${esc(stage.metric)}</strong><p>${esc(stage.plain)}</p></article>`).join("");
  const coverage = pipeline.coverage || {};
  const council = pipeline.council || {};
  $("scan-audit").innerHTML = `<div class="audit-block">
    <div class="audit-row"><span>Market history coverage</span><b>${pct(coverage.coverage_pct, false, 1)} · ${number(coverage.usable)} / ${number(coverage.universe)}</b></div>
    <div class="audit-row"><span>Stocks ranked</span><b>${number(coverage.ranked)}</b></div>
    <div class="audit-row"><span>Lottery-like behavior excluded</span><b>${number(coverage.lottery_excluded)}</b></div>
    <div class="audit-row"><span>Scout candidate cards</span><b>${number(pipeline.routing?.total || 0)}</b></div>
    <div class="audit-row"><span>Council reviews</span><b>${number(council.count)}</b></div>
    <div class="audit-row"><span>Buy-side verdicts</span><b>${number(council.buy_side_count)}</b></div>
    <div class="audit-row"><span>Scout intelligence</span><b>${pipeline.agentic_used ? `${esc(pipeline.model || "agentic model")} · ${esc(pipeline.reasoning_effort || "deep reasoning")}` : "Deterministic fallback"}</b></div>
  </div><p class="plain-note">${esc(pipeline.summary || "")}</p>`;
  const reasons = pipeline.routing?.top_reasons || [];
  $("route-reasons").innerHTML = reasons.length ? reasons.map((item) => `<div class="reason-row"><span>${esc(compactReason(item.reason))}</span><b>${number(item.count)}</b></div>`).join("") : `<div class="empty">No recent router reasons were persisted.</div>`;
}

function renderLearning() {
  const learning = DATA.learning || {};
  const card = learning.report_card || {};
  const episode = DATA.performance.episodes || {};
  $("learning-stamp").innerHTML = `<strong>${learning.status === "PASS" ? "Feedback plumbing passed" : "Feedback needs attention"}</strong><br>Live rules do not rewrite themselves`;
  $("learning-truth").innerHTML = `<strong>What “self-learning” means here:</strong> ARTHA measures old calls, completed trades, and experimental rules. Those findings may enter later analysis only through explicit sample gates. ${esc(learning.guardrail || "")}`;
  $("report-card").innerHTML = scorePanel(card.accuracy_pct, [
    ["Correct", card.correct], ["Partly correct", card.partial], ["Wrong", card.incorrect], ["Waiting to mature", card.pending]
  ], card.definition);
  $("episode-card").innerHTML = scorePanel(episode.win_rate_pct, [
    ["Closed episodes", episode.closed], ["Winners", episode.wins], ["Losers", episode.losses], ["Average result", pct(episode.average_return_pct)]
  ], DATA.performance.sample_warning);
  $("learning-stages").innerHTML = (learning.stages || []).map((stage) => `<article class="learning-stage"><span class="learning-state ${esc(stage.status)}">${esc(stage.status)}</span><h3>${esc(stage.name)}</h3><strong>${esc(stage.metric)}</strong><p>${esc(stage.plain)}</p></article>`).join("");
  const shadow = learning.shadow || {};
  const lessons = learning.lessons || {};
  $("shadow-summary").innerHTML = `<div class="shadow-block">
    <div class="shadow-row"><span>Paper evaluations</span><b>${number(shadow.total)}</b></div>
    <div class="shadow-row"><span>Complete</span><b>${number(shadow.completed)}</b></div>
    <div class="shadow-row"><span>Still tracking</span><b>${number(shadow.tracking)}</b></div>
    <div class="shadow-row"><span>Automatic promotions</span><b>${shadow.automatic_promotion ? "Allowed" : "Not allowed"}</b></div>
    <div class="shadow-row"><span>Recorded lesson observations</span><b>${number(lessons.observational)}</b></div>
    <div class="shadow-row"><span>Manual rule reviews</span><b>${number(lessons.manual_review)}</b></div>
  </div><p class="plain-note">A promising paper result is not a live rule. It still needs independence, enough mature samples, a positive advantage, and manual approval.</p>`;
  const commits = DATA.improvements?.commits || [];
  $("improvement-list").innerHTML = commits.length ? commits.slice(0, 9).map((item) => `<div class="commit-row"><time>${esc(item.date)}</time><span>${esc(item.subject)}</span></div>`).join("") : `<div class="empty">No recent source changes found.</div>`;
}

function scorePanel(value, facts, note) {
  const normalized = Math.max(0, Math.min(100, Number(value || 0)));
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const dash = circumference * normalized / 100;
  return `<div class="score-panel"><div class="score-main"><div class="score-ring"><svg viewBox="0 0 104 104"><circle cx="52" cy="52" r="${radius}" fill="none" stroke="#eceee9" stroke-width="10"/><circle cx="52" cy="52" r="${radius}" fill="none" stroke="#16784a" stroke-width="10" stroke-linecap="round" stroke-dasharray="${dash} ${circumference - dash}"/></svg><div class="score-ring-value">${value == null ? "—" : `${number(value)}%`}</div></div><div class="score-facts">${facts.map(([name, fact]) => `<div class="score-fact"><span>${esc(name)}</span><b>${esc(fact ?? "—")}</b></div>`).join("")}</div></div><p class="score-note">${esc(note || "")}</p></div>`;
}

function renderHealth() {
  const system = DATA.system || {};
  const policy = system.policy || {};
  const passCount = system.counts?.PASS || 0;
  const total = (system.checks || []).length + (system.services || []).length;
  $("health-stamp").innerHTML = `<strong>${esc(system.status)} · ${passCount}/${total} passed</strong><br>Supervisor checked ${esc(dateLabel(system.checked_at))}`;
  $("policy-source").textContent = `Source: ${policy.source || "unknown"}`;
  const policies = [
    ["Maximum positions", number(policy.max_positions), "New positions stop at this count"],
    ["Maximum buys per day", number(policy.max_buys_per_day), "Sells do not consume this limit"],
    ["Maximum per holding", money(policy.max_position_dollars), "Cumulative position ceiling"],
    ["Maximum one auto-buy", money(policy.max_auto_order_dollars), "Per-order limit"],
    ["Daily auto-buy dollars", money(policy.max_auto_daily_dollars), "Across unattended buys"],
    ["Maximum invested", pct(policy.max_invested_pct, false, 0), "Remaining cash is the buffer"],
    ["Maximum one sector", pct(policy.max_sector_pct, false, 0), "Checked after the proposed buy"],
    ["Autonomy", policy.auto_buy_enabled && policy.auto_sell_enabled ? "Buy + sell on" : "Restricted", "Eligible trades need no user tap"]
  ];
  $("policy-grid").innerHTML = policies.map(([name, value, detail]) => `<article class="policy-item"><span>${esc(name)}</span><strong>${esc(value)}</strong><small>${esc(detail)}</small></article>`).join("");
  $("service-grid").innerHTML = (system.services || []).map((item) => `<article class="service-card"><div class="health-head"><span class="health-dot ${esc(item.status)}"></span><h3>${esc(item.name)}</h3></div><p>${esc(item.detail)}</p></article>`).join("");
  $("checks-updated").textContent = `Last full supervisor run: ${dateLabel(system.checked_at)}`;
  const grouped = Object.groupBy ? Object.groupBy(system.checks || [], (item) => item.group) : (system.checks || []).reduce((acc, item) => { (acc[item.group] ||= []).push(item); return acc; }, {});
  $("health-groups").innerHTML = Object.entries(grouped).map(([group, checks]) => `<section class="health-group"><div class="health-group-title">${esc(group)}</div>${checks.map((item) => `<div class="health-check"><div class="health-head"><span class="health-name"><i class="health-dot ${esc(item.status)}"></i>${esc(item.name)}</span><span class="health-status">${esc(item.status)}</span></div><p>${esc(item.detail)}</p></div>`).join("")}</section>`).join("");
  $("schedule-list").innerHTML = (DATA.schedule || []).map((item) => `<div class="schedule-row"><time>${esc(item.time)}</time><b>${esc(item.name)}</b><span>${esc(item.plain)}</span></div>`).join("");
  $("privacy-note").innerHTML = `<strong>Read-only boundary:</strong> ${esc(DATA.privacy?.explanation || "")} Account numbers and broker credentials are never included in this dashboard feed.`;
}

setInterval(() => {
  if (lastSuccess && Date.now() - lastSuccess > 120000) {
    $("live-dot").className = "live-dot warn";
    $("header-status").textContent = "Dashboard data is getting stale";
  }
}, 10000);

activateView(location.hash.slice(1) || "overview", false);
refresh();
