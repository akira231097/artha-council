# Sell Engine v1 — Implementation Changelog

Implemented March 2026 across 6 phases. All phases are production-ready.

---

## Phase 1: Foundation + Portfolio Management

### `artha/config.py` (MODIFIED)
Added ~60 new sell-side configuration constants after `CASH_DEPLOY_EXTREME_GREED`:
- Hard stops by position type: `SELL_HARD_STOP_BUY=-0.25`, `SELL_HARD_STOP_TACTICAL=-0.12`, `SELL_HARD_STOP_ACCUMULATE=-0.30`, `SELL_HARD_STOP_LEGACY=-0.15`
- Review schedules: `SELL_REVIEW_DAYS_TACTICAL=21`, `SELL_REVIEW_DAYS_BUY=45`, `SELL_REVIEW_DAYS_ACCUMULATE=90`
- Minimum hold periods: `SELL_MIN_HOLD_TACTICAL_DAYS=5`, `SELL_MIN_HOLD_BUY_DAYS=21`, `SELL_MIN_HOLD_ACCUMULATE_DAYS=45`
- Score thresholds: `SELL_SCORE_EXIT_TACTICAL`, `SELL_SCORE_EXIT_BUY`, `SELL_SCORE_EXIT_ACCUMULATE`, `SELL_SCORE_TRIM_TACTICAL`, `SELL_SCORE_TRIM_BUY`, `SELL_SCORE_TRIM_ACCUMULATE`, `SELL_SCORE_HOLD_THRESHOLD`
- Scale-out schedules: `SELL_SCALE_OUT_TACTICAL`, `SELL_SCALE_OUT_BUY`
- Circuit breaker: `SELL_MAX_EXITS_PER_DAY=2`, `SELL_PORTFOLIO_LOSS_PAUSE_PCT=-0.10`, `SELL_COOLDOWN_DAYS=7`
- Rotation: `SELL_ROTATE_MIN_DELTA=20`, `SELL_CONVICTION_LOCK_MIN_HEALTH=70`
- Trailing stop: `SELL_TRAILING_STOP_ATR_MULT=2.0`, `SELL_TRAILING_STOP_MIN_PCT=0.06`

### `artha/journal.py` (MODIFIED)
Added 4 new SQLite tables to `_init_db()`:
- `position_theses` — position lifecycle state (thesis_id, ticker, status, position_type, entry_price, hard_stop_price, trailing stop fields, health score, review dates, etc.)
- `sell_signals` — aggregated sell signals with severity + source
- `sell_sessions` — audit log of every sell council run (analyst reports, synthesis, action taken)
- `post_sell_tracking` — shadow-track sold positions for 60 days (5d/20d/60d benchmark checkpoints)

Added new journal methods: `save_thesis()`, `get_thesis()`, `get_active_thesis_for_ticker()`, `get_pending_theses()`, `get_all_active_theses()`, `get_due_reviews()`, `save_sell_signal()`, `get_active_sell_signals()`, `save_sell_session()`, `save_post_sell_tracking()`, `get_pending_post_sell_reviews()`

### `artha/thesis_tracker.py` (CREATED)
- `PositionThesis` dataclass: full position lifecycle fields — thesis_id, ticker, status (pending/active/archived/expired), position_type, invalidation_conditions, hard_stop_price, trailing_stop_price/high, thesis_health_score, next_review_date, sell_cooldown_until, scale_out_completed
- Properties: `is_active`, `is_pending`, `days_held`, `min_hold_days`, `in_minimum_hold`, `in_cooldown`
- `ThesisTracker` class: `create_thesis()`, `activate_thesis()`, `get()`, `get_active()`, `get_all_active()`, `get_pending_for_ticker()`, `get_due_reviews()`, `update_health()`, `update_review_date()`, `update_trailing_stop()`, `set_cooldown()`, `record_scale_out()`, `update_thesis_fields()`, `archive_thesis()`, `expire_stale_pending()`

### `artha/cli/__init__.py` (CREATED)
Empty package init for the `artha/cli/` module directory.

### `artha/cli/portfolio_update.py` (CREATED)
CLI bridge for Ammu (human operator) to manage portfolio state:
- Commands: `buy`, `sell`, `add`, `trim`, `activate-thesis`, `list-pending`, `status`
- All commands output JSON `{success, message, data}`
- `buy` command: finds/creates pending thesis, activates it, attaches sell-engine fields (thesis_id, hard_stop_price, position_type, entry_date, next_sell_review, sell_cooldown_until, scale_out_completed) to portfolio.json
- `sell` command: handles full exit (archives thesis, starts post-sell tracking) vs partial trim
- `_set_position_sell_fields()`: attaches sell-engine metadata to Position objects

### `artha/monitor.py` (MODIFIED)
- Replaced legacy `_alerts_from_portfolio_limits()` with position-type-aware version:
  - Checks `pos.hard_stop_price` (thesis-tracked) first → CRITICAL `hard_stop_breached` alert
  - Falls back to position-type-specific stops from `_pos_type_stops` dict
  - Falls back to legacy `-15%` for positions without `position_type`
  - Scale-out milestone detection: `+15%` for TACTICAL_BUY, `+40%` for BUY
- Added `_check_trailing_stops()`: checks `trailing_stop_price` for TACTICAL_BUY positions
- Wired `_check_trailing_stops()` into `run_check()` after `_alerts_from_portfolio_limits()`

### `artha/council.py` (MODIFIED)
Added thesis auto-creation block after `final_decision` is computed:
- Triggers for verdicts: `BUY`, `STARTER`, `TACTICAL_BUY`, `ACCUMULATE`, `ADD`
- Uses lazy import `from .thesis_tracker import ThesisTracker` (avoids circular import)
- Extracts price_target, stop_pct, regime_str, thesis_summary from council decision
- Creates pending thesis with `tracker.create_thesis()`
- Wrapped in try/except to never break buy-side flow

---

## Phase 2: Proactive News Sentinel

### `artha/sentinel.py` (MODIFIED)
- Added `SentinelDeduplicator` class with 24hr TTL:
  - `is_new(headline_hash)`, `mark_seen(headline_hash)`, `filter_new_events(events)`
  - In-memory `_seen` dict with expiry on access
- Added `self._fast_deduplicator = SentinelDeduplicator(ttl_hours=24)` to `NewsSentinel.__init__`
- Added `run_fast_scan(held_tickers)`: Tier-1 keyword-only fast scan for held positions, uses deduplicator, returns Alert objects
- Added `run_scan_for_tickers(tickers, priority)`: explicit ticker list version of `run_scan()`

### `artha/scheduler.py` (MODIFIED)
- Added `_should_run_sentinel_held()`: variable-frequency gate:
  - Market hours: 5 min, Pre-market: 15 min, After-hours: 30 min, Overnight: 2 hr, Weekends: 4 hr
- Added `_get_held_tickers()`: loads portfolio, returns held ticker list
- Added `_should_run_periodic_review_check()`: delegates to `_should_run_daily_health()` (close+30m window)
- Added `_run_held_sentinel()` async: fast scan for held tickers, escalates CRITICAL alerts to `_assess_thesis_impact_and_alert()`
- Added `_assess_thesis_impact_and_alert()` async: uses Claude Sonnet to map CRITICAL news to invalidation conditions, sends enriched Telegram alert
- Added `_run_periodic_review_check()` async: checks due reviews via ThesisTracker, fires SellCouncil for each, updates review dates
- Wired new tasks into `_tick()` loop
- Updated `run_forever()` startup message with sell engine status (active/pending thesis count)

---

## Phase 3: Sell Council

### `artha/sell_prompts.py` (CREATED)
- `SELL_CONTEXT_HEADER`: shared position context template (entry price, P&L, thesis, invalidation conditions, health score, hard/trailing stops, regime)
- `SELL_FUNDAMENTAL_ANALYST`: Claude Opus 4.6 prompt, thesis integrity focus, outputs 0-100 FUNDAMENTAL SELL SCORE + THESIS STATUS per invalidation condition
- `SELL_TECHNICAL_ANALYST`: Gemini prompt, price action deterioration/momentum, outputs TECHNICAL SELL SCORE + TREND STATUS
- `SELL_CONTRARIAN_ANALYST`: GPT prompt, risk escalation + opportunity cost analysis, outputs CONTRARIAN SELL SCORE + bear case
- `SELL_SYNTHESIS_PROMPT`: CIO synthesizer with weighted score (Fundamental 40% / Technical 30% / Contrarian 30%), position-type score thresholds, anti-hold-bias rule, JSON output block
- `build_sell_context(thesis, stock_data)`: constructs shared position context string
- `build_sell_synthesis_prompt(thesis, reports)`: constructs CIO prompt with correct thresholds per position type

### `artha/sell_council.py` (CREATED)
- `SellAnalystReport` dataclass: analyst_name, model, verdict, sell_score (0-100), confidence, report
- `SellDecision` dataclass: ticker, position_type, action (HOLD/TRIM/EXIT/URGENT_EXIT), sell_score, thesis_status, health_score, fundamental/technical/contrarian reports, synthesis_report, key_reasons, next_review_date, is_urgent, trim_pct, confidence, trigger_type, session_id
- Analyst runners: `_run_sell_fundamental()`, `_run_sell_technical()`, `_run_sell_contrarian()` (same models as buy-side)
- Parsing helpers: `_parse_sell_score()`, `_parse_sell_verdict()`, `_parse_confidence()`, `_parse_synthesis_json()`, `_parse_key_reasons()`
- `SellCouncil` class:
  - `run_sell_review(thesis, stock_data)`: runs 3 analysts in parallel (ThreadPoolExecutor, 180s timeout), CIO synthesis, score computation, action validation, DB persistence
  - `_build_adjustments_text()`: conviction adjustment, health bonus, regime change bonus, time decay
  - `_validate_action(action, thesis)`: enforces min hold period, cooldown, score thresholds
  - `_persist_session()`: saves to `sell_sessions` table
  - `format_sell_telegram(decision)`: Telegram-formatted sell recommendation message

---

## Phase 4: Trailing Stops + Scale-Out + Regime Integration

### `artha/trailing_stop.py` (CREATED)
- `compute_atr(price_history, period=14)`: Wilder's smoothing ATR-14 from OHLCV candle list; falls back to simple average if < 14 candles
- `compute_trailing_stop(entry_price, current_price, high_water_mark, atr, ...)`: ATR-based trailing stop that only ratchets up; applies floor at `min_pct` from high and absolute floor at hard stop
- `check_trailing_stop_breach(current_price, trailing_stop)`: returns True if price <= stop
- `TrailingStopManager` class with `update_position_trailing_stop(thesis, current_price, price_history)`:
  - Only operates on TACTICAL_BUY positions
  - Persists updated stop + high-water mark via ThesisTracker
  - Returns `(new_stop_price, is_breached)` tuple

### `artha/sell_engine.py` (CREATED)
- `SellSignal` dataclass: signal_id, ticker, thesis_id, signal_type, severity (URGENT/HIGH/MEDIUM/LOW), source, message, sell_score, action_recommended
- `SellSignalAggregator`: records signals in DB, retrieves active signals by priority, marks suppressed/actioned
- `PortfolioCircuitBreaker`: tracks exit count per day (max SELL_MAX_EXITS_PER_DAY=2), checks portfolio drawdown > SELL_PORTFOLIO_LOSS_PAUSE_PCT=10%
- `SellEngine` orchestrator:
  - `run_price_check_sell_tasks(portfolio, price_data)`: called during 30-min price checks; updates trailing stops, detects scale-out milestones, checks regime change
  - `_check_scale_out(thesis, current_price)`: compares current gain% vs SELL_SCALE_OUT_* schedules per position type; returns milestone hit if any
  - `_check_regime_change(thesis, regime_state)`: reads data/regime_state.json; flags TACTICAL_BUY if regime changed 3+ days
  - `get_position_health_summary()`, `format_health_report()`: portfolio health reports for Telegram

---

## Phase 5: Opportunity Cost + Post-Sell Tracking

### `artha/opportunity_cost.py` (CREATED)
- `RotationRecommendation` dataclass: from_ticker, to_ticker, from_health, to_score, delta, from_position_type, rationale
- `PostSellTracker` class:
  - `record_sell(ticker, thesis, sell_price, sell_reason, session_id)`: creates post_sell_tracking DB record
  - `update_shadow_prices(collector)`: fetches current prices for all open shadow records, updates 5d/20d/60d checkpoints, computes regret_score, grades sell (CORRECT/NEUTRAL/EARLY/INCORRECT)
  - `format_report()`: text summary of completed and active shadow tracking sessions
- `OpportunityCostScanner` class:
  - `find_weakest_position(theses)`: lowest-health non-locked, non-cooldown, post-min-hold position
  - `_is_conviction_locked(thesis)`: BUY/ACCUMULATE with health>=70 and days_held<180 are locked
  - `evaluate_rotation(weak_thesis, candidate_score, candidate_ticker)`: returns RotationRecommendation if delta >= SELL_ROTATE_MIN_DELTA
  - `scan_for_rotation(theses, collector)`: main entry point, compares weakest held vs best council scan candidate
  - `format_rotation_telegram(rec)`: Telegram message for rotation opportunity

### `artha/accuracy.py` (MODIFIED)
- Added `grade_sell_decisions(collector)`: delegates to PostSellTracker.update_shadow_prices() to refresh sell grades
- Added `format_sell_accuracy_report()`: delegates to PostSellTracker.format_report() for sell accuracy summary

---

## Phase 6: Polish + Reporting

### `artha/scheduler.py` (MODIFIED — additional changes)
- Updated `run_forever()` startup message to include sell engine status:
  - Shows active thesis count and pending thesis count
  - Gracefully falls back to "Sell Engine: active" on any import/DB error

### `artha/SELL_ENGINE_CHANGELOG.md` (CREATED)
This file.

---

## New Files Summary

| File | Type | Description |
|------|------|-------------|
| `artha/thesis_tracker.py` | NEW | PositionThesis dataclass + ThesisTracker CRUD |
| `artha/cli/__init__.py` | NEW | CLI package init |
| `artha/cli/portfolio_update.py` | NEW | Ammu CLI bridge for buy/sell/status |
| `artha/sell_prompts.py` | NEW | Sell-side analyst + synthesis prompts |
| `artha/sell_council.py` | NEW | SellCouncil + SellDecision orchestrator |
| `artha/trailing_stop.py` | NEW | ATR-14 trailing stop logic |
| `artha/sell_engine.py` | NEW | SellEngine + SellSignalAggregator + CircuitBreaker |
| `artha/opportunity_cost.py` | NEW | OpportunityCostScanner + PostSellTracker |
| `artha/SELL_ENGINE_CHANGELOG.md` | NEW | This file |

## Modified Files Summary

| File | Changes |
|------|---------|
| `artha/config.py` | +60 sell-side config constants |
| `artha/journal.py` | +4 DB tables, +11 DB methods |
| `artha/monitor.py` | Position-type hard stops, trailing stop checks |
| `artha/council.py` | Thesis auto-creation on BUY verdicts |
| `artha/sentinel.py` | SentinelDeduplicator, run_fast_scan, run_scan_for_tickers |
| `artha/scheduler.py` | Held sentinel, periodic review, sell engine startup message |
| `artha/accuracy.py` | grade_sell_decisions, format_sell_accuracy_report |
