# Sell-Side Engine — Bug Fixes Applied
_Applied: 2026-03-13 (GPT 5.4 code review findings)_

---

## CRITICAL

### FIX 1 — thesis_tracker.py: `expire_stale_pending()` never expired rows
**Root cause:** `expire_stale_pending()` called `get_pending_theses()`, which already filters out rows with `pending_expiry <= now`. So stale rows were invisible to the expiry loop.
**Fix:**
- Added `journal.get_all_pending_theses_raw()` — queries ALL `status='pending'` rows with no expiry filter
- `expire_stale_pending()` now uses this raw query and marks rows with `pending_expiry <= now` as `'expired'`

### FIX 2 — cli/portfolio_update.py: `total_nav()` missing + save before validation
**Root cause:** `cmd_buy()` called `portfolio.total_nav()` (method didn't exist) after saving the portfolio. Crash always occurred on line 121.
**Fix:**
- Added `Portfolio.total_nav()` method to `portfolio.py` — returns sum of `market_value` if set, else cost basis
- In `cmd_buy()`, moved NAV computation and `alloc_pct` calculation **before** `_save_portfolio()` so validation completes before persistence

---

## HIGH

### FIX 3 — portfolio.py: sell fields silently dropped on save
**Root cause:** `_set_position_sell_fields()` added `thesis_id`, `position_type`, `hard_stop_price`, etc. as ad-hoc Python attributes. `Portfolio.save()` uses `asdict()` which only serializes declared dataclass fields.
**Fix:** Added the following as `Optional` fields to the `Position` dataclass:
- `thesis_id`, `position_type`, `hard_stop_price`, `trailing_stop_price`
- `next_sell_review`, `sell_cooldown_until`
- `current_price`, `market_value` (used in CLI price display)

`save()` and `load()` automatically include them via `asdict()` / `Position(**p)`.

### FIX 4 — cli/portfolio_update.py: `activate-thesis` didn't update portfolio
**Root cause:** `cmd_activate_thesis()` updated the thesis DB record but never touched the live portfolio position, so `hard_stop_price`, `thesis_id`, etc. were never written to `portfolio.json`.
**Fix:** After thesis activation, locate the matching portfolio position by ticker, call `_set_position_sell_fields()` with the newly activated thesis data, and persist.

### FIX 5 — scheduler.py: `SellEngine` never instantiated (dead code)
**Root cause:** Sell-side logic (trailing stops, scale-out, stale cleanup) was implemented in `sell_engine.py` but the scheduler never created a `SellEngine` instance.
**Fix:**
- Added `from .sell_engine import SellEngine` import
- Created `self.sell_engine = SellEngine(journal=DecisionJournal(), collector=self.collector)` in `ArthaScheduler.__init__()`
- Added `_run_sell_engine_price_check()` async method that fetches live quotes and calls `self.sell_engine.run_price_check_sell_tasks(portfolio, quotes)`
- Wired the call into `_run_monitor_check()` (runs every 30 min during market hours)

### FIX 6 — sell_council.py: `TimeoutError` not caught in `as_completed`
**Root cause:** If `as_completed(futures, timeout=180)` raised `TimeoutError`/`FuturesTimeout`, the exception propagated out of the loop and aborted the entire sell review.
**Fix:** Wrapped the `for future in as_completed(...)` loop in `try/except (TimeoutError, FuturesTimeout)`. On timeout, remaining futures are cancelled and the review continues with whichever analyst reports completed.

### FIX 7 — sell_council.py: unavailable analyst defaulted to score 50 (neutral)
**Root cause:** Missing analyst reports were replaced with the string `"Analyst unavailable — treating as HOLD."`, then `_parse_sell_score()` was called on it and returned the default `50`, polluting the weighted average.
**Fix:**
- Unavailable analysts stay `None`; no placeholder text is injected into the scoring path
- If `< 2` analysts available, return an explicit `HOLD` `SellDecision` immediately
- `f_score`, `t_score`, `c_score` are `None` for unavailable analysts
- Fallback composite sell score uses **renormalized weights** over available analysts only
- `SellAnalystReport` objects are created only for available analysts; `None` is passed for missing ones

### FIX 8 — council.py: duplicate pending theses created on every BUY verdict
**Root cause:** Every call to `analyze_stock()` that returned a BUY verdict called `tracker.create_thesis()` unconditionally, even if a pending or active thesis already existed.
**Fix:** Before creating a thesis:
1. If an **active** thesis exists for the ticker → skip creation entirely
2. If a **pending** thesis exists → call `tracker.update_thesis_fields()` to refresh it in place
3. Otherwise → create new pending thesis as before

### FIX 9 — cli/portfolio_update.py: `cmd_sell()` wrote proceeds to `portfolio.cash` (nonexistent)
**Root cause:** Lines 216-217 set `portfolio.cash = ...` (field doesn't exist on the `Portfolio` dataclass) and called `_save_portfolio()` a second time. The ad-hoc attribute was silently dropped on save, losing all proceeds tracking.
**Fix:**
- Removed the bogus `portfolio.cash` assignment and duplicate save
- After the sell/trim position logic, subtract the released cost basis from `portfolio.cash_deployed` (same logic as `Portfolio.sell_position()`)

---

## MEDIUM

### FIX 10 — journal.py: `get_pending_post_sell_reviews()` excluded completed rows
**Root cause:** The method filtered `status = 'tracking'` only; completed post-sell rows were inaccessible.
**Fix:** Added `get_all_post_sell_reviews()` that returns all rows (both `tracking` and `completed`) for historical analysis. Existing `get_pending_post_sell_reviews()` is preserved for active tracking use cases.

### FIX 11 — thesis_tracker.py / journal.py: concurrent thesis updates overwrote each other
**Root cause:** `ThesisTracker._save()` pre-set `thesis.updated_at = _utcnow_iso()` before the write, making it impossible to detect whether another concurrent write had already updated the row.
**Fix:**
- `ThesisTracker._save()` no longer pre-sets `updated_at`; the old value (from when the thesis was loaded) is preserved and passed to `journal.save_thesis()`
- `journal.save_thesis()` reads the DB's current `updated_at` before writing; if `DB.updated_at > loaded.updated_at`, it logs a warning and skips the write (optimistic locking)
- `journal.save_thesis()` then sets `updated_at = now` on the data that will be written

### FIX 12 — scheduler.py: confirmation gate not implemented for EXIT signals
**Root cause:** Non-urgent `EXIT` decisions from sell council were sent immediately, ignoring the documented 2-day confirmation rule (`Config.SELL_CONFIRMATION_DAYS`).
**Fix:**
- Added `self._pending_exit_signals: dict[str, datetime]` to `ArthaScheduler.__init__()` to track first-seen timestamps per thesis
- In `_run_periodic_review_check()`, when `action == "EXIT"` and `not decision.is_urgent`:
  - First occurrence: store `first_seen = now`, send a "pending confirmation" Telegram alert, and **skip advancing the review date** so the thesis re-triggers the next daily cycle
  - Subsequent occurrence within `< SELL_CONFIRMATION_DAYS`: log waiting status, do nothing
  - After `>= SELL_CONFIRMATION_DAYS`: confirmed — send full sell alert and advance review date
  - If the signal changes to HOLD/TRIM before confirmation: discard the pending signal
- `URGENT_EXIT` always bypasses the gate and sends immediately
