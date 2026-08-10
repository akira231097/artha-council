# ARTHA Dashboard

Real-time, phone-friendly observability for the ARTHA trading agent. One page:
portfolio vitals, equity curve with trade markers, positions with stop-loss
protection bars, today's activity timeline, council decisions, the self-grading
report card + lessons, system health, and the fixes/self-improvement feed.

## Architecture

- `server.py` — stdlib-only Python HTTP server (no dependencies). Assembles
  `/api/dashboard` from ARTHA's live data files and the journal DB
  (**strictly read-only**: sqlite `mode=ro`, 2s busy timeout). Shellouts
  (openclaw cron state, git log) are cached 5 min and time-boxed. Its only
  writes are under `data/dashboard/` (access token, intraday equity samples).
- `index.html` — single-page vanilla JS/CSS, hand-rolled SVG charts, zero CDN.
  Polls every 15s while the market is open, 60s otherwise.

## Access

- Served on `0.0.0.0:8787` (`ARTHA_DASHBOARD_PORT` to change).
- Token auth: `data/dashboard/token.txt` (auto-generated, chmod 600).
  Open `http://<host>:8787/?k=<token>` once — a cookie keeps you signed in.
- Phone on home Wi-Fi: `http://192.168.1.158:8787/?k=<token>`
- Anywhere via Tailscale: `http://100.88.234.49:8787/?k=<token>`
  (requires Tailscale on the phone, same tailnet).

## Service

launchd label `com.artha.dashboard` (plist in `data/launchd/`):

```sh
cp data/launchd/com.artha.dashboard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.artha.dashboard.plist
```

Logs: `data/logs/dashboard.{out,err}.log`.

## Notes

- Equity history before 2026-06-16 18:00Z is excluded: those snapshots
  double-counted the first buy (pre-fix accounting) and would show a fake loss.
- The dashboard never places orders and cannot modify ARTHA state.
