# Public Release Notes

This repository is a sanitized public release intended to show the architecture and implementation quality of Artha.

The live/private workspace may contain:

- `.env` credentials
- generated investment reports
- Robinhood snapshots
- SQLite journals
- Telegram action tokens
- runtime locks
- local launchd/OpenClaw state

Those files are deliberately excluded here.

If you clone this repository, create your own `.env` from `.env.example`, use your own provider accounts, and keep live broker actions disabled until you have verified the full system.
