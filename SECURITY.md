# Security Policy

This public release should never contain real credentials, broker snapshots, portfolio state, Telegram tokens, OAuth tokens, or live trading journals.

## Secret Handling

- Put local secrets in `.env`.
- Use `.env.example` only as a template.
- Do not commit `.env`, `.env.*`, database files, runtime reports, logs, or broker snapshots.
- Rotate any key that may have been committed to a private or public repository history.

## Live Trading Safety

The public defaults are safe by design:

- Robinhood review-only mode is enabled.
- Robinhood dry-run mode is enabled.
- Agentic trading is disabled.
- The kill switch is enabled.
- Auto-buy is disabled.

Only enable live trading after reviewing the full broker bridge, execution officer, account caps, and Robinhood terms.

## Reporting Issues

If you find a security issue, open a private disclosure channel with the repository owner rather than posting secrets or exploit details in a public issue.
