# Contributing

Thank you for improving Artha Council. Contributions should preserve the core
design rule: investment reasoning and broker execution are separate stages, and
money-moving paths fail closed when required evidence is missing.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
npm ci
cp .env.example .env
```

Do not put real credentials or broker data in tests, examples, issues, or pull
requests. Public fixtures must use invented account identifiers, prices, orders,
positions, and API responses.

## Before Opening a Pull Request

```bash
python -m compileall -q artha dashboard run.py
python -m artha.test_enhancements
python -m artha.test_production_hardening
```

Keep changes focused. Add regression coverage for changes to scoring, routing,
portfolio limits, broker review, order placement, sell monitoring, or runtime
state transitions. Never weaken a safety gate solely to make a test pass.

## License

Unless explicitly stated otherwise, submitted contributions are licensed under
Apache License 2.0, as described in [LICENSE](LICENSE).
