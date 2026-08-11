"""Command-line entry point for Artha MCP."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace

from mcp.server.transport_security import TransportSecuritySettings

from . import __version__
from .provenance import build_provenance
from .server import create_server
from .settings import MCPSettings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Artha Council MCP server")
    parser.add_argument(
        "--transport", choices=("stdio", "streamable-http"), help="MCP transport"
    )
    parser.add_argument("--host", help="HTTP bind host")
    parser.add_argument("--port", type=int, help="HTTP bind port")
    parser.add_argument("--path", help="Streamable HTTP endpoint path")
    parser.add_argument(
        "--check", action="store_true", help="Validate redacted configuration and exit"
    )
    parser.add_argument(
        "--version", action="version", version=f"artha-mcp {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        settings = MCPSettings.from_env()
    except ValueError as exc:
        print(f"Artha MCP configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    settings = replace(
        settings,
        transport=args.transport or settings.transport,
        host=args.host or settings.host,
        port=args.port or settings.port,
        http_path=args.path or settings.http_path,
    )
    findings = settings.startup_findings()
    provenance = build_provenance()
    if args.check:
        payload = settings.public_summary()
        payload["provenance"] = provenance
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        raise SystemExit(
            1 if findings["errors"] or provenance["status"] == "FAIL" else 0
        )
    if findings["errors"] or provenance["status"] == "FAIL":
        for message in findings["errors"]:
            print(f"Artha MCP configuration error: {message}", file=sys.stderr)
        for message in provenance["errors"]:
            print(f"Artha MCP source-alignment error: {message}", file=sys.stderr)
        raise SystemExit(2)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = create_server(settings)
    if settings.transport == "stdio":
        server.run(transport="stdio")
        return
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(settings.effective_allowed_hosts),
        allowed_origins=list(settings.allowed_origins),
    )
    server.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
        streamable_http_path=settings.http_path,
        json_response=True,
        stateless_http=True,
        max_request_body_size=1024 * 1024,
        transport_security=transport_security,
    )


if __name__ == "__main__":
    main()
