"""Cortex main entry point."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    """Main entry point for the Cortex service."""
    parser = argparse.ArgumentParser(
        prog="aperion-cortex",
        description="Aperion Cortex - Unified memory and state management service",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4949,
        help="Port to listen on (default: 4949)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Force JSON log output (default: auto-detect)",
    )

    args = parser.parse_args()

    # Configure structured logging before anything else
    try:
        from aperion_cortex.service.logging import configure_logging
        configure_logging(
            level=args.log_level.upper(),
            json_output=args.json_logs if args.json_logs else None,
        )
    except ImportError:
        # Fallback to basic logging if structlog not installed
        import logging
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # Set environment defaults if not set
    if "CORTEX_SIGNING_KEY" not in os.environ:
        # For development only - generate a random key
        import secrets
        os.environ["CORTEX_SIGNING_KEY"] = secrets.token_hex(32)
        print("Warning: Using auto-generated signing key. Set CORTEX_SIGNING_KEY for production.")

    os.environ.setdefault("CORTEX_PORT", str(args.port))

    try:
        import uvicorn

        uvicorn.run(
            "aperion_cortex.service.app:create_app_from_env",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            factory=True,
        )
        return 0
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install uvicorn[standard]")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
