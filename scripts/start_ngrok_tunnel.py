from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
NGROK_ENDPOINT_URL = "https://wren-fun-hyena.ngrok-free.app"


@dataclass(frozen=True)
class TunnelConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    reload: bool = False


@dataclass(frozen=True)
class DiagnosticResult:
    ok: bool
    output: str


def uvicorn_command(config: TunnelConfig, python_executable: str = sys.executable) -> list[str]:
    command = [
        python_executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    if config.reload:
        command.append("--reload")
    return command


def ngrok_tunnel_command(config: TunnelConfig, ngrok_executable: str) -> list[str]:
    return [
        ngrok_executable,
        "http",
        _local_url(config),
        "--url",
        NGROK_ENDPOINT_URL,
    ]


def ngrok_auth_command(authtoken: str, ngrok_executable: str) -> list[str]:
    return [ngrok_executable, "config", "add-authtoken", authtoken]


def ngrok_diagnose_command(ngrok_executable: str) -> list[str]:
    return [ngrok_executable, "diagnose"]


def diagnose_ngrok(ngrok_executable: str) -> DiagnosticResult:
    completed = subprocess.run(
        ngrok_diagnose_command(ngrok_executable),
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return DiagnosticResult(ok=completed.returncode == 0, output=output)


def run(config: TunnelConfig, ngrok_executable: str, authtoken: str | None, start_server: bool) -> int:
    resolved_ngrok = _resolve_executable(ngrok_executable)

    if authtoken:
        subprocess.run(
            ngrok_auth_command(authtoken, resolved_ngrok),
            check=True,
            stdout=subprocess.DEVNULL,
        )

    diagnostic = diagnose_ngrok(resolved_ngrok)
    if not diagnostic.ok:
        print("ngrok connectivity check failed before starting the local API.", file=sys.stderr)
        print(diagnostic.output, file=sys.stderr)
        if "failed to fetch CRL" in diagnostic.output:
            print(
                "This is a TLS certificate revocation-list connectivity problem, not a FastAPI port problem. "
                "Check system time, VPN/proxy/firewall interception, and access to ngrok TLS/CRL endpoints.",
                file=sys.stderr,
            )
        return 1

    server_process: subprocess.Popen | None = None
    ngrok_process: subprocess.Popen | None = None
    try:
        if start_server:
            print(f"Starting FastAPI at {_local_url(config)}")
            server_process = subprocess.Popen(uvicorn_command(config))
            time.sleep(1.5)
            if server_process.poll() is not None:
                return server_process.returncode or 1
        else:
            print(f"Using existing FastAPI server at {_local_url(config)}")

        print("Starting ngrok tunnel. Copy the Forwarding HTTPS URL from the ngrok output.")
        ngrok_process = subprocess.Popen(ngrok_tunnel_command(config, resolved_ngrok))
        return ngrok_process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_process(ngrok_process)
        _stop_process(server_process)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Start the local API and expose it through ngrok.")
    parser.add_argument("--host", default=os.getenv("API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", DEFAULT_PORT)))
    parser.add_argument("--reload", action="store_true", help="Start uvicorn with auto-reload.")
    parser.add_argument("--ngrok-path", default=os.getenv("NGROK_PATH", "ngrok"))
    parser.add_argument("--authtoken", default=os.getenv("NGROK_AUTHTOKEN"))
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Only start ngrok; use this when uvicorn is already running.",
    )
    args = parser.parse_args()

    config = TunnelConfig(host=args.host, port=args.port, reload=args.reload)
    return run(
        config=config,
        ngrok_executable=args.ngrok_path,
        authtoken=args.authtoken,
        start_server=not args.no_server,
    )


def _local_url(config: TunnelConfig) -> str:
    return f"http://{config.host}:{config.port}"


def _resolve_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    raise SystemExit(
        f"Could not find '{executable}'. Install ngrok and make sure it is on PATH, "
        "or pass --ngrok-path C:\\path\\to\\ngrok.exe."
    )


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
