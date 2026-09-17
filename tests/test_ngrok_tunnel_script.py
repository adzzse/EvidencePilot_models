from scripts.start_ngrok_tunnel import (
    TunnelConfig,
    diagnose_ngrok,
    ngrok_auth_command,
    ngrok_diagnose_command,
    ngrok_tunnel_command,
    uvicorn_command,
)


def test_builds_default_uvicorn_and_ngrok_commands():
    config = TunnelConfig(host="127.0.0.1", port=8000)

    assert uvicorn_command(config, python_executable="python") == [
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert ngrok_tunnel_command(config, ngrok_executable="ngrok") == [
        "ngrok",
        "http",
        "http://127.0.0.1:8000",
        "--url",
        "https://wren-fun-hyena.ngrok-free.app",
    ]


def test_builds_reload_and_authtoken_commands():
    config = TunnelConfig(host="0.0.0.0", port=9000, reload=True)

    assert uvicorn_command(config, python_executable="python")[-1] == "--reload"
    assert ngrok_auth_command("secret-token", ngrok_executable="ngrok") == [
        "ngrok",
        "config",
        "add-authtoken",
        "secret-token",
    ]


def test_builds_ngrok_diagnose_command():
    assert ngrok_diagnose_command("ngrok") == ["ngrok", "diagnose"]


def test_diagnose_ngrok_returns_error_output(monkeypatch):
    import subprocess

    def fake_run(args, **kwargs):
        assert args == ["ngrok", "diagnose"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="No tunnel servers could establish a TLS connection.",
            stderr="failed to fetch CRL",
        )

    monkeypatch.setattr("scripts.start_ngrok_tunnel.subprocess.run", fake_run)

    result = diagnose_ngrok("ngrok")

    assert result.ok is False
    assert "failed to fetch CRL" in result.output
