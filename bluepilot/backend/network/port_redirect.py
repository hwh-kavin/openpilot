#!/usr/bin/env python3
"""
Redirect public HTTP port (80) to the portal bind port via iptables.

Comma devices run openpilot as a non-root user, so binding to port 80 often
fails with EACCES. We listen on FALLBACK_BIND_PORT and redirect inbound :80.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Iterable

logger = logging.getLogger(__name__)

_IPTABLES = ("sudo", "iptables-legacy")
_active_rules: list[tuple[str, int, int, str | None]] = []


def _rule_match_args(public_port: int, bind_port: int, dest_ip: str | None) -> list[str]:
    args = ["-p", "tcp", "--dport", str(public_port)]
    if dest_ip:
        args = ["-d", dest_ip, *args]
    args += ["-j", "REDIRECT", "--to-ports", str(bind_port)]
    return args


def _iptables(action: str, chain: str, public_port: int, bind_port: int, dest_ip: str | None) -> list[str]:
    return [*_IPTABLES, "-t", "nat", action, chain, *_rule_match_args(public_port, bind_port, dest_ip)]


def _ensure_rule(chain: str, public_port: int, bind_port: int, dest_ip: str | None) -> bool:
    key = (chain, public_port, bind_port, dest_ip)
    if key in _active_rules:
        return True

    check = subprocess.run(_iptables("-C", chain, public_port, bind_port, dest_ip), capture_output=True, timeout=5)
    if check.returncode == 0:
        _active_rules.append(key)
        return True

    add = subprocess.run(_iptables("-A", chain, public_port, bind_port, dest_ip), capture_output=True, timeout=5)
    if add.returncode != 0:
        stderr = add.stderr.decode("utf-8", errors="ignore").strip()
        logger.warning("iptables %s redirect failed for %s:%s -> %s: %s", chain, dest_ip or "*", public_port, bind_port, stderr)
        return False

    _active_rules.append(key)
    logger.info("iptables %s redirect %s:%s -> %s", chain, dest_ip or "*", public_port, bind_port)
    return True


def setup_port_redirect(public_port: int, bind_port: int, dest_ips: Iterable[str] | None = None) -> bool:
    """Redirect inbound public_port traffic to bind_port."""
    if public_port == bind_port:
        return True

    ips = list(dest_ips or [])
    if not ips:
        ips = [None]  # type: ignore[list-item]

    ok = False
    for dest_ip in ips:
        if _ensure_rule("PREROUTING", public_port, bind_port, dest_ip):
            ok = True
        if dest_ip and _ensure_rule("OUTPUT", public_port, bind_port, dest_ip):
            ok = True
    return ok


def teardown_port_redirect() -> None:
    for chain, public_port, bind_port, dest_ip in reversed(_active_rules):
        subprocess.run(_iptables("-D", chain, public_port, bind_port, dest_ip), capture_output=True, timeout=5)
    _active_rules.clear()


def resolve_bind_port(public_port: int, dest_ips: Iterable[str] | None = None) -> tuple[int, bool]:
    """
    Pick the local bind port for the HTTP server.

    Returns:
        (bind_port, redirect_active)
    """
    import socket

    from bluepilot.backend.config import FALLBACK_BIND_PORT

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", public_port))
        sock.close()
        teardown_port_redirect()
        return public_port, False
    except OSError as exc:
        sock.close()
        logger.info("Cannot bind port %s (%s), using %s with iptables redirect", public_port, exc, FALLBACK_BIND_PORT)

    bind_port = FALLBACK_BIND_PORT
    redirect_ok = setup_port_redirect(public_port, bind_port, dest_ips)
    if not redirect_ok:
        logger.warning("Port redirect setup failed; clients must use :%s explicitly", bind_port)
    return bind_port, redirect_ok
