#!/usr/bin/env python3
"""
BluePilot Backend Network Utilities
Network interface detection and connection type management
"""

import logging

from bluepilot.backend.utils.params_fallback import get_params_with_defaults

logger = logging.getLogger(__name__)

params = get_params_with_defaults({
    "IsOnRoad": False,
    "BPPortalPort": "80",
    "EnableWebRoutesServer": True,
    "EnableCopyparty": False,
})


def is_onroad():
    """Check if vehicle is currently driving"""
    try:
        return params.get_bool("IsOnRoad")
    except:
        return False


def portal_enabled() -> bool:
    """Return whether the BluePilot Portal service is enabled."""
    try:
        if params.get_bool("EnableCopyparty"):
            return True
        return params.get_bool("EnableWebRoutesServer")
    except Exception:
        return False


def should_server_run():
    """Check if server should be running (always runs when enabled, rate-limited onroad)"""
    return portal_enabled()


def get_portal_language(port_params=None) -> str:
    """Resolve portal UI language: zh-CHS or en (only two locales bundled)."""
    port_params = port_params or params
    try:
        raw_lang = port_params.get("LanguageSetting")
        if isinstance(raw_lang, bytes):
            raw_lang = raw_lang.decode("utf-8", errors="ignore")
        raw_lang = str(raw_lang or "en").removeprefix("main_")
        if raw_lang.startswith("zh"):
            return "zh-CHS"
    except Exception:
        logger.exception("Failed to read LanguageSetting, using English portal locale")
    return "en"


def get_portal_port(port_params=None, migrate: bool = True) -> int:
    """Resolve the public HTTP port for BluePilot Portal (default 80)."""
    from bluepilot.backend.config import DEFAULT_PORT

    port_params = port_params or params
    try:
        raw_port = port_params.get("BPPortalPort")
        if isinstance(raw_port, bytes):
            raw_port = raw_port.decode("utf-8", errors="ignore")
        if raw_port in (None, "", 0, "0", 8088, "8088"):
            if migrate and raw_port in (8088, "8088"):
                try:
                    port_params.put("BPPortalPort", DEFAULT_PORT)
                    logger.info("Migrated BPPortalPort from 8088 to %s", DEFAULT_PORT)
                except Exception:
                    logger.exception("Failed to migrate BPPortalPort to %s", DEFAULT_PORT)
            return DEFAULT_PORT
        port = int(raw_port)
        if 1 <= port <= 65535:
            return port
    except Exception:
        logger.exception("Failed to read BPPortalPort, using default portal port")
    return DEFAULT_PORT


def get_wifi_ip():
    """Get WiFi interface IP address (first wlan interface found)"""
    try:
        import netifaces
        for iface in netifaces.interfaces():
            if iface.startswith('wlan'):
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr.get('addr')
                        if ip and not ip.startswith('127.'):
                            return ip
    except ImportError:
        # Fallback without netifaces
        import subprocess
        try:
            result = subprocess.run(['ip', 'addr', 'show', 'wlan0'],
                                    capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\n'):
                if 'inet ' in line:
                    ip = line.strip().split()[1].split('/')[0]
                    return ip
        except:
            pass
    return None


def get_all_wifi_ips():
    """Get all WiFi interface IP addresses (for hotspot support)"""
    ips = []
    try:
        import netifaces
        for iface in netifaces.interfaces():
            # Include wlan interfaces only (not cellular)
            if iface.startswith('wlan'):
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr.get('addr')
                        if ip and not ip.startswith('127.'):
                            ips.append((iface, ip))
    except ImportError:
        # Fallback: just get wlan0
        wifi_ip = get_wifi_ip()
        if wifi_ip:
            ips.append(('wlan0', wifi_ip))
    return ips


def get_connection_type():
    """Determine current network connection type"""
    try:
        import subprocess
        # Check which interface is being used for default route
        result = subprocess.run(['ip', 'route', 'get', '8.8.8.8'],
                                capture_output=True, text=True, timeout=2)
        output = result.stdout.lower()

        if 'wlan' in output:
            return 'wifi'
        elif 'rmnet' in output or 'ccmni' in output:
            return 'cellular'
        elif 'eth' in output:
            return 'ethernet'
        else:
            return 'unknown'
    except:
        return 'unknown'
