"""
BluePilot Backend Network Module
Network utilities and connection management
"""

from .utils import (
    is_onroad,
    should_server_run,
    get_portal_port,
    get_portal_language,
    get_wifi_ip,
    get_all_wifi_ips,
    get_connection_type,
)
from .port_redirect import resolve_bind_port, setup_port_redirect, teardown_port_redirect

__all__ = [
    'is_onroad',
    'should_server_run',
    'get_portal_port',
    'get_portal_language',
    'get_wifi_ip',
    'get_all_wifi_ips',
    'get_connection_type',
    'resolve_bind_port',
    'setup_port_redirect',
    'teardown_port_redirect',
]
