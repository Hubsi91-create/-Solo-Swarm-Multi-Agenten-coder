"""
Dashboard Backend Module - API and WebSocket Services
"""

from .api import app
from .dashboard_manager import DashboardManager, UpdateType, DashboardUpdate

__all__ = ["app", "DashboardManager", "UpdateType", "DashboardUpdate"]
