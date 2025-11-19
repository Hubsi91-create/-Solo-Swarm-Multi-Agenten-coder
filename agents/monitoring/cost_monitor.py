"""
Cost Monitor - Continuous Budget Monitoring & Hard Limit Enforcement

This module implements a background monitoring task that continuously checks
daily costs against configured hard limits and triggers emergency shutdown
when limits are exceeded.

Features:
- Runs every 60 seconds (configurable)
- Checks daily costs from database
- Enforces hard budget limits
- Automatic emergency shutdown on limit breach
- Dashboard notifications
- Graceful shutdown support

Critical Safety Component: This is the last line of defense against
runaway costs. It must be robust and reliable.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from core.token_tracker import TokenTracker
from core.emergency_shutdown import EmergencyShutdown, ShutdownReason, is_system_shutdown


logger = logging.getLogger(__name__)


class CostMonitor:
    """
    Background Cost Monitor with Hard Limit Enforcement.

    This monitor runs continuously in the background and enforces
    budget limits by triggering emergency shutdown when necessary.

    Usage:
        monitor = CostMonitor(
            token_tracker=tracker,
            emergency_shutdown=shutdown_manager,
            daily_limit_usd=15.00,
            check_interval_seconds=60
        )

        # Start monitoring (runs in background)
        await monitor.start()

        # Stop monitoring (graceful shutdown)
        await monitor.stop()
    """

    def __init__(
        self,
        token_tracker: TokenTracker,
        emergency_shutdown: EmergencyShutdown,
        daily_limit_usd: float = 15.00,
        check_interval_seconds: int = 60,
        warning_threshold_pct: float = 80.0,
        dashboard_manager: Optional[Any] = None
    ):
        """
        Initialize the Cost Monitor.

        Args:
            token_tracker: TokenTracker instance with database access
            emergency_shutdown: EmergencyShutdown manager
            daily_limit_usd: Maximum allowed daily cost (default: $15.00)
            check_interval_seconds: How often to check costs (default: 60s)
            warning_threshold_pct: Warning threshold percentage (default: 80%)
            dashboard_manager: Optional dashboard for cost updates
        """
        self.token_tracker = token_tracker
        self.emergency_shutdown = emergency_shutdown
        self.daily_limit_usd = daily_limit_usd
        self.check_interval_seconds = check_interval_seconds
        self.warning_threshold_pct = warning_threshold_pct
        self.dashboard_manager = dashboard_manager

        # Monitoring state
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.check_count = 0
        self.last_check_time: Optional[datetime] = None
        self.warning_sent = False  # Track if warning was already sent

        # Statistics
        self.stats = {
            "total_checks": 0,
            "warnings_triggered": 0,
            "shutdowns_triggered": 0,
            "errors_encountered": 0
        }

        logger.info(
            f"CostMonitor initialized - "
            f"Daily Limit: ${daily_limit_usd:.2f}, "
            f"Check Interval: {check_interval_seconds}s"
        )

    async def start(self) -> None:
        """
        Start the cost monitoring background task.

        This creates an async task that runs continuously until stopped.
        """
        if self.is_running:
            logger.warning("CostMonitor is already running")
            return

        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            f"🚀 CostMonitor started - "
            f"Monitoring daily budget: ${self.daily_limit_usd:.2f}"
        )

    async def stop(self) -> None:
        """
        Stop the cost monitoring background task.

        Performs graceful shutdown of the monitoring loop.
        """
        if not self.is_running:
            logger.warning("CostMonitor is not running")
            return

        self.is_running = False

        # Cancel the monitoring task
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"🛑 CostMonitor stopped - "
            f"Total checks: {self.stats['total_checks']}, "
            f"Warnings: {self.stats['warnings_triggered']}"
        )

    async def _monitor_loop(self) -> None:
        """
        Main monitoring loop.

        Runs continuously, checking costs at the configured interval.
        """
        logger.info("Cost monitoring loop started")

        while self.is_running:
            try:
                # Check if system is already shutdown
                if is_system_shutdown():
                    logger.info(
                        "System is in shutdown state, "
                        "cost monitor entering standby mode"
                    )
                    # Continue monitoring but don't take action
                    await asyncio.sleep(self.check_interval_seconds)
                    continue

                # Perform cost check
                await self._perform_check()

                # Update statistics
                self.stats["total_checks"] += 1
                self.check_count += 1
                self.last_check_time = datetime.utcnow()

                # Wait for next check interval
                await asyncio.sleep(self.check_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Cost monitor loop cancelled")
                break

            except Exception as e:
                logger.error(f"Error in cost monitor loop: {e}", exc_info=True)
                self.stats["errors_encountered"] += 1

                # Continue monitoring despite errors (but wait a bit longer)
                await asyncio.sleep(self.check_interval_seconds * 2)

        logger.info("Cost monitoring loop ended")

    async def _perform_check(self) -> None:
        """
        Perform a single cost check against the hard limit.

        This is where the critical safety logic lives.
        """
        try:
            # Check hard limit
            limit_exceeded, details = await self.token_tracker.check_hard_limit(
                self.daily_limit_usd
            )

            # Get current utilization
            utilization_pct = details.get("utilization_percent", 0)

            # CRITICAL: Hard limit exceeded - trigger emergency shutdown
            if limit_exceeded:
                logger.critical(
                    f"💥 HARD LIMIT EXCEEDED: ${details['daily_cost_usd']:.4f} / "
                    f"${self.daily_limit_usd:.2f} - TRIGGERING EMERGENCY SHUTDOWN"
                )

                # Trigger emergency shutdown
                await self.emergency_shutdown.stop_all_agents(
                    reason=ShutdownReason.COST_LIMIT_REACHED,
                    details=details
                )

                self.stats["shutdowns_triggered"] += 1

                # Send critical notification to dashboard
                if self.dashboard_manager:
                    await self._send_shutdown_notification(details)

                # Stop monitoring (system is now locked)
                self.is_running = False

            # WARNING: Approaching limit
            elif utilization_pct >= self.warning_threshold_pct and not self.warning_sent:
                logger.warning(
                    f"⚠️  Budget warning: {utilization_pct:.1f}% utilized "
                    f"(${details['daily_cost_usd']:.4f} / ${self.daily_limit_usd:.2f})"
                )

                self.stats["warnings_triggered"] += 1
                self.warning_sent = True

                # Send warning to dashboard
                if self.dashboard_manager:
                    await self._send_warning_notification(details)

            # Normal operation
            else:
                # Log periodic status (every 10th check)
                if self.check_count % 10 == 0:
                    logger.info(
                        f"Budget status: {utilization_pct:.1f}% "
                        f"(${details['daily_cost_usd']:.4f} / ${self.daily_limit_usd:.2f})"
                    )

                # Send routine cost update to dashboard
                if self.dashboard_manager and self.check_count % 5 == 0:
                    await self._send_cost_update(details)

        except Exception as e:
            logger.error(f"Error performing cost check: {e}", exc_info=True)
            raise

    async def _send_shutdown_notification(self, details: Dict[str, Any]) -> None:
        """
        Send critical shutdown notification to dashboard.

        Args:
            details: Cost limit details
        """
        try:
            from dashboard.backend.dashboard_manager import DashboardUpdate, UpdateType

            update = DashboardUpdate(
                update_type=UpdateType.COST_UPDATE,
                data={
                    "event": "COST_LIMIT_EXCEEDED",
                    "severity": "critical",
                    "daily_cost_usd": details["daily_cost_usd"],
                    "daily_limit_usd": details["daily_limit_usd"],
                    "utilization_percent": details["utilization_percent"],
                    "overage_usd": details["daily_cost_usd"] - details["daily_limit_usd"],
                    "shutdown_triggered": True,
                    "message": "Emergency shutdown triggered due to budget limit breach"
                },
                timestamp=datetime.utcnow(),
                priority=2  # Critical
            )

            await self.dashboard_manager.broadcast(update)

        except Exception as e:
            logger.error(f"Failed to send shutdown notification to dashboard: {e}")

    async def _send_warning_notification(self, details: Dict[str, Any]) -> None:
        """
        Send warning notification to dashboard.

        Args:
            details: Cost limit details
        """
        try:
            from dashboard.backend.dashboard_manager import DashboardUpdate, UpdateType

            update = DashboardUpdate(
                update_type=UpdateType.COST_UPDATE,
                data={
                    "event": "BUDGET_WARNING",
                    "severity": "warning",
                    "daily_cost_usd": details["daily_cost_usd"],
                    "daily_limit_usd": details["daily_limit_usd"],
                    "utilization_percent": details["utilization_percent"],
                    "remaining_budget_usd": details["remaining_budget_usd"],
                    "message": f"Budget utilization at {details['utilization_percent']:.1f}%"
                },
                timestamp=datetime.utcnow(),
                priority=1
            )

            await self.dashboard_manager.broadcast(update)

        except Exception as e:
            logger.error(f"Failed to send warning notification to dashboard: {e}")

    async def _send_cost_update(self, details: Dict[str, Any]) -> None:
        """
        Send routine cost update to dashboard.

        Args:
            details: Cost limit details
        """
        try:
            await self.dashboard_manager.update_costs(
                total_cost_usd=details["daily_cost_usd"],
                total_tokens=0,  # Not tracked in this update
                breakdown={
                    "daily_limit_usd": details["daily_limit_usd"],
                    "utilization_percent": details["utilization_percent"],
                    "remaining_budget_usd": details["remaining_budget_usd"]
                }
            )

        except Exception as e:
            logger.error(f"Failed to send cost update to dashboard: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with monitoring status and statistics
        """
        return {
            "is_running": self.is_running,
            "daily_limit_usd": self.daily_limit_usd,
            "check_interval_seconds": self.check_interval_seconds,
            "warning_threshold_percent": self.warning_threshold_pct,
            "check_count": self.check_count,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "statistics": self.stats.copy()
        }

    def update_limit(self, new_limit_usd: float) -> None:
        """
        Update the daily cost limit.

        Use with caution. This should only be changed with HOTL approval.

        Args:
            new_limit_usd: New daily limit in USD
        """
        old_limit = self.daily_limit_usd
        self.daily_limit_usd = new_limit_usd

        logger.warning(
            f"⚠️  Daily cost limit updated: ${old_limit:.2f} → ${new_limit_usd:.2f}"
        )

        # Reset warning flag to allow new warning at new threshold
        self.warning_sent = False

    def __repr__(self) -> str:
        status = "RUNNING" if self.is_running else "STOPPED"
        return (
            f"CostMonitor("
            f"status={status}, "
            f"limit=${self.daily_limit_usd:.2f}, "
            f"checks={self.check_count})"
        )


# Convenience function for creating and starting a cost monitor
async def start_cost_monitoring(
    token_tracker: TokenTracker,
    emergency_shutdown: EmergencyShutdown,
    daily_limit_usd: float = 15.00,
    **kwargs
) -> CostMonitor:
    """
    Convenience function to create and start a cost monitor.

    Args:
        token_tracker: TokenTracker instance
        emergency_shutdown: EmergencyShutdown manager
        daily_limit_usd: Daily cost limit in USD
        **kwargs: Additional arguments for CostMonitor

    Returns:
        Started CostMonitor instance

    Example:
        monitor = await start_cost_monitoring(
            token_tracker=tracker,
            emergency_shutdown=shutdown,
            daily_limit_usd=15.00,
            dashboard_manager=dashboard
        )
    """
    monitor = CostMonitor(
        token_tracker=token_tracker,
        emergency_shutdown=emergency_shutdown,
        daily_limit_usd=daily_limit_usd,
        **kwargs
    )

    await monitor.start()

    return monitor
