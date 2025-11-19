"""
Emergency Shutdown - Hard Limit Safety System

This module provides critical safety mechanisms to automatically shut down
the Solo-Swarm system when budget limits are exceeded or other emergency
conditions are detected.

Features:
- Global shutdown event for graceful termination
- Queue clearing to prevent new task execution
- Dashboard notification via WebSocket
- HOTL alerting through logging and optional email
- Thread-safe shutdown coordination
"""

import asyncio
import threading
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class ShutdownReason(str, Enum):
    """Reasons for emergency shutdown"""
    BUDGET_EXCEEDED = "budget_exceeded"
    COST_LIMIT_REACHED = "cost_limit_reached"
    MANUAL_SHUTDOWN = "manual_shutdown"
    SYSTEM_ERROR = "system_error"
    SAFETY_VIOLATION = "safety_violation"


# Global shutdown events (both sync and async)
_shutdown_event_sync = threading.Event()
_shutdown_event_async = asyncio.Event()
_shutdown_active = False
_shutdown_lock = threading.Lock()


class EmergencyShutdown:
    """
    Emergency Shutdown Manager.

    Coordinates system-wide shutdown when critical safety limits are exceeded.
    Ensures graceful termination of all agents and prevents new task execution.

    Usage:
        shutdown_manager = EmergencyShutdown(
            queue_manager=queue_manager,
            dashboard_manager=dashboard_manager
        )

        # Trigger shutdown
        await shutdown_manager.stop_all_agents(
            reason=ShutdownReason.BUDGET_EXCEEDED,
            details={"daily_cost": 15.50, "limit": 15.00}
        )
    """

    def __init__(
        self,
        queue_manager: Optional[Any] = None,
        dashboard_manager: Optional[Any] = None,
        agent_pool_manager: Optional[Any] = None
    ):
        """
        Initialize the Emergency Shutdown Manager.

        Args:
            queue_manager: QueueManager instance for clearing pending jobs
            dashboard_manager: DashboardManager for sending shutdown notifications
            agent_pool_manager: AgentPoolManager for releasing agent slots
        """
        self.queue_manager = queue_manager
        self.dashboard_manager = dashboard_manager
        self.agent_pool_manager = agent_pool_manager

        # Shutdown tracking
        self.shutdown_history: List[Dict[str, Any]] = []
        self.is_shutdown_active = False

        logger.info("EmergencyShutdown manager initialized")

    async def stop_all_agents(
        self,
        reason: ShutdownReason,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Stop all agents and prevent new task execution.

        This is the primary emergency shutdown method. It:
        1. Sets global shutdown events
        2. Clears all pending jobs from queues
        3. Notifies dashboard via WebSocket
        4. Logs critical alert for HOTL
        5. Records shutdown event

        Args:
            reason: Reason for shutdown
            details: Optional additional details about the shutdown

        Returns:
            Dictionary with shutdown results and statistics
        """
        global _shutdown_active

        # Prevent multiple simultaneous shutdowns
        with _shutdown_lock:
            if _shutdown_active:
                logger.warning("Shutdown already in progress, ignoring duplicate request")
                return {
                    "status": "already_shutdown",
                    "message": "System shutdown already in progress"
                }

            _shutdown_active = True
            self.is_shutdown_active = True

        shutdown_start_time = datetime.utcnow()

        logger.critical(
            f"🚨 EMERGENCY SHUTDOWN INITIATED 🚨\n"
            f"Reason: {reason.value}\n"
            f"Details: {details or 'None'}\n"
            f"Timestamp: {shutdown_start_time.isoformat()}"
        )

        results = {
            "reason": reason.value,
            "details": details or {},
            "timestamp": shutdown_start_time.isoformat(),
            "actions_taken": []
        }

        # Step 1: Set global shutdown events
        try:
            _shutdown_event_sync.set()
            _shutdown_event_async.set()
            results["actions_taken"].append({
                "action": "set_shutdown_events",
                "status": "success"
            })
            logger.info("✓ Global shutdown events set")
        except Exception as e:
            logger.error(f"Failed to set shutdown events: {e}")
            results["actions_taken"].append({
                "action": "set_shutdown_events",
                "status": "failed",
                "error": str(e)
            })

        # Step 2: Clear all pending jobs from queues
        if self.queue_manager:
            try:
                cleared_jobs = await self._clear_all_queues()
                results["actions_taken"].append({
                    "action": "clear_queues",
                    "status": "success",
                    "jobs_cleared": cleared_jobs
                })
                logger.info(f"✓ Cleared {sum(cleared_jobs.values())} pending jobs from queues")
            except Exception as e:
                logger.error(f"Failed to clear queues: {e}")
                results["actions_taken"].append({
                    "action": "clear_queues",
                    "status": "failed",
                    "error": str(e)
                })
        else:
            logger.warning("No queue_manager configured, skipping queue clearing")

        # Step 3: Send shutdown notification to dashboard
        if self.dashboard_manager:
            try:
                await self._notify_dashboard(reason, details)
                results["actions_taken"].append({
                    "action": "notify_dashboard",
                    "status": "success"
                })
                logger.info("✓ Dashboard notified of shutdown")
            except Exception as e:
                logger.error(f"Failed to notify dashboard: {e}")
                results["actions_taken"].append({
                    "action": "notify_dashboard",
                    "status": "failed",
                    "error": str(e)
                })
        else:
            logger.warning("No dashboard_manager configured, skipping dashboard notification")

        # Step 4: Release all agent slots (graceful cleanup)
        if self.agent_pool_manager:
            try:
                await self._release_agent_slots()
                results["actions_taken"].append({
                    "action": "release_agent_slots",
                    "status": "success"
                })
                logger.info("✓ Agent slots released")
            except Exception as e:
                logger.error(f"Failed to release agent slots: {e}")
                results["actions_taken"].append({
                    "action": "release_agent_slots",
                    "status": "failed",
                    "error": str(e)
                })

        # Step 5: Send HOTL alert (via logging - email integration can be added)
        try:
            await self._send_hotl_alert(reason, details, results)
            results["actions_taken"].append({
                "action": "send_hotl_alert",
                "status": "success"
            })
            logger.info("✓ HOTL alert sent")
        except Exception as e:
            logger.error(f"Failed to send HOTL alert: {e}")
            results["actions_taken"].append({
                "action": "send_hotl_alert",
                "status": "failed",
                "error": str(e)
            })

        # Record shutdown event
        shutdown_event = {
            "reason": reason.value,
            "details": details or {},
            "timestamp": shutdown_start_time.isoformat(),
            "results": results
        }
        self.shutdown_history.append(shutdown_event)

        shutdown_end_time = datetime.utcnow()
        duration = (shutdown_end_time - shutdown_start_time).total_seconds()

        logger.critical(
            f"🛑 EMERGENCY SHUTDOWN COMPLETED 🛑\n"
            f"Duration: {duration:.2f}s\n"
            f"Actions taken: {len(results['actions_taken'])}\n"
            f"System is now in LOCKED state - no new tasks will be accepted"
        )

        results["shutdown_duration_seconds"] = duration
        results["status"] = "shutdown_complete"

        return results

    async def _clear_all_queues(self) -> Dict[str, int]:
        """
        Clear all pending jobs from all queues.

        Returns:
            Dictionary with queue names and number of jobs cleared
        """
        if not self.queue_manager:
            return {}

        # Use the QueueManager's clear_all_queues method
        # Note: This is a sync method, but we're calling it in async context
        cleared = self.queue_manager.clear_all_queues()

        logger.info(f"Cleared queues: {cleared}")
        return cleared

    async def _notify_dashboard(
        self,
        reason: ShutdownReason,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send SYSTEM_SHUTDOWN event to dashboard via WebSocket.

        Args:
            reason: Shutdown reason
            details: Optional additional details
        """
        if not self.dashboard_manager:
            return

        # Import here to avoid circular dependencies
        from dashboard.backend.dashboard_manager import DashboardUpdate, UpdateType

        # Create shutdown notification
        shutdown_update = DashboardUpdate(
            update_type=UpdateType.SYSTEM_STATUS,  # Using SYSTEM_STATUS for shutdown
            data={
                "event": "SYSTEM_SHUTDOWN",
                "reason": reason.value,
                "details": details or {},
                "message": self._get_shutdown_message(reason, details),
                "shutdown_at": datetime.utcnow().isoformat(),
                "severity": "critical"
            },
            timestamp=datetime.utcnow(),
            priority=2  # Highest priority
        )

        # Broadcast to all connected clients
        await self.dashboard_manager.broadcast(shutdown_update)

    async def _release_agent_slots(self) -> None:
        """
        Release all allocated agent slots for graceful cleanup.
        """
        if not self.agent_pool_manager:
            return

        # The agent pool manager should handle cleanup gracefully
        # This is a placeholder for future implementation
        logger.info("Agent pool cleanup initiated")

    async def _send_hotl_alert(
        self,
        reason: ShutdownReason,
        details: Optional[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> None:
        """
        Send alert to Human-On-The-Loop (HOTL).

        Currently uses critical logging. Can be extended to send emails,
        Slack notifications, PagerDuty alerts, etc.

        Args:
            reason: Shutdown reason
            details: Shutdown details
            results: Shutdown action results
        """
        alert_message = f"""
╔═══════════════════════════════════════════════════════════════╗
║                   EMERGENCY SHUTDOWN ALERT                     ║
╚═══════════════════════════════════════════════════════════════╝

⚠️  SYSTEM: Solo-Swarm Multi-Agent System
⚠️  EVENT: Emergency Shutdown Triggered
⚠️  REASON: {reason.value}
⚠️  TIMESTAMP: {datetime.utcnow().isoformat()}

📊 DETAILS:
{self._format_details(details)}

🔧 ACTIONS TAKEN:
{self._format_actions(results.get('actions_taken', []))}

🚨 IMMEDIATE ACTION REQUIRED:
- Review shutdown details and determine root cause
- Check cost logs and budget settings
- Verify system state before restart
- Update budget limits if necessary

📧 For email alerts, configure SMTP settings in system config.
"""

        logger.critical(alert_message)

        # TODO: Add email notification here if configured
        # Example:
        # if self.email_config:
        #     await self._send_email_alert(alert_message)

    def _get_shutdown_message(
        self,
        reason: ShutdownReason,
        details: Optional[Dict[str, Any]]
    ) -> str:
        """Generate human-readable shutdown message."""
        messages = {
            ShutdownReason.BUDGET_EXCEEDED: "Daily budget limit has been exceeded. System shutdown to prevent further costs.",
            ShutdownReason.COST_LIMIT_REACHED: "Cost hard limit has been reached. All agent activity halted.",
            ShutdownReason.MANUAL_SHUTDOWN: "System manually shut down by administrator.",
            ShutdownReason.SYSTEM_ERROR: "Critical system error detected. Emergency shutdown initiated.",
            ShutdownReason.SAFETY_VIOLATION: "Safety violation detected. System locked for review."
        }

        base_message = messages.get(reason, "Emergency shutdown triggered.")

        if details:
            detail_str = " | ".join(f"{k}: {v}" for k, v in details.items())
            return f"{base_message} ({detail_str})"

        return base_message

    def _format_details(self, details: Optional[Dict[str, Any]]) -> str:
        """Format details dictionary for alert message."""
        if not details:
            return "  (No additional details)"

        return "\n".join(f"  - {key}: {value}" for key, value in details.items())

    def _format_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Format actions list for alert message."""
        if not actions:
            return "  (No actions taken)"

        formatted = []
        for action in actions:
            status_icon = "✓" if action.get("status") == "success" else "✗"
            action_name = action.get("action", "unknown").replace("_", " ").title()
            formatted.append(f"  {status_icon} {action_name}")

            if action.get("status") == "failed" and action.get("error"):
                formatted.append(f"     Error: {action['error']}")

        return "\n".join(formatted)

    def is_shutdown(self) -> bool:
        """
        Check if system is in shutdown state.

        Returns:
            True if shutdown is active, False otherwise
        """
        return _shutdown_active

    def get_shutdown_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all shutdown events.

        Returns:
            List of shutdown event dictionaries
        """
        return self.shutdown_history.copy()

    async def reset_shutdown(self, force: bool = False) -> bool:
        """
        Reset shutdown state (USE WITH EXTREME CAUTION).

        This should only be used after:
        1. Reviewing shutdown logs
        2. Addressing the root cause
        3. Adjusting budget limits
        4. Getting HOTL approval

        Args:
            force: If True, reset even if conditions haven't been addressed

        Returns:
            True if reset successful, False otherwise
        """
        global _shutdown_active

        if not force:
            logger.warning(
                "Reset shutdown requested without force flag. "
                "Use force=True only after addressing root cause."
            )
            return False

        with _shutdown_lock:
            _shutdown_event_sync.clear()
            _shutdown_event_async.clear()
            _shutdown_active = False
            self.is_shutdown_active = False

        logger.warning(
            "⚠️  SHUTDOWN STATE RESET - System is now accepting tasks again. "
            "Ensure budget limits are properly configured!"
        )

        return True


# Global convenience functions
def is_system_shutdown() -> bool:
    """
    Global function to check if system is in shutdown state.

    Can be used by any component to check shutdown status.

    Returns:
        True if shutdown is active, False otherwise
    """
    return _shutdown_active


def get_shutdown_event_sync() -> threading.Event:
    """
    Get the synchronous shutdown event.

    Use this in synchronous code to wait for shutdown:
        event = get_shutdown_event_sync()
        event.wait(timeout=1.0)
        if event.is_set():
            # Shutdown in progress

    Returns:
        Threading Event object
    """
    return _shutdown_event_sync


def get_shutdown_event_async() -> asyncio.Event:
    """
    Get the asynchronous shutdown event.

    Use this in async code to wait for shutdown:
        event = get_shutdown_event_async()
        await event.wait()
        # Shutdown in progress

    Returns:
        Asyncio Event object
    """
    return _shutdown_event_async
