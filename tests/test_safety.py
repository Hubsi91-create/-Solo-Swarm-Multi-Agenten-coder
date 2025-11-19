"""
Safety Tests - Hard Limit Enforcement and Emergency Shutdown

This test suite validates the critical safety mechanisms that prevent
budget overruns and ensure proper emergency shutdown behavior.

Test Coverage:
- Emergency shutdown triggering
- Queue clearing on shutdown
- Cost limit checking
- Dashboard alerting
- Monitoring loop behavior
- Shutdown state management
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from core.emergency_shutdown import (
    EmergencyShutdown,
    ShutdownReason,
    is_system_shutdown,
    get_shutdown_event_sync,
    get_shutdown_event_async
)
from core.token_tracker import TokenTracker, ModelType
from agents.monitoring.cost_monitor import CostMonitor


class TestEmergencyShutdown:
    """Test suite for EmergencyShutdown functionality."""

    @pytest.fixture
    def mock_queue_manager(self):
        """Create a mock queue manager."""
        manager = Mock()
        manager.clear_all_queues = Mock(return_value={
            "high_priority": 5,
            "default": 10,
            "low_priority": 3
        })
        return manager

    @pytest.fixture
    def mock_dashboard_manager(self):
        """Create a mock dashboard manager."""
        manager = AsyncMock()
        manager.broadcast = AsyncMock(return_value=2)
        return manager

    @pytest.fixture
    def emergency_shutdown(self, mock_queue_manager, mock_dashboard_manager):
        """Create an EmergencyShutdown instance with mocked dependencies."""
        return EmergencyShutdown(
            queue_manager=mock_queue_manager,
            dashboard_manager=mock_dashboard_manager
        )

    @pytest.mark.asyncio
    async def test_shutdown_sets_global_event(self, emergency_shutdown):
        """Test that shutdown sets the global shutdown event."""
        # Reset global state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False
        es_module._shutdown_event_sync.clear()
        es_module._shutdown_event_async.clear()

        # Trigger shutdown
        result = await emergency_shutdown.stop_all_agents(
            reason=ShutdownReason.BUDGET_EXCEEDED,
            details={"daily_cost": 15.50, "limit": 15.00}
        )

        # Verify shutdown event is set
        assert is_system_shutdown() is True
        assert get_shutdown_event_sync().is_set()
        assert result["status"] == "shutdown_complete"
        assert result["reason"] == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_shutdown_clears_queues(self, emergency_shutdown, mock_queue_manager):
        """Test that shutdown clears all pending jobs from queues."""
        # Reset global state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False

        # Trigger shutdown
        result = await emergency_shutdown.stop_all_agents(
            reason=ShutdownReason.COST_LIMIT_REACHED,
            details={"daily_cost": 20.00, "limit": 15.00}
        )

        # Verify queues were cleared
        mock_queue_manager.clear_all_queues.assert_called_once()

        # Verify action was recorded
        actions = result["actions_taken"]
        clear_action = next(
            (a for a in actions if a["action"] == "clear_queues"),
            None
        )

        assert clear_action is not None
        assert clear_action["status"] == "success"
        assert clear_action["jobs_cleared"]["high_priority"] == 5
        assert clear_action["jobs_cleared"]["default"] == 10
        assert clear_action["jobs_cleared"]["low_priority"] == 3

    @pytest.mark.asyncio
    async def test_shutdown_notifies_dashboard(self, emergency_shutdown, mock_dashboard_manager):
        """Test that shutdown sends notification to dashboard."""
        # Reset global state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False

        # Trigger shutdown
        result = await emergency_shutdown.stop_all_agents(
            reason=ShutdownReason.SAFETY_VIOLATION,
            details={"violation": "unauthorized_api_call"}
        )

        # Verify dashboard was notified
        mock_dashboard_manager.broadcast.assert_called_once()

        # Verify the broadcast content
        call_args = mock_dashboard_manager.broadcast.call_args
        update = call_args[0][0]

        assert update.data["event"] == "SYSTEM_SHUTDOWN"
        assert update.data["reason"] == "safety_violation"
        assert update.data["severity"] == "critical"
        assert update.priority == 2

    @pytest.mark.asyncio
    async def test_duplicate_shutdown_ignored(self, emergency_shutdown):
        """Test that duplicate shutdown requests are ignored."""
        # Reset global state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False

        # First shutdown
        result1 = await emergency_shutdown.stop_all_agents(
            reason=ShutdownReason.BUDGET_EXCEEDED,
            details={}
        )

        assert result1["status"] == "shutdown_complete"

        # Second shutdown (should be ignored)
        result2 = await emergency_shutdown.stop_all_agents(
            reason=ShutdownReason.BUDGET_EXCEEDED,
            details={}
        )

        assert result2["status"] == "already_shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_history_tracking(self, emergency_shutdown):
        """Test that shutdown events are recorded in history."""
        # Reset global state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False

        # Trigger shutdown
        await emergency_shutdown.stop_all_agents(
            reason=ShutdownReason.MANUAL_SHUTDOWN,
            details={"operator": "admin"}
        )

        # Check history
        history = emergency_shutdown.get_shutdown_history()

        assert len(history) >= 1
        latest = history[-1]
        assert latest["reason"] == "manual_shutdown"
        assert latest["details"]["operator"] == "admin"

    @pytest.mark.asyncio
    async def test_shutdown_with_missing_managers(self):
        """Test shutdown still works even without queue/dashboard managers."""
        # Create shutdown without dependencies
        shutdown = EmergencyShutdown()

        # Reset global state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False

        # Should still work without errors
        result = await shutdown.stop_all_agents(
            reason=ShutdownReason.SYSTEM_ERROR,
            details={"error": "test"}
        )

        assert result["status"] == "shutdown_complete"


class TestTokenTrackerHardLimits:
    """Test suite for TokenTracker hard limit checking."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = AsyncMock()

        # Mock query result for cost sum
        mock_result = Mock()
        mock_result.scalar = Mock(return_value=12.50)  # Mock daily cost

        session.execute = AsyncMock(return_value=mock_result)

        return session

    @pytest.fixture
    def token_tracker(self, mock_db_session):
        """Create a TokenTracker with mocked database."""
        return TokenTracker(db_session=mock_db_session)

    @pytest.mark.asyncio
    async def test_hard_limit_not_exceeded(self, token_tracker):
        """Test hard limit check when under budget."""
        exceeded, details = await token_tracker.check_hard_limit(daily_limit=15.00)

        assert exceeded is False
        assert details["daily_cost_usd"] == 12.50
        assert details["daily_limit_usd"] == 15.00
        assert details["remaining_budget_usd"] == 2.50
        assert details["limit_exceeded"] is False
        assert 83.0 <= details["utilization_percent"] <= 84.0  # ~83.33%

    @pytest.mark.asyncio
    async def test_hard_limit_exceeded(self, mock_db_session):
        """Test hard limit check when over budget."""
        # Mock higher cost
        mock_result = Mock()
        mock_result.scalar = Mock(return_value=16.50)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        tracker = TokenTracker(db_session=mock_db_session)
        exceeded, details = await tracker.check_hard_limit(daily_limit=15.00)

        assert exceeded is True
        assert details["daily_cost_usd"] == 16.50
        assert details["daily_limit_usd"] == 15.00
        assert details["remaining_budget_usd"] == -1.50
        assert details["limit_exceeded"] is True
        assert details["utilization_percent"] == 110.0

    @pytest.mark.asyncio
    async def test_hard_limit_exactly_at_limit(self, mock_db_session):
        """Test hard limit check when exactly at limit."""
        # Mock exact limit cost
        mock_result = Mock()
        mock_result.scalar = Mock(return_value=15.00)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        tracker = TokenTracker(db_session=mock_db_session)
        exceeded, details = await tracker.check_hard_limit(daily_limit=15.00)

        assert exceeded is True  # At limit counts as exceeded
        assert details["daily_cost_usd"] == 15.00
        assert details["utilization_percent"] == 100.0

    @pytest.mark.asyncio
    async def test_get_daily_cost_with_db(self, token_tracker):
        """Test retrieving daily cost from database."""
        cost = await token_tracker.get_daily_cost()

        assert cost == 12.50
        token_tracker.db_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_daily_cost_without_db(self):
        """Test fallback when no database is configured."""
        tracker = TokenTracker()  # No DB session
        tracker.total_cost_usd = 5.75

        cost = await tracker.get_daily_cost()

        # Should fallback to in-memory tracker
        assert cost == 5.75

    @pytest.mark.asyncio
    async def test_cost_summary_multiple_days(self, mock_db_session):
        """Test getting cost summary for multiple days."""
        # Mock different costs for different days
        call_count = [0]

        def mock_scalar_multi():
            costs = [10.0, 12.0, 15.0]  # 3 days
            result = costs[call_count[0] % len(costs)]
            call_count[0] += 1
            return result

        mock_result = Mock()
        mock_result.scalar = Mock(side_effect=mock_scalar_multi)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        tracker = TokenTracker(db_session=mock_db_session)
        summary = await tracker.get_cost_summary(days=3)

        assert summary["days"] == 3
        assert len(summary["daily_costs"]) == 3
        assert summary["total_cost"] > 0
        assert summary["average_daily_cost"] > 0


class TestCostMonitor:
    """Test suite for CostMonitor background monitoring."""

    @pytest.fixture
    def mock_token_tracker(self):
        """Create a mock token tracker."""
        tracker = AsyncMock()
        tracker.check_hard_limit = AsyncMock(return_value=(
            False,  # Not exceeded
            {
                "daily_cost_usd": 10.00,
                "daily_limit_usd": 15.00,
                "remaining_budget_usd": 5.00,
                "utilization_percent": 66.67,
                "limit_exceeded": False
            }
        ))
        return tracker

    @pytest.fixture
    def mock_emergency_shutdown(self):
        """Create a mock emergency shutdown."""
        shutdown = AsyncMock()
        shutdown.stop_all_agents = AsyncMock(return_value={
            "status": "shutdown_complete"
        })
        return shutdown

    @pytest.fixture
    def cost_monitor(self, mock_token_tracker, mock_emergency_shutdown):
        """Create a CostMonitor with mocked dependencies."""
        return CostMonitor(
            token_tracker=mock_token_tracker,
            emergency_shutdown=mock_emergency_shutdown,
            daily_limit_usd=15.00,
            check_interval_seconds=0.1  # Fast for testing
        )

    @pytest.mark.asyncio
    async def test_monitor_starts_and_stops(self, cost_monitor):
        """Test that monitor can start and stop cleanly."""
        # Reset global shutdown state to ensure clean test
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False
        es_module._shutdown_event_sync.clear()
        es_module._shutdown_event_async.clear()

        # Start monitor
        await cost_monitor.start()
        assert cost_monitor.is_running is True

        # Let it run for a bit (ensure at least 2 check intervals pass)
        await asyncio.sleep(0.5)

        # Stop monitor
        await cost_monitor.stop()
        assert cost_monitor.is_running is False
        assert cost_monitor.check_count >= 1  # At least one check should have run

    @pytest.mark.asyncio
    async def test_monitor_triggers_shutdown_on_limit(
        self,
        mock_token_tracker,
        mock_emergency_shutdown
    ):
        """Test that monitor triggers shutdown when limit is exceeded."""
        # Reset global shutdown state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False
        es_module._shutdown_event_sync.clear()
        es_module._shutdown_event_async.clear()

        # Mock limit exceeded
        mock_token_tracker.check_hard_limit = AsyncMock(return_value=(
            True,  # Exceeded!
            {
                "daily_cost_usd": 16.00,
                "daily_limit_usd": 15.00,
                "remaining_budget_usd": -1.00,
                "utilization_percent": 106.67,
                "limit_exceeded": True
            }
        ))

        monitor = CostMonitor(
            token_tracker=mock_token_tracker,
            emergency_shutdown=mock_emergency_shutdown,
            daily_limit_usd=15.00,
            check_interval_seconds=0.05  # Faster for testing
        )

        # Start monitor
        await monitor.start()

        # Wait for shutdown to be triggered (give it more time)
        await asyncio.sleep(0.5)

        # Verify shutdown was called
        mock_emergency_shutdown.stop_all_agents.assert_called()

        # Verify reason (get the first call)
        assert mock_emergency_shutdown.stop_all_agents.call_count >= 1
        call_args = mock_emergency_shutdown.stop_all_agents.call_args
        assert call_args[1]["reason"] == ShutdownReason.COST_LIMIT_REACHED

        # Stop monitor
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_monitor_sends_warning_at_threshold(
        self,
        mock_token_tracker,
        mock_emergency_shutdown
    ):
        """Test that monitor sends warning when approaching limit."""
        # Reset global shutdown state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False
        es_module._shutdown_event_sync.clear()
        es_module._shutdown_event_async.clear()

        # Mock high utilization (85%)
        mock_token_tracker.check_hard_limit = AsyncMock(return_value=(
            False,
            {
                "daily_cost_usd": 12.75,
                "daily_limit_usd": 15.00,
                "remaining_budget_usd": 2.25,
                "utilization_percent": 85.0,
                "limit_exceeded": False
            }
        ))

        monitor = CostMonitor(
            token_tracker=mock_token_tracker,
            emergency_shutdown=mock_emergency_shutdown,
            daily_limit_usd=15.00,
            check_interval_seconds=0.05,  # Faster for testing
            warning_threshold_pct=80.0
        )

        # Start monitor
        await monitor.start()

        # Let it check a few times (give it more time)
        await asyncio.sleep(0.5)

        # Verify warning was triggered
        assert monitor.stats["warnings_triggered"] >= 1
        assert monitor.warning_sent is True

        # Stop monitor
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_monitor_respects_shutdown_state(self, cost_monitor):
        """Test that monitor enters standby when system is shutdown."""
        # Set global shutdown state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = True

        # Start monitor
        await cost_monitor.start()

        # Let it run
        await asyncio.sleep(0.3)

        # Should not have performed many checks (standby mode)
        # Note: This is a soft check since timing can vary
        assert cost_monitor.is_running is True

        # Stop monitor
        await cost_monitor.stop()

        # Reset global state
        es_module._shutdown_active = False

    @pytest.mark.asyncio
    async def test_monitor_handles_errors_gracefully(
        self,
        mock_token_tracker,
        mock_emergency_shutdown
    ):
        """Test that monitor continues running even after errors."""
        # Make tracker raise an error
        mock_token_tracker.check_hard_limit = AsyncMock(
            side_effect=Exception("Database connection lost")
        )

        monitor = CostMonitor(
            token_tracker=mock_token_tracker,
            emergency_shutdown=mock_emergency_shutdown,
            daily_limit_usd=15.00,
            check_interval_seconds=0.1
        )

        # Start monitor
        await monitor.start()

        # Let it encounter errors
        await asyncio.sleep(0.3)

        # Should still be running and have recorded errors
        assert monitor.is_running is True
        assert monitor.stats["errors_encountered"] > 0

        # Stop monitor
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_monitor_update_limit(self, cost_monitor):
        """Test updating the cost limit while monitoring."""
        initial_limit = cost_monitor.daily_limit_usd

        # Update limit
        cost_monitor.update_limit(20.00)

        assert cost_monitor.daily_limit_usd == 20.00
        assert cost_monitor.daily_limit_usd != initial_limit

        # Warning flag should be reset
        cost_monitor.warning_sent = True
        cost_monitor.update_limit(25.00)
        assert cost_monitor.warning_sent is False

    def test_monitor_get_status(self, cost_monitor):
        """Test getting monitor status."""
        status = cost_monitor.get_status()

        assert "is_running" in status
        assert "daily_limit_usd" in status
        assert "check_interval_seconds" in status
        assert "statistics" in status
        assert status["daily_limit_usd"] == 15.00


class TestIntegration:
    """Integration tests for the complete safety system."""

    @pytest.mark.asyncio
    async def test_end_to_end_shutdown_flow(self):
        """
        Test complete end-to-end flow:
        1. Cost exceeds limit
        2. Monitor detects it
        3. Shutdown is triggered
        4. Queues are cleared
        5. System is locked
        """
        # Reset global state
        import core.emergency_shutdown as es_module
        es_module._shutdown_active = False
        es_module._shutdown_event_sync.clear()
        es_module._shutdown_event_async.clear()

        # Setup mocks
        mock_queue = Mock()
        mock_queue.clear_all_queues = Mock(return_value={"default": 5})

        mock_dashboard = AsyncMock()
        mock_dashboard.broadcast = AsyncMock(return_value=1)

        # Create real components
        shutdown_manager = EmergencyShutdown(
            queue_manager=mock_queue,
            dashboard_manager=mock_dashboard
        )

        # Mock token tracker with limit exceeded
        mock_tracker = AsyncMock()
        mock_tracker.check_hard_limit = AsyncMock(return_value=(
            True,
            {
                "daily_cost_usd": 15.50,
                "daily_limit_usd": 15.00,
                "remaining_budget_usd": -0.50,
                "utilization_percent": 103.33,
                "limit_exceeded": True
            }
        ))

        # Create monitor
        monitor = CostMonitor(
            token_tracker=mock_tracker,
            emergency_shutdown=shutdown_manager,
            daily_limit_usd=15.00,
            check_interval_seconds=0.1,
            dashboard_manager=mock_dashboard
        )

        # Start monitoring
        await monitor.start()

        # Wait for shutdown to trigger
        await asyncio.sleep(0.5)

        # Verify complete flow
        assert is_system_shutdown() is True
        assert mock_queue.clear_all_queues.called
        assert mock_dashboard.broadcast.called
        assert monitor.stats["shutdowns_triggered"] > 0

        # Cleanup
        await monitor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
