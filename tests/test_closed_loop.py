"""
Closed Loop Tests - Webhook, QA Agent, and Auto-Merge

This test suite validates the autonomous fix-test-merge cycle:
- Modl.ai webhook reception and parsing
- QA Agent report analysis and task creation
- Closed Loop Manager orchestration
- Auto-merge decision logic
- HOTL review triggering
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from integrations.modl_ai import (
    ModlAiPayload,
    CrashReport,
    BalanceIssue,
    PerformanceIssue,
    TestFailure,
    ReportType,
    Severity,
    _generate_simulated_report
)
from agents.workers.qa_agent import QAAgent, ConfidenceLevel
from core.closed_loop import (
    ClosedLoopManager,
    ClosedLoopCycle,
    LoopStatus,
    AutoMergeDecision
)
from core.tdf_schema import TaskDefinition, TaskType


class TestModlAiWebhook:
    """Test suite for Modl.ai webhook integration."""

    def test_generate_crash_report(self):
        """Test generating simulated crash report."""
        report = _generate_simulated_report(ReportType.CRASH, Severity.HIGH)

        assert report.report_type == ReportType.CRASH
        assert report.severity == Severity.HIGH
        assert report.crash_report is not None
        assert report.crash_report.exception_type == "NullReferenceException"
        assert report.crash_report.frequency > 0
        assert "GameManager" in report.crash_report.stack_trace

    def test_generate_balance_report(self):
        """Test generating simulated balance report."""
        report = _generate_simulated_report(ReportType.BALANCE, Severity.MEDIUM)

        assert report.report_type == ReportType.BALANCE
        assert report.balance_issue is not None
        assert report.balance_issue.metric_name == "enemy_difficulty"
        assert report.balance_issue.expected_value == 50.0
        assert report.balance_issue.actual_value == 85.0
        assert report.balance_issue.deviation_percent == 70.0

    def test_generate_performance_report(self):
        """Test generating simulated performance report."""
        report = _generate_simulated_report(ReportType.PERFORMANCE, Severity.HIGH)

        assert report.report_type == ReportType.PERFORMANCE
        assert report.performance_issue is not None
        assert report.performance_issue.metric_type == "fps"
        assert report.performance_issue.threshold == 60.0
        assert report.performance_issue.measured_value < report.performance_issue.threshold

    def test_generate_test_failure_report(self):
        """Test generating simulated test failure report."""
        report = _generate_simulated_report(ReportType.UNIT_TEST, Severity.MEDIUM)

        assert report.report_type == ReportType.UNIT_TEST
        assert report.test_failure is not None
        assert report.test_failure.test_name == "test_enemy_spawn_rate"
        assert report.test_failure.test_type == "unit"
        assert report.test_failure.file_path is not None

    def test_crash_report_validation(self):
        """Test crash report data validation."""
        crash = CrashReport(
            exception_type="NullReferenceException",
            message="Test error",
            stack_trace="at Test.cs:42",
            frequency=5,
            affected_users=2,
            first_seen=datetime.utcnow().isoformat(),
            last_seen=datetime.utcnow().isoformat()
        )

        assert crash.exception_type == "NullReferenceException"
        assert crash.frequency == 5
        assert crash.affected_users == 2

    def test_payload_serialization(self):
        """Test payload can be serialized to dict."""
        report = _generate_simulated_report(ReportType.CRASH, Severity.CRITICAL)
        report_dict = report.model_dump()

        assert report_dict['report_type'] == 'crash'
        assert report_dict['severity'] == 'critical'
        assert 'crash_report' in report_dict
        assert report_dict['crash_report'] is not None


class TestQAAgent:
    """Test suite for QA Agent report parsing and task creation."""

    @pytest.fixture
    def qa_agent(self):
        """Create a QA Agent instance."""
        return QAAgent(
            agent_id="qa_test_001",
            config={"auto_merge_threshold": 90.0}
        )

    @pytest.fixture
    def crash_report(self):
        """Create a simulated crash report."""
        return _generate_simulated_report(ReportType.CRASH, Severity.HIGH)

    @pytest.fixture
    def balance_report(self):
        """Create a simulated balance report."""
        return _generate_simulated_report(ReportType.BALANCE, Severity.MEDIUM)

    @pytest.fixture
    def test_failure_report(self):
        """Create a simulated test failure report."""
        return _generate_simulated_report(ReportType.UNIT_TEST, Severity.LOW)

    def test_analyze_crash_report(self, qa_agent, crash_report):
        """Test QA Agent analyzing crash report."""
        tasks = qa_agent.analyze_report(crash_report)

        assert len(tasks) > 0
        task = tasks[0]
        assert task.task_type == TaskType.IMPLEMENTATION
        assert "NullReferenceException" in task.description
        assert task.context['exception_type'] == "NullReferenceException"
        assert 'confidence_score' in task.metadata
        assert 'auto_merge_eligible' in task.metadata

    def test_analyze_balance_report(self, qa_agent, balance_report):
        """Test QA Agent analyzing balance report."""
        tasks = qa_agent.analyze_report(balance_report)

        assert len(tasks) > 0
        task = tasks[0]
        assert task.task_type == TaskType.REFACTORING
        assert "enemy_difficulty" in task.description
        assert task.context['metric_name'] == "enemy_difficulty"
        assert task.context['expected_value'] == 50.0
        assert task.context['actual_value'] == 85.0

    def test_analyze_test_failure_report(self, qa_agent, test_failure_report):
        """Test QA Agent analyzing test failure report."""
        tasks = qa_agent.analyze_report(test_failure_report)

        assert len(tasks) > 0
        task = tasks[0]
        assert task.task_type == TaskType.IMPLEMENTATION
        assert "test_enemy_spawn_rate" in task.description
        assert task.context['test_name'] == "test_enemy_spawn_rate"
        assert task.context['test_type'] == "unit"

    def test_confidence_score_crash(self, qa_agent, crash_report):
        """Test confidence scoring for crash fixes."""
        tasks = qa_agent.analyze_report(crash_report)

        assert len(tasks) > 0
        confidence = tasks[0].metadata['confidence_score']

        # Crash with clear stack trace should have decent confidence
        assert 0 <= confidence <= 100
        # High severity should reduce confidence
        assert confidence < 90  # Not auto-merge for high severity

    def test_confidence_score_balance(self, qa_agent, balance_report):
        """Test confidence scoring for balance fixes."""
        tasks = qa_agent.analyze_report(balance_report)

        assert len(tasks) > 0
        confidence = tasks[0].metadata['confidence_score']

        # Balance fixes with suggested fix should have higher confidence
        assert 0 <= confidence <= 100
        # Config-based fixes should boost confidence
        assert confidence > 50

    def test_stack_trace_parsing(self, qa_agent):
        """Test stack trace parsing for file location extraction."""
        # Test C# style
        location = qa_agent._parse_stack_trace(
            "at GameManager.SpawnEnemy() in GameManager.cs:line 42"
        )
        assert location is not None
        assert "GameManager.cs" in location['file']
        assert location['line'] == 42

        # Test Python style
        location = qa_agent._parse_stack_trace(
            'File "test.py", line 123'
        )
        assert location is not None
        assert "test.py" in location['file']
        assert location['line'] == 123

    def test_severity_to_priority_mapping(self, qa_agent):
        """Test severity to priority conversion."""
        assert qa_agent._severity_to_priority(Severity.CRITICAL) == 1
        assert qa_agent._severity_to_priority(Severity.HIGH) == 3
        assert qa_agent._severity_to_priority(Severity.MEDIUM) == 5
        assert qa_agent._severity_to_priority(Severity.LOW) == 7
        assert qa_agent._severity_to_priority(Severity.INFO) == 9

    def test_high_confidence_factors(self, qa_agent):
        """Test factors that increase confidence score."""
        # Create a low-severity crash with clear location
        report = ModlAiPayload(
            report_type=ReportType.CRASH,
            severity=Severity.LOW,  # Low severity = higher confidence
            timestamp=datetime.utcnow().isoformat(),
            project_id="test",
            version="1.0.0",
            crash_report=CrashReport(
                exception_type="NullReferenceException",
                message="Single occurrence with clear location",
                stack_trace="at Test.cs:line 42",
                frequency=1,  # Single occurrence = higher confidence
                affected_users=0,  # No affected users = higher confidence
                first_seen=datetime.utcnow().isoformat(),
                last_seen=datetime.utcnow().isoformat()
            )
        )

        tasks = qa_agent.analyze_report(report)
        assert len(tasks) > 0

        confidence = tasks[0].metadata['confidence_score']
        # Should have high confidence due to:
        # - Low severity
        # - Single occurrence
        # - No affected users
        # - Clear stack trace
        assert confidence >= 70


class TestClosedLoopManager:
    """Test suite for Closed Loop Manager orchestration."""

    @pytest.fixture
    def qa_agent(self):
        """Create a QA Agent."""
        return QAAgent(agent_id="qa_test_001")

    @pytest.fixture
    def closed_loop_manager(self, qa_agent):
        """Create a Closed Loop Manager."""
        return ClosedLoopManager(
            qa_agent=qa_agent,
            config={
                "auto_merge_threshold": 90.0,
                "min_test_coverage": 80.0
            }
        )

    @pytest.fixture
    def high_confidence_report(self):
        """Create a report that should trigger auto-merge."""
        return ModlAiPayload(
            report_type=ReportType.BALANCE,  # Config-based = higher confidence
            severity=Severity.LOW,  # Low severity = higher confidence
            timestamp=datetime.utcnow().isoformat(),
            project_id="test",
            version="1.0.0",
            balance_issue=BalanceIssue(
                metric_name="test_metric",
                expected_value=50.0,
                actual_value=55.0,
                deviation_percent=10.0,
                sample_size=1000,
                context="Small deviation",
                suggested_fix="Adjust config value"  # Suggested fix = higher confidence
            )
        )

    @pytest.fixture
    def low_confidence_report(self):
        """Create a report that should require HOTL review."""
        return ModlAiPayload(
            report_type=ReportType.CRASH,
            severity=Severity.CRITICAL,  # Critical = lower confidence
            timestamp=datetime.utcnow().isoformat(),
            project_id="test",
            version="1.0.0",
            crash_report=CrashReport(
                exception_type="SystemException",
                message="Complex crash",
                stack_trace="Unknown location",  # No clear location = lower confidence
                frequency=50,  # High frequency = complex issue
                affected_users=100,  # Many users = critical
                first_seen=datetime.utcnow().isoformat(),
                last_seen=datetime.utcnow().isoformat()
            )
        )

    @pytest.mark.asyncio
    async def test_process_report_flow(self, closed_loop_manager, high_confidence_report):
        """Test complete report processing flow."""
        cycle = await closed_loop_manager.process_report(high_confidence_report)

        assert cycle is not None
        assert cycle.cycle_id is not None
        assert cycle.status in [LoopStatus.AUTO_MERGED, LoopStatus.HOTL_REVIEW, LoopStatus.COMPLETED]
        assert len(cycle.tasks) > 0
        assert cycle.merge_decision is not None
        assert cycle.completed_at is not None

    @pytest.mark.asyncio
    async def test_auto_merge_decision_high_confidence(
        self,
        closed_loop_manager,
        high_confidence_report
    ):
        """Test auto-merge decision for high confidence fix."""
        cycle = await closed_loop_manager.process_report(high_confidence_report)

        # With high confidence and passing tests, should auto-merge
        # Note: This depends on simulated test results
        assert cycle.merge_decision in [AutoMergeDecision.AUTO_APPROVE, AutoMergeDecision.HOTL_REQUIRED]

    @pytest.mark.asyncio
    async def test_hotl_review_required_low_confidence(
        self,
        closed_loop_manager,
        low_confidence_report
    ):
        """Test HOTL review trigger for low confidence fix."""
        cycle = await closed_loop_manager.process_report(low_confidence_report)

        # Low confidence should trigger HOTL review
        # (or fail if tests don't pass)
        assert cycle.merge_decision in [AutoMergeDecision.HOTL_REQUIRED, AutoMergeDecision.REJECTED]

        if cycle.merge_decision == AutoMergeDecision.HOTL_REQUIRED:
            assert cycle.status == LoopStatus.HOTL_REVIEW

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, closed_loop_manager, high_confidence_report):
        """Test statistics are tracked correctly."""
        initial_stats = closed_loop_manager.get_statistics()

        # Process a report
        await closed_loop_manager.process_report(high_confidence_report)

        updated_stats = closed_loop_manager.get_statistics()

        assert updated_stats['total_cycles'] == initial_stats['total_cycles'] + 1
        assert updated_stats['average_confidence'] >= 0

    @pytest.mark.asyncio
    async def test_cycle_tracking(self, closed_loop_manager, high_confidence_report):
        """Test cycle status tracking."""
        cycle = await closed_loop_manager.process_report(high_confidence_report)

        # Check cycle can be retrieved
        status = closed_loop_manager.get_cycle_status(cycle.cycle_id)
        assert status is not None
        assert status['cycle_id'] == cycle.cycle_id
        assert 'duration_seconds' in status

    @pytest.mark.asyncio
    async def test_multiple_concurrent_cycles(
        self,
        closed_loop_manager,
        high_confidence_report
    ):
        """Test handling multiple concurrent cycles."""
        # Process multiple reports concurrently
        cycles = await asyncio.gather(
            closed_loop_manager.process_report(high_confidence_report),
            closed_loop_manager.process_report(high_confidence_report),
            closed_loop_manager.process_report(high_confidence_report)
        )

        assert len(cycles) == 3
        assert all(c.completed_at is not None for c in cycles)

        # Check statistics
        stats = closed_loop_manager.get_statistics()
        assert stats['total_cycles'] >= 3

    def test_closed_loop_cycle_serialization(self, qa_agent, high_confidence_report):
        """Test ClosedLoopCycle can be serialized."""
        tasks = qa_agent.analyze_report(high_confidence_report)
        cycle = ClosedLoopCycle(
            cycle_id="test_001",
            report=high_confidence_report,
            tasks=tasks
        )

        cycle_dict = cycle.to_dict()

        assert cycle_dict['cycle_id'] == "test_001"
        assert cycle_dict['status'] == LoopStatus.PENDING.value
        assert cycle_dict['task_count'] == len(tasks)
        assert 'created_at' in cycle_dict


class TestIntegrationFlow:
    """Integration tests for the complete closed loop flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_auto_merge(self):
        """
        Test complete end-to-end flow with auto-merge:
        1. Receive Modl.ai report
        2. QA Agent parses and creates tasks
        3. Tasks are processed through fix-test-validate
        4. High confidence + passing tests = auto-merge
        """
        # Create components
        qa_agent = QAAgent(agent_id="qa_integration_test")
        manager = ClosedLoopManager(
            qa_agent=qa_agent,
            config={
                "auto_merge_threshold": 85.0,  # Lower for testing
                "min_test_coverage": 75.0
            }
        )

        # Create high-confidence report
        report = ModlAiPayload(
            report_type=ReportType.BALANCE,
            severity=Severity.LOW,
            timestamp=datetime.utcnow().isoformat(),
            project_id="integration_test",
            version="1.0.0",
            balance_issue=BalanceIssue(
                metric_name="test_balance",
                expected_value=100.0,
                actual_value=105.0,
                deviation_percent=5.0,
                sample_size=1000,
                context="Minor balance adjustment",
                suggested_fix="Reduce by 5%"
            )
        )

        # Process report
        cycle = await manager.process_report(report)

        # Verify complete flow
        assert cycle.status in [LoopStatus.AUTO_MERGED, LoopStatus.COMPLETED]
        assert len(cycle.tasks) > 0
        assert cycle.confidence_score > 0
        assert len(cycle.test_results) > 0
        assert cycle.merge_decision is not None

    @pytest.mark.asyncio
    async def test_end_to_end_hotl_review(self):
        """
        Test complete end-to-end flow requiring HOTL review:
        1. Receive critical crash report
        2. QA Agent creates tasks
        3. Low confidence or test issues
        4. HOTL review required
        """
        qa_agent = QAAgent(agent_id="qa_hotl_test")
        manager = ClosedLoopManager(
            qa_agent=qa_agent,
            config={"auto_merge_threshold": 90.0}
        )

        # Create critical crash report
        report = ModlAiPayload(
            report_type=ReportType.CRASH,
            severity=Severity.CRITICAL,
            timestamp=datetime.utcnow().isoformat(),
            project_id="hotl_test",
            version="1.0.0",
            crash_report=CrashReport(
                exception_type="CriticalException",
                message="System crash",
                stack_trace="Unknown",
                frequency=100,
                affected_users=1000,
                first_seen=datetime.utcnow().isoformat(),
                last_seen=datetime.utcnow().isoformat()
            )
        )

        # Process report
        cycle = await manager.process_report(report)

        # Should require HOTL review or be rejected
        assert cycle.merge_decision in [
            AutoMergeDecision.HOTL_REQUIRED,
            AutoMergeDecision.REJECTED
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
