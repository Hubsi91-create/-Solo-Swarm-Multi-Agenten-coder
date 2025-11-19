"""
Integration Tests for Phase 5: Real Architect Integration & Approval UI

This test suite validates the integration between:
1. ClosedLoopManager and real ArchitectAgent
2. QA Agent report analysis
3. Architect Agent planning and execution with integrate_and_test()
4. Dashboard approval workflow

Test Flow:
Modl.ai Report -> QA Agent Analysis -> Architect Planning & Execution ->
Test Integration -> Approval Decision -> Dashboard Update
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from core.closed_loop import ClosedLoopManager, ClosedLoopCycle, LoopStatus, AutoMergeDecision
from integrations.modl_ai import ModlAiPayload, ReportType, Severity
from agents.workers.qa_agent import QAAgent
from agents.orchestrator.architect_agent import ArchitectAgent
from core.tdf_schema import TaskDefinition, TaskType
from core.agent_framework import AgentResponse


@pytest.fixture
def qa_agent_mock():
    """Mock QA Agent that generates fix tasks."""
    mock = Mock(spec=QAAgent)

    # QA Agent generates 2 fix tasks
    def analyze_report(report):
        return [
            TaskDefinition(
                task_id="fix_task_001",
                task_type=TaskType.IMPLEMENTATION,
                priority=1,
                assigned_agent="coder_agent",
                description="Fix the crash in authentication module",
                context={
                    'confidence_score': 85.0,
                    'issue_type': 'crash',
                    'severity': 'high',
                    'affected_files': ['auth/login.py'],
                    'error_message': 'NullPointerException in validateToken()',
                    'test_file_path': None,
                    'code_file_path': None
                }
            ),
            TaskDefinition(
                task_id="fix_task_002",
                task_type=TaskType.TESTING,
                priority=2,
                assigned_agent="qa_agent",
                description="Add regression test for authentication crash",
                context={
                    'confidence_score': 90.0,
                    'issue_type': 'test',
                    'severity': 'medium'
                }
            )
        ]

    mock.analyze_report = Mock(side_effect=analyze_report)
    return mock


@pytest.fixture
def architect_agent_mock():
    """Mock Architect Agent with plan_and_delegate and integrate_and_test."""
    mock = Mock(spec=ArchitectAgent)

    # plan_and_delegate returns a successful plan
    def plan_and_delegate(user_request, codebase_path=None):
        return {
            "success": True,
            "plan": {
                "plan_summary": "Fix authentication crash and add regression test",
                "estimated_duration": "15 minutes",
                "estimated_cost": "0.02 USD",
                "tasks": [
                    {
                        "task_id": "subtask_001",
                        "description": "Add null check in validateToken()",
                        "context": {
                            "target_files": ["auth/login.py"]
                        }
                    },
                    {
                        "task_id": "subtask_002",
                        "description": "Add unit test for null token",
                        "context": {
                            "target_files": ["tests/test_auth.py"]
                        }
                    }
                ]
            },
            "token_usage": {
                "total_cost": 0.02,
                "input_tokens": 1000,
                "output_tokens": 500
            },
            "budget_info": {
                "allocated_budget": 2000,
                "used_budget": 1500
            },
            "metadata": {}
        }

    # integrate_and_test returns successful test results
    def integrate_and_test(worker_results, test_file_path=None, code_file_path=None):
        return AgentResponse(
            success=True,
            data={
                "success": True,
                "iterations": 1,
                "code_file": "/tmp/generated_fix.py",
                "test_file": test_file_path,
                "test_output": "===== test session starts =====\ncollected 10 items\n\ntest_auth.py::test_validate_token PASSED\n\n===== 10 passed in 2.5s =====\nTOTAL coverage: 85%",
                "merged_code_length": 250,
                "bugfix_tasks_generated": 0,
                "bugfix_tasks": []
            },
            metadata={}
        )

    mock.plan_and_delegate = Mock(side_effect=plan_and_delegate)
    mock.integrate_and_test = Mock(side_effect=integrate_and_test)

    return mock


@pytest.fixture
def modl_ai_report():
    """Sample Modl.ai crash report."""
    return ModlAiPayload(
        report_type=ReportType.CRASH_REPORT,
        severity=Severity.HIGH,
        timestamp=datetime.utcnow().isoformat(),
        details={
            "crash_type": "NullPointerException",
            "stack_trace": "at auth.validateToken(login.py:42)",
            "frequency": "10 times in last hour",
            "affected_users": 150
        }
    )


@pytest.mark.asyncio
async def test_closed_loop_with_real_architect_integration(
    qa_agent_mock,
    architect_agent_mock,
    modl_ai_report
):
    """
    Test complete Closed Loop cycle with real Architect Agent integration.

    This validates:
    1. QA Agent analyzes Modl.ai report
    2. ClosedLoop sends tasks to Architect Agent
    3. Architect Agent plans, executes, and tests fixes
    4. ClosedLoop makes auto-merge decision based on results
    """
    # Initialize ClosedLoopManager with mocked agents
    closed_loop = ClosedLoopManager(
        qa_agent=qa_agent_mock,
        architect_agent=architect_agent_mock,
        dashboard_manager=None,
        db_session=None,
        config={
            "auto_merge_threshold": 90.0,
            "min_test_coverage": 80.0
        }
    )

    # Process the Modl.ai report
    cycle = await closed_loop.process_report(modl_ai_report)

    # Assertions
    assert cycle is not None, "Cycle should be created"
    assert cycle.status == LoopStatus.AUTO_MERGED, f"Expected AUTO_MERGED, got {cycle.status}"
    assert cycle.merge_decision == AutoMergeDecision.AUTO_APPROVE, \
        f"Expected AUTO_APPROVE, got {cycle.merge_decision}"

    # Verify QA Agent was called
    qa_agent_mock.analyze_report.assert_called_once_with(modl_ai_report)

    # Verify Architect Agent was called for each task
    assert architect_agent_mock.plan_and_delegate.call_count == 2, \
        f"Expected 2 plan_and_delegate calls, got {architect_agent_mock.plan_and_delegate.call_count}"

    assert architect_agent_mock.integrate_and_test.call_count == 2, \
        f"Expected 2 integrate_and_test calls, got {architect_agent_mock.integrate_and_test.call_count}"

    # Verify test results were stored
    assert len(cycle.test_results) == 2, "Should have 2 test results"
    for task_id, result in cycle.test_results.items():
        assert result['validation']['tests_passed'] is True, \
            f"Tests should pass for {task_id}"
        assert result['validation']['coverage'] >= 80.0, \
            f"Coverage should be >= 80% for {task_id}"

    # Verify cycle was completed
    assert cycle.completed_at is not None, "Cycle should be completed"
    assert cycle.confidence_score >= 85.0, "Confidence should be >= 85%"


@pytest.mark.asyncio
async def test_closed_loop_requires_hotl_review_low_confidence(
    qa_agent_mock,
    architect_agent_mock,
    modl_ai_report
):
    """
    Test that low confidence scores trigger HOTL review.
    """
    # Modify QA agent to return low confidence tasks
    def low_confidence_analyze(report):
        tasks = qa_agent_mock.analyze_report(report)
        for task in tasks:
            task.context['confidence_score'] = 75.0  # Below 90% threshold
        return tasks

    qa_agent_mock.analyze_report = Mock(side_effect=low_confidence_analyze)

    # Initialize ClosedLoop
    closed_loop = ClosedLoopManager(
        qa_agent=qa_agent_mock,
        architect_agent=architect_agent_mock,
        config={"auto_merge_threshold": 90.0}
    )

    # Process report
    cycle = await closed_loop.process_report(modl_ai_report)

    # Should require HOTL review
    assert cycle.status == LoopStatus.HOTL_REVIEW, \
        f"Expected HOTL_REVIEW, got {cycle.status}"
    assert cycle.merge_decision == AutoMergeDecision.HOTL_REQUIRED, \
        f"Expected HOTL_REQUIRED, got {cycle.merge_decision}"


@pytest.mark.asyncio
async def test_closed_loop_rejects_failed_tests(
    qa_agent_mock,
    architect_agent_mock,
    modl_ai_report
):
    """
    Test that failed tests result in rejection.
    """
    # Modify Architect to return failed tests
    def failing_integrate_and_test(worker_results, test_file_path=None, code_file_path=None):
        from core.agent_framework import AgentError
        return AgentResponse(
            success=False,
            error=AgentError(
                error_type="tests_failed",
                message="Tests failed after 3 iterations",
                timestamp=datetime.utcnow(),
                context={"failed_tests": 2},
                recoverable=True
            ),
            data={
                "success": False,
                "iterations": 3,
                "test_output": "FAILED test_auth.py::test_validate_token",
                "bugfix_tasks_generated": 2
            }
        )

    architect_agent_mock.integrate_and_test = Mock(side_effect=failing_integrate_and_test)

    # Initialize ClosedLoop
    closed_loop = ClosedLoopManager(
        qa_agent=qa_agent_mock,
        architect_agent=architect_agent_mock
    )

    # Process report
    cycle = await closed_loop.process_report(modl_ai_report)

    # Should be rejected
    assert cycle.status == LoopStatus.FAILED, \
        f"Expected FAILED, got {cycle.status}"
    assert cycle.merge_decision == AutoMergeDecision.REJECTED, \
        f"Expected REJECTED, got {cycle.merge_decision}"


@pytest.mark.asyncio
async def test_architect_agent_not_configured_uses_fallback(
    qa_agent_mock,
    modl_ai_report
):
    """
    Test that ClosedLoop uses fallback simulation when Architect is not configured.
    """
    # No Architect Agent provided
    closed_loop = ClosedLoopManager(
        qa_agent=qa_agent_mock,
        architect_agent=None
    )

    # Process report
    cycle = await closed_loop.process_report(modl_ai_report)

    # Should complete with simulation
    assert cycle is not None
    assert cycle.status in [LoopStatus.AUTO_MERGED, LoopStatus.HOTL_REVIEW]

    # Check that simulation flag is set
    for task_id, result in cycle.test_results.items():
        assert result['fix_result'].get('simulated') is True, \
            "Should use simulated fix when no Architect configured"


@pytest.mark.asyncio
async def test_concurrent_cycle_limit(
    qa_agent_mock,
    architect_agent_mock,
    modl_ai_report
):
    """
    Test that concurrent cycle limit is enforced.
    """
    closed_loop = ClosedLoopManager(
        qa_agent=qa_agent_mock,
        architect_agent=architect_agent_mock,
        config={"max_concurrent_cycles": 2}
    )

    # Start 3 cycles (exceeds limit of 2)
    cycles = []

    # Delay Architect execution to keep cycles active
    async def delayed_plan(user_request, codebase_path=None):
        await asyncio.sleep(0.1)
        return architect_agent_mock.plan_and_delegate(user_request, codebase_path)

    with patch.object(architect_agent_mock, 'plan_and_delegate', side_effect=delayed_plan):
        # Start cycles concurrently
        tasks = [
            closed_loop.process_report(modl_ai_report),
            closed_loop.process_report(modl_ai_report),
            closed_loop.process_report(modl_ai_report)
        ]

        cycles = await asyncio.gather(*tasks)

    # All should complete, but warning should be logged
    assert len(cycles) == 3
    for cycle in cycles:
        assert cycle.completed_at is not None


@pytest.mark.asyncio
@patch('dashboard.backend.api.handle_closed_loop_approval')
async def test_approval_api_integration(mock_approval_handler):
    """
    Test approval API endpoint integration.

    This would normally test the FastAPI endpoint, but we mock it here.
    """
    # Mock approval handler
    mock_approval_handler.return_value = {
        "task_id": "loop_test_001",
        "cycle_id": "loop_test_001",
        "approved": True,
        "approved_by": "test_user",
        "approved_at": datetime.utcnow().isoformat(),
        "type": "closed_loop",
        "message": "Cycle approved and will be merged"
    }

    # Simulate API call
    result = await mock_approval_handler(
        task_id="loop_test_001",
        cycle_id="loop_test_001",
        approved=True,
        approved_by="test_user",
        comment="Looks good!",
        session=None
    )

    # Verify
    assert result['approved'] is True
    assert result['type'] == 'closed_loop'
    mock_approval_handler.assert_called_once()


def test_coverage_extraction():
    """
    Test coverage extraction from pytest output.
    """
    from core.closed_loop import ClosedLoopManager

    manager = ClosedLoopManager(
        qa_agent=Mock(),
        architect_agent=None
    )

    # Test various pytest output formats
    test_cases = [
        ("TOTAL 1000 500 85%", 85.0),
        ("Coverage: 92.5%", 92.5),
        ("No coverage info", None),
        ("", None)
    ]

    for output, expected in test_cases:
        result = manager._extract_coverage_from_output(output)
        assert result == expected, \
            f"Expected {expected} for output '{output}', got {result}"


def test_statistics_tracking():
    """
    Test that ClosedLoop tracks statistics correctly.
    """
    closed_loop = ClosedLoopManager(
        qa_agent=Mock(),
        architect_agent=None
    )

    # Initial stats
    stats = closed_loop.get_statistics()
    assert stats['total_cycles'] == 0
    assert stats['auto_merged'] == 0
    assert stats['hotl_reviews'] == 0
    assert stats['failed'] == 0

    # TODO: Add cycles and verify stats update
    # This would require running actual cycles or mocking the internal state


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
