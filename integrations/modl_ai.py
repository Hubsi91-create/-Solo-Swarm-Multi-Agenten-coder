"""
Modl.ai Integration - Webhook Endpoint for Test Results and Crash Reports

This module provides integration with Modl.ai (or similar QA/testing platforms)
to receive automated test results, crash reports, and performance metrics.

Features:
- FastAPI webhook endpoint for receiving test results
- Payload validation and parsing
- Support for multiple report types (crashes, balance, performance)
- Automatic QA task generation from reports
- Integration with Closed Loop Manager for autonomous fixes

Webhook Endpoint: POST /webhooks/modl-ai-result
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field, ConfigDict


logger = logging.getLogger(__name__)


# ==================== Report Types ====================

class ReportType(str, Enum):
    """Types of reports from Modl.ai"""
    CRASH = "crash"
    BALANCE = "balance"
    PERFORMANCE = "performance"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"


class Severity(str, Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ==================== Pydantic Models ====================

class CrashReport(BaseModel):
    """Crash report from Modl.ai"""
    exception_type: str = Field(..., description="Exception type (e.g., NullReferenceException)")
    message: str = Field(..., description="Error message")
    stack_trace: str = Field(..., description="Full stack trace")
    frequency: int = Field(default=1, description="Number of occurrences")
    affected_users: int = Field(default=0, description="Number of affected users")
    first_seen: str = Field(..., description="Timestamp of first occurrence")
    last_seen: str = Field(..., description="Timestamp of last occurrence")
    environment: str = Field(default="production", description="Environment (dev/staging/production)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BalanceIssue(BaseModel):
    """Game balance issue report"""
    metric_name: str = Field(..., description="Metric that's out of balance")
    expected_value: float = Field(..., description="Expected value")
    actual_value: float = Field(..., description="Actual value")
    deviation_percent: float = Field(..., description="Percentage deviation")
    sample_size: int = Field(default=1000, description="Number of samples")
    context: str = Field(..., description="Context/description of the issue")
    suggested_fix: Optional[str] = Field(None, description="Suggested fix if available")


class PerformanceIssue(BaseModel):
    """Performance degradation report"""
    metric_type: str = Field(..., description="Performance metric (fps, memory, load_time)")
    threshold: float = Field(..., description="Performance threshold")
    measured_value: float = Field(..., description="Measured value")
    location: str = Field(..., description="Code location or scene")
    duration_seconds: float = Field(..., description="Duration of the issue")
    severity: Severity = Field(..., description="Issue severity")


class TestFailure(BaseModel):
    """Test failure report"""
    test_name: str = Field(..., description="Name of failed test")
    test_type: str = Field(..., description="Type of test (unit/integration)")
    failure_message: str = Field(..., description="Failure message")
    expected: Optional[str] = Field(None, description="Expected result")
    actual: Optional[str] = Field(None, description="Actual result")
    file_path: str = Field(..., description="Test file path")
    line_number: Optional[int] = Field(None, description="Line number")


class ModlAiPayload(BaseModel):
    """
    Main payload structure from Modl.ai webhook.

    This is a flexible structure that can contain different report types.
    """
    report_type: ReportType = Field(..., description="Type of report")
    severity: Severity = Field(..., description="Issue severity")
    timestamp: str = Field(..., description="Report timestamp")
    project_id: str = Field(..., description="Project identifier")
    version: str = Field(default="1.0.0", description="Game/app version")

    # Report-specific data (only one should be populated)
    crash_report: Optional[CrashReport] = None
    balance_issue: Optional[BalanceIssue] = None
    performance_issue: Optional[PerformanceIssue] = None
    test_failure: Optional[TestFailure] = None

    # Generic metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "report_type": "crash",
                "severity": "high",
                "timestamp": "2025-01-19T10:30:00Z",
                "project_id": "solo-swarm-game",
                "version": "0.1.0",
                "crash_report": {
                    "exception_type": "NullReferenceException",
                    "message": "Object reference not set to an instance of an object",
                    "stack_trace": "at GameManager.SpawnEnemy() line 42",
                    "frequency": 15,
                    "affected_users": 3,
                    "first_seen": "2025-01-19T09:00:00Z",
                    "last_seen": "2025-01-19T10:30:00Z",
                    "environment": "production"
                }
            }
        }
    )


# ==================== FastAPI Router ====================

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/modl-ai-result")
async def receive_modl_ai_result(
    payload: ModlAiPayload,
    background_tasks: BackgroundTasks,
    request: Request
) -> Dict[str, Any]:
    """
    Webhook endpoint for receiving Modl.ai test results and reports.

    This endpoint:
    1. Validates the incoming payload
    2. Logs the report
    3. Queues the report for QA Agent analysis (background task)
    4. Returns acknowledgment

    Args:
        payload: Modl.ai payload with report data
        background_tasks: FastAPI background tasks
        request: FastAPI request object

    Returns:
        Acknowledgment with report ID
    """
    logger.info(
        f"📥 Received Modl.ai report: {payload.report_type.value} "
        f"(severity: {payload.severity.value})"
    )

    # Generate report ID for tracking
    report_id = f"modl_{payload.report_type.value}_{int(datetime.utcnow().timestamp())}"

    # Log detailed report information
    _log_report_details(payload, report_id)

    # Queue for QA Agent processing (background task)
    background_tasks.add_task(
        _process_report_async,
        payload=payload,
        report_id=report_id
    )

    return {
        "status": "received",
        "report_id": report_id,
        "report_type": payload.report_type.value,
        "severity": payload.severity.value,
        "queued_for_processing": True,
        "received_at": datetime.utcnow().isoformat()
    }


@router.post("/modl-ai-result/simulate/{report_type}")
async def simulate_modl_ai_report(
    report_type: ReportType,
    background_tasks: BackgroundTasks,
    severity: Severity = Severity.MEDIUM
) -> Dict[str, Any]:
    """
    Simulate a Modl.ai report for testing purposes.

    This endpoint generates realistic test data for different report types.

    Args:
        report_type: Type of report to simulate
        background_tasks: FastAPI background tasks
        severity: Report severity

    Returns:
        Simulated report data
    """
    # Generate simulated payload
    payload = _generate_simulated_report(report_type, severity)

    # Process it like a real report
    return await receive_modl_ai_result(payload, background_tasks, None)


# ==================== Helper Functions ====================

def _log_report_details(payload: ModlAiPayload, report_id: str) -> None:
    """Log detailed report information."""
    log_msg = f"\n{'='*70}\n"
    log_msg += f"🔍 Modl.ai Report Details (ID: {report_id})\n"
    log_msg += f"{'='*70}\n"
    log_msg += f"Type: {payload.report_type.value}\n"
    log_msg += f"Severity: {payload.severity.value}\n"
    log_msg += f"Project: {payload.project_id}\n"
    log_msg += f"Version: {payload.version}\n"
    log_msg += f"Timestamp: {payload.timestamp}\n"

    if payload.crash_report:
        cr = payload.crash_report
        log_msg += f"\n📛 CRASH REPORT:\n"
        log_msg += f"  Exception: {cr.exception_type}\n"
        log_msg += f"  Message: {cr.message}\n"
        log_msg += f"  Frequency: {cr.frequency} occurrences\n"
        log_msg += f"  Affected Users: {cr.affected_users}\n"
        log_msg += f"  Stack Trace:\n{cr.stack_trace}\n"

    elif payload.balance_issue:
        bi = payload.balance_issue
        log_msg += f"\n⚖️  BALANCE ISSUE:\n"
        log_msg += f"  Metric: {bi.metric_name}\n"
        log_msg += f"  Expected: {bi.expected_value}\n"
        log_msg += f"  Actual: {bi.actual_value}\n"
        log_msg += f"  Deviation: {bi.deviation_percent}%\n"
        log_msg += f"  Context: {bi.context}\n"

    elif payload.performance_issue:
        pi = payload.performance_issue
        log_msg += f"\n⚡ PERFORMANCE ISSUE:\n"
        log_msg += f"  Metric: {pi.metric_type}\n"
        log_msg += f"  Threshold: {pi.threshold}\n"
        log_msg += f"  Measured: {pi.measured_value}\n"
        log_msg += f"  Location: {pi.location}\n"
        log_msg += f"  Duration: {pi.duration_seconds}s\n"

    elif payload.test_failure:
        tf = payload.test_failure
        log_msg += f"\n❌ TEST FAILURE:\n"
        log_msg += f"  Test: {tf.test_name}\n"
        log_msg += f"  Type: {tf.test_type}\n"
        log_msg += f"  Message: {tf.failure_message}\n"
        log_msg += f"  File: {tf.file_path}:{tf.line_number or 'N/A'}\n"

    log_msg += f"{'='*70}\n"

    # Log at appropriate level based on severity
    if payload.severity in [Severity.CRITICAL, Severity.HIGH]:
        logger.error(log_msg)
    elif payload.severity == Severity.MEDIUM:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)


async def _process_report_async(payload: ModlAiPayload, report_id: str) -> None:
    """
    Background task to process the report.

    This will be called by the QA Agent in the next step.
    For now, it just logs that processing would happen.
    """
    logger.info(f"🔄 Processing report {report_id} in background...")

    # TODO: In the next step, this will:
    # 1. Pass report to QA Agent for analysis
    # 2. QA Agent creates TaskDefinitions
    # 3. Tasks are sent to Closed Loop Manager
    # 4. Closed Loop Manager orchestrates fix-test-merge cycle

    logger.info(f"✅ Report {report_id} queued for QA Agent analysis")


def _generate_simulated_report(report_type: ReportType, severity: Severity) -> ModlAiPayload:
    """Generate realistic simulated report data for testing."""
    timestamp = datetime.utcnow().isoformat()

    if report_type == ReportType.CRASH:
        return ModlAiPayload(
            report_type=report_type,
            severity=severity,
            timestamp=timestamp,
            project_id="solo-swarm-game",
            version="0.1.0",
            crash_report=CrashReport(
                exception_type="NullReferenceException",
                message="Object reference not set to an instance of an object at SpawnEnemy",
                stack_trace=(
                    "at GameManager.SpawnEnemy() in GameManager.cs:line 42\n"
                    "at WaveController.StartWave() in WaveController.cs:line 128\n"
                    "at GameLoop.Update() in GameLoop.cs:line 89"
                ),
                frequency=15,
                affected_users=3,
                first_seen=(datetime.utcnow()).isoformat(),
                last_seen=timestamp,
                environment="production",
                metadata={"platform": "Unity", "scene": "GameplayScene"}
            )
        )

    elif report_type == ReportType.BALANCE:
        return ModlAiPayload(
            report_type=report_type,
            severity=severity,
            timestamp=timestamp,
            project_id="solo-swarm-game",
            version="0.1.0",
            balance_issue=BalanceIssue(
                metric_name="enemy_difficulty",
                expected_value=50.0,
                actual_value=85.0,
                deviation_percent=70.0,
                sample_size=1000,
                context="Enemy difficulty scaling too aggressive in wave 3-5",
                suggested_fix="Reduce difficulty multiplier from 1.5 to 1.2"
            )
        )

    elif report_type == ReportType.PERFORMANCE:
        return ModlAiPayload(
            report_type=report_type,
            severity=severity,
            timestamp=timestamp,
            project_id="solo-swarm-game",
            version="0.1.0",
            performance_issue=PerformanceIssue(
                metric_type="fps",
                threshold=60.0,
                measured_value=28.5,
                location="ParticleManager.Update()",
                duration_seconds=45.0,
                severity=severity
            )
        )

    elif report_type == ReportType.UNIT_TEST:
        return ModlAiPayload(
            report_type=report_type,
            severity=severity,
            timestamp=timestamp,
            project_id="solo-swarm-game",
            version="0.1.0",
            test_failure=TestFailure(
                test_name="test_enemy_spawn_rate",
                test_type="unit",
                failure_message="AssertionError: Expected 10 enemies, got 8",
                expected="10 enemies spawned",
                actual="8 enemies spawned",
                file_path="tests/test_game_manager.py",
                line_number=156
            )
        )

    else:  # INTEGRATION_TEST
        return ModlAiPayload(
            report_type=report_type,
            severity=severity,
            timestamp=timestamp,
            project_id="solo-swarm-game",
            version="0.1.0",
            test_failure=TestFailure(
                test_name="test_full_gameplay_loop",
                test_type="integration",
                failure_message="Timeout: Game loop did not complete within 60s",
                expected="Game loop completes in <60s",
                actual="Game loop timed out after 60s",
                file_path="tests/integration/test_gameplay.py",
                line_number=89
            )
        )


# ==================== Health Check ====================

@router.get("/modl-ai-health")
async def modl_ai_health_check() -> Dict[str, Any]:
    """Health check endpoint for Modl.ai webhook integration."""
    return {
        "status": "healthy",
        "service": "modl-ai-webhook",
        "timestamp": datetime.utcnow().isoformat(),
        "supported_report_types": [rt.value for rt in ReportType],
        "supported_severities": [s.value for s in Severity]
    }
