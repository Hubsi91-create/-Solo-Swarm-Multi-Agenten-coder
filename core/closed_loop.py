"""
Closed Loop Manager - Autonomous Fix-Test-Merge Orchestration

This module implements the closed-loop QA system that enables true autonomy:
1. Receive test results/crash reports from Modl.ai
2. QA Agent analyzes and creates fix tasks
3. Architect Agent plans and implements fixes
4. Automated testing validates fixes
5. Auto-merge for high-confidence fixes
6. HOTL review for uncertain cases

This is the final piece for self-healing capability.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from core.tdf_schema import TaskDefinition, TaskType
from integrations.modl_ai import ModlAiPayload, Severity
from agents.workers.qa_agent import QAAgent


logger = logging.getLogger(__name__)


class LoopStatus(str, Enum):
    """Status of a closed loop cycle"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    FIXING = "fixing"
    TESTING = "testing"
    VALIDATING = "validating"
    AUTO_MERGED = "auto_merged"
    HOTL_REVIEW = "hotl_review"
    COMPLETED = "completed"
    FAILED = "failed"


class AutoMergeDecision(str, Enum):
    """Auto-merge decision types"""
    AUTO_APPROVE = "auto_approve"      # High confidence - merge automatically
    HOTL_REQUIRED = "hotl_required"    # Low confidence - requires review
    REJECTED = "rejected"               # Failed validation - do not merge


class ClosedLoopCycle:
    """Represents a single closed-loop fix cycle."""

    def __init__(
        self,
        cycle_id: str,
        report: ModlAiPayload,
        tasks: List[TaskDefinition]
    ):
        """
        Initialize a closed loop cycle.

        Args:
            cycle_id: Unique identifier for this cycle
            report: Original Modl.ai report
            tasks: Generated fix tasks
        """
        self.cycle_id = cycle_id
        self.report = report
        self.tasks = tasks
        self.status = LoopStatus.PENDING
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.test_results: Dict[str, Any] = {}
        self.merge_decision: Optional[AutoMergeDecision] = None
        self.confidence_score: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_id": self.cycle_id,
            "status": self.status.value,
            "report_type": self.report.report_type.value,
            "severity": self.report.severity.value,
            "task_count": len(self.tasks),
            "confidence_score": self.confidence_score,
            "merge_decision": self.merge_decision.value if self.merge_decision else None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.created_at).total_seconds()
                if self.completed_at else None
            ),
            "metadata": self.metadata
        }


class ClosedLoopManager:
    """
    Manages the autonomous fix-test-merge cycle.

    This is the orchestrator that coordinates:
    - QA Agent (report analysis)
    - Architect Agent (fix implementation)
    - Test execution
    - Auto-merge decisions
    - HOTL approval flow
    """

    def __init__(
        self,
        qa_agent: QAAgent,
        architect_agent: Optional[Any] = None,
        dashboard_manager: Optional[Any] = None,
        db_session: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Closed Loop Manager.

        Args:
            qa_agent: QA Agent for report analysis
            architect_agent: Architect Agent for fix implementation
            dashboard_manager: Dashboard for status updates
            db_session: Database session for tracking
            config: Configuration dictionary
        """
        self.qa_agent = qa_agent
        self.architect_agent = architect_agent
        self.dashboard_manager = dashboard_manager
        self.db_session = db_session
        self.config = config or {}

        # Configuration
        self.auto_merge_threshold = self.config.get("auto_merge_threshold", 90.0)
        self.require_test_coverage = self.config.get("require_test_coverage", True)
        self.min_test_coverage = self.config.get("min_test_coverage", 80.0)
        self.max_concurrent_cycles = self.config.get("max_concurrent_cycles", 3)

        # Active cycles tracking
        self.active_cycles: Dict[str, ClosedLoopCycle] = {}
        self.cycle_history: List[ClosedLoopCycle] = []

        # Statistics
        self.stats = {
            "total_cycles": 0,
            "auto_merged": 0,
            "hotl_reviews": 0,
            "failed": 0,
            "average_confidence": 0.0
        }

        logger.info(
            f"ClosedLoopManager initialized - "
            f"Auto-merge threshold: {self.auto_merge_threshold}%, "
            f"Test coverage: {self.min_test_coverage}%"
        )

    async def process_report(self, report: ModlAiPayload) -> ClosedLoopCycle:
        """
        Process a Modl.ai report through the closed loop system.

        This is the main entry point for the closed loop cycle.

        Args:
            report: Modl.ai report to process

        Returns:
            ClosedLoopCycle tracking the fix process
        """
        # Generate cycle ID (use UUID to ensure uniqueness for concurrent calls)
        cycle_id = f"loop_{report.report_type.value}_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"🔄 Starting closed loop cycle: {cycle_id} "
            f"(type: {report.report_type.value}, severity: {report.severity.value})"
        )

        # Check concurrent cycle limit
        if len(self.active_cycles) >= self.max_concurrent_cycles:
            logger.warning(
                f"Max concurrent cycles ({self.max_concurrent_cycles}) reached. "
                "Queueing cycle..."
            )
            # TODO: Implement queueing mechanism
            # For now, we'll proceed but log the warning

        # Phase 1: QA Agent analyzes report and creates tasks
        cycle = await self._analyze_report(cycle_id, report)

        if not cycle.tasks:
            logger.error(f"No tasks generated for cycle {cycle_id}")
            cycle.status = LoopStatus.FAILED
            cycle.completed_at = datetime.utcnow()
            self.cycle_history.append(cycle)
            return cycle

        # Add to active cycles
        self.active_cycles[cycle_id] = cycle
        self.stats["total_cycles"] += 1

        # Phase 2: Process QA tasks (fix, test, validate)
        await self.process_qa_tasks(cycle.tasks, cycle)

        # Phase 3: Make auto-merge decision
        await self._make_merge_decision(cycle)

        # Complete cycle
        cycle.completed_at = datetime.utcnow()
        del self.active_cycles[cycle_id]
        self.cycle_history.append(cycle)

        duration = (cycle.completed_at - cycle.created_at).total_seconds()
        logger.info(
            f"✅ Closed loop cycle completed: {cycle_id} "
            f"(duration: {duration:.1f}s, decision: {cycle.merge_decision.value})"
        )

        return cycle

    async def _analyze_report(
        self,
        cycle_id: str,
        report: ModlAiPayload
    ) -> ClosedLoopCycle:
        """
        Phase 1: Analyze report using QA Agent.

        Args:
            cycle_id: Cycle identifier
            report: Modl.ai report

        Returns:
            ClosedLoopCycle with generated tasks
        """
        logger.info(f"📊 Analyzing report for cycle {cycle_id}...")

        # Use QA Agent to analyze report
        tasks = self.qa_agent.analyze_report(report)

        # Calculate average confidence
        if tasks:
            avg_confidence = sum(
                t.context.get('confidence_score', 0) for t in tasks
            ) / len(tasks)
        else:
            avg_confidence = 0.0

        # Create cycle
        cycle = ClosedLoopCycle(
            cycle_id=cycle_id,
            report=report,
            tasks=tasks
        )
        cycle.status = LoopStatus.ANALYZING
        cycle.confidence_score = avg_confidence
        cycle.metadata["analysis_timestamp"] = datetime.utcnow().isoformat()

        logger.info(
            f"  Generated {len(tasks)} tasks with avg confidence {avg_confidence:.1f}%"
        )

        return cycle

    async def process_qa_tasks(
        self,
        tasks: List[TaskDefinition],
        cycle: ClosedLoopCycle
    ) -> None:
        """
        Phase 2: Process QA tasks through fix-test-validate pipeline.

        Args:
            tasks: List of tasks to process
            cycle: Closed loop cycle tracking
        """
        logger.info(f"🔧 Processing {len(tasks)} QA tasks for cycle {cycle.cycle_id}...")

        cycle.status = LoopStatus.FIXING

        for task in tasks:
            try:
                # Step 1: Send to Architect Agent for implementation
                if self.architect_agent:
                    logger.info(f"  Sending task {task.task_id} to Architect...")
                    # TODO: Integrate with actual Architect Agent
                    # For now, simulate success
                    fix_result = await self._simulate_fix(task)
                else:
                    logger.warning("No Architect Agent configured, simulating fix")
                    fix_result = await self._simulate_fix(task)

                # Step 2: Run tests
                cycle.status = LoopStatus.TESTING
                test_result = await self._run_tests(task, fix_result)

                # Step 3: Validate results
                cycle.status = LoopStatus.VALIDATING
                validation = await self._validate_fix(task, test_result)

                # Store results
                cycle.test_results[task.task_id] = {
                    "fix_result": fix_result,
                    "test_result": test_result,
                    "validation": validation,
                    "timestamp": datetime.utcnow().isoformat()
                }

                logger.info(
                    f"  Task {task.task_id} processed: "
                    f"tests {'✅ passed' if validation['tests_passed'] else '❌ failed'}, "
                    f"coverage {validation['coverage']}%"
                )

            except Exception as e:
                logger.error(f"Error processing task {task.task_id}: {e}", exc_info=True)
                cycle.test_results[task.task_id] = {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }

    async def _make_merge_decision(self, cycle: ClosedLoopCycle) -> None:
        """
        Phase 3: Make auto-merge decision based on test results and confidence.

        Args:
            cycle: Closed loop cycle with test results
        """
        logger.info(f"🤔 Making merge decision for cycle {cycle.cycle_id}...")

        # Collect validation results
        all_tests_passed = True
        total_coverage = 0.0
        validated_count = 0

        for task_id, result in cycle.test_results.items():
            if 'error' in result:
                all_tests_passed = False
                continue

            validation = result.get('validation', {})
            if not validation.get('tests_passed', False):
                all_tests_passed = False

            total_coverage += validation.get('coverage', 0)
            validated_count += 1

        # Calculate average coverage
        avg_coverage = total_coverage / validated_count if validated_count > 0 else 0

        # Decision criteria
        confidence_ok = cycle.confidence_score >= self.auto_merge_threshold
        tests_ok = all_tests_passed
        coverage_ok = avg_coverage >= self.min_test_coverage if self.require_test_coverage else True

        logger.info(
            f"  Confidence: {cycle.confidence_score:.1f}% (threshold: {self.auto_merge_threshold}%)\n"
            f"  Tests: {'✅' if tests_ok else '❌'}\n"
            f"  Coverage: {avg_coverage:.1f}% (min: {self.min_test_coverage}%)"
        )

        # Make decision
        if confidence_ok and tests_ok and coverage_ok:
            cycle.merge_decision = AutoMergeDecision.AUTO_APPROVE
            cycle.status = LoopStatus.AUTO_MERGED
            self.stats["auto_merged"] += 1

            logger.info(f"✅ AUTO-APPROVE: High confidence fix with good test coverage")

            # Automatically approve in database
            if self.db_session:
                await self._auto_approve_tasks(cycle.tasks)

        elif not tests_ok:
            cycle.merge_decision = AutoMergeDecision.REJECTED
            cycle.status = LoopStatus.FAILED
            self.stats["failed"] += 1

            logger.error(f"❌ REJECTED: Tests failed - cannot merge")

        else:
            cycle.merge_decision = AutoMergeDecision.HOTL_REQUIRED
            cycle.status = LoopStatus.HOTL_REVIEW
            self.stats["hotl_reviews"] += 1

            logger.warning(
                f"⚠️  HOTL REVIEW REQUIRED: "
                f"Confidence or coverage below threshold"
            )

            # Send to dashboard for HOTL review
            if self.dashboard_manager:
                await self._request_hotl_review(cycle)

        # Update statistics
        if self.stats["total_cycles"] > 0:
            self.stats["average_confidence"] = (
                cycle.confidence_score +
                (self.stats.get("average_confidence", 0) * (self.stats["total_cycles"] - 1))
            ) / self.stats["total_cycles"]

    async def _simulate_fix(self, task: TaskDefinition) -> Dict[str, Any]:
        """Simulate fix implementation (placeholder for Architect Agent integration)."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "status": "completed",
            "changes": ["file1.py", "file2.py"],
            "commit_sha": f"abc123_{task.task_id}"
        }

    async def _run_tests(
        self,
        task: TaskDefinition,
        fix_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run automated tests on the fix."""
        await asyncio.sleep(0.1)  # Simulate test execution

        # Simulate test results based on confidence score
        confidence = task.context.get('confidence_score', 50)
        tests_pass = confidence > 60  # Higher confidence = more likely to pass

        return {
            "passed": tests_pass,
            "total_tests": 15,
            "failed_tests": 0 if tests_pass else 2,
            "coverage_percent": min(100, confidence + 10),  # Simulated coverage
            "duration_seconds": 2.5
        }

    async def _validate_fix(
        self,
        task: TaskDefinition,
        test_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate fix quality and test coverage."""
        return {
            "tests_passed": test_result.get('passed', False),
            "coverage": test_result.get('coverage_percent', 0),
            "meets_requirements": test_result.get('passed', False),
            "ready_for_merge": (
                test_result.get('passed', False) and
                test_result.get('coverage_percent', 0) >= self.min_test_coverage
            )
        }

    async def _auto_approve_tasks(self, tasks: List[TaskDefinition]) -> None:
        """Automatically approve tasks in the database."""
        logger.info(f"✅ Auto-approving {len(tasks)} tasks in database...")

        # TODO: Update database to mark tasks as approved
        # For now, just update context
        for task in tasks:
            task.context['auto_approved'] = True
            task.context['approved_at'] = datetime.utcnow().isoformat()
            task.context['approved_by'] = 'ClosedLoopManager'

    async def _request_hotl_review(self, cycle: ClosedLoopCycle) -> None:
        """Request HOTL review via dashboard."""
        logger.info(f"📋 Requesting HOTL review for cycle {cycle.cycle_id}...")

        # TODO: Send approval request to dashboard
        # This would integrate with the existing approval_request system

    def get_statistics(self) -> Dict[str, Any]:
        """Get closed loop statistics."""
        return {
            "total_cycles": self.stats["total_cycles"],
            "active_cycles": len(self.active_cycles),
            "auto_merged": self.stats["auto_merged"],
            "hotl_reviews": self.stats["hotl_reviews"],
            "failed": self.stats["failed"],
            "average_confidence": round(self.stats["average_confidence"], 2),
            "auto_merge_rate": (
                round(self.stats["auto_merged"] / self.stats["total_cycles"] * 100, 1)
                if self.stats["total_cycles"] > 0 else 0
            )
        }

    def get_cycle_status(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific cycle."""
        # Check active cycles
        if cycle_id in self.active_cycles:
            return self.active_cycles[cycle_id].to_dict()

        # Check history
        for cycle in self.cycle_history:
            if cycle.cycle_id == cycle_id:
                return cycle.to_dict()

        return None

    def get_active_cycles(self) -> List[Dict[str, Any]]:
        """Get all active cycles."""
        return [cycle.to_dict() for cycle in self.active_cycles.values()]
