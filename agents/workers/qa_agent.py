"""
QA Agent - Specialized agent for parsing test reports and creating fix tasks

This agent analyzes crash reports, test failures, and performance issues from
Modl.ai and converts them into actionable TaskDefinitions for the Closed Loop system.

Key Responsibilities:
- Parse unstructured log data and crash reports
- Extract actionable information (stack traces, error messages)
- Create TaskDefinitions for fixes
- Assign confidence scores for auto-merge decisions
- Handle multiple report types (crashes, balance, performance, tests)
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import re
from datetime import datetime

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType
from integrations.modl_ai import (
    ModlAiPayload,
    ReportType,
    Severity,
    CrashReport,
    BalanceIssue,
    PerformanceIssue,
    TestFailure
)


logger = logging.getLogger(__name__)


class ConfidenceLevel(str):
    """Confidence levels for auto-merge decisions"""
    HIGH = "high"          # 90-100% confidence - auto-merge safe
    MEDIUM = "medium"      # 60-89% confidence - requires review
    LOW = "low"            # <60% confidence - definitely requires review


class QAAgent(BaseAgent):
    """
    QA Agent for analyzing test reports and creating fix tasks.

    This agent uses Claude Haiku for cost-effective parsing and analysis
    of test reports, crash logs, and performance metrics.

    Workflow:
    1. gather_context: Parse report and extract relevant information
    2. take_action: Create TaskDefinitions for fixes
    3. verify_work: Validate task quality and assign confidence scores
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "QAAgent",
        config: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[TokenTracker] = None
    ):
        """
        Initialize the QA Agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_name: Human-readable name for the agent
            config: Optional configuration dictionary
            token_tracker: Optional TokenTracker instance for cost tracking
        """
        super().__init__(agent_id, agent_name, config)

        # Initialize token tracker
        self.token_tracker = token_tracker or TokenTracker()

        # QA-specific configuration
        self.auto_merge_confidence_threshold = self.config.get(
            "auto_merge_threshold", 90.0
        )
        self.use_haiku = self.config.get("use_haiku", True)  # Cost-effective
        self.max_tasks_per_report = self.config.get("max_tasks_per_report", 5)

        logger.info(
            f"QAAgent initialized: {agent_name} (ID: {agent_id}), "
            f"auto-merge threshold: {self.auto_merge_confidence_threshold}%"
        )

    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather context from the report.

        Extracts structured information from potentially unstructured logs.

        Args:
            task: TaskDefinition containing the report data

        Returns:
            AgentResponse with parsed report context
        """
        logger.info(f"QAAgent gathering context for task: {task.task_id}")

        try:
            # The report should be in task.context['report']
            if 'report' not in task.context:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="missing_report",
                        message="No report found in task context",
                        timestamp=datetime.utcnow(),
                        context={"task_id": task.task_id},
                        recoverable=False
                    )
                )

            report_data = task.context['report']

            # Parse based on report type
            context = self._extract_report_context(report_data)

            return AgentResponse(
                success=True,
                data=context,
                metadata={"agent_id": self.agent_id,
                "task_id": task.task_id,
                "timestamp": datetime.utcnow().isoformat()}
            )

        except Exception as e:
            logger.error(f"Error gathering context: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=AgentError(
                    error_type="context_error",
                    message=str(e),
                    timestamp=datetime.utcnow(),
                    context={"task_id": task.task_id},
                    recoverable=True
                )
            )

    def take_action(self, task: TaskDefinition, context: Dict[str, Any]) -> AgentResponse:
        """
        Phase 2: Create TaskDefinitions for fixes.

        Converts parsed report data into actionable tasks.

        Args:
            task: Original task definition
            context: Context gathered in phase 1

        Returns:
            AgentResponse with list of TaskDefinitions
        """
        logger.info(f"QAAgent creating fix tasks for: {task.task_id}")

        try:
            report_type = context.get('report_type')
            severity = context.get('severity')

            # Generate tasks based on report type
            if report_type == ReportType.CRASH:
                tasks = self._create_crash_fix_tasks(context)
            elif report_type == ReportType.BALANCE:
                tasks = self._create_balance_fix_tasks(context)
            elif report_type == ReportType.PERFORMANCE:
                tasks = self._create_performance_fix_tasks(context)
            elif report_type in [ReportType.UNIT_TEST, ReportType.INTEGRATION_TEST]:
                tasks = self._create_test_fix_tasks(context)
            else:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="unknown_report_type",
                        message=f"Unknown report type: {report_type}",
                        timestamp=datetime.utcnow(),
                        context={"report_type": report_type},
                        recoverable=False
                    )
                )

            # Limit number of tasks
            if len(tasks) > self.max_tasks_per_report:
                logger.warning(
                    f"Generated {len(tasks)} tasks, limiting to {self.max_tasks_per_report}"
                )
                tasks = tasks[:self.max_tasks_per_report]

            return AgentResponse(
                success=True,
                data={"tasks": tasks, "task_count": len(tasks)},
                metadata={"agent_id": self.agent_id,
                "task_id": task.task_id,
                "timestamp": datetime.utcnow().isoformat()}
            )

        except Exception as e:
            logger.error(f"Error creating fix tasks: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=AgentError(
                    error_type="task_creation_error",
                    message=str(e),
                    timestamp=datetime.utcnow(),
                    context={"task_id": task.task_id},
                    recoverable=True
                )
            )

    def verify_work(self, task: TaskDefinition, result: Any) -> AgentResponse:
        """
        Phase 3: Verify task quality and assign confidence scores.

        Determines if generated tasks are suitable for auto-merge.

        Args:
            task: Original task definition
            result: Result from take_action phase

        Returns:
            AgentResponse with validation results and confidence scores
        """
        logger.info(f"QAAgent verifying tasks for: {task.task_id}")

        try:
            tasks = result.get('tasks', [])

            # Validate each task and assign confidence
            validated_tasks = []
            for task_def in tasks:
                confidence = self._calculate_confidence(task_def)
                task_def.context['confidence_score'] = confidence
                task_def.context['auto_merge_eligible'] = (
                    confidence >= self.auto_merge_confidence_threshold
                )
                validated_tasks.append(task_def)

            # Calculate overall confidence
            if validated_tasks:
                avg_confidence = sum(
                    t.context.get('confidence_score', 0) for t in validated_tasks
                ) / len(validated_tasks)
            else:
                avg_confidence = 0.0

            return AgentResponse(
                success=True,
                data={
                    "tasks": validated_tasks,
                    "average_confidence": avg_confidence,
                    "auto_merge_count": sum(
                        1 for t in validated_tasks
                        if t.context.get('auto_merge_eligible', False)
                    )
                },
                metadata={"agent_id": self.agent_id,
                "task_id": task.task_id,
                "timestamp": datetime.utcnow().isoformat()}
            )

        except Exception as e:
            logger.error(f"Error verifying tasks: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=AgentError(
                    error_type="verification_error",
                    message=str(e),
                    timestamp=datetime.utcnow(),
                    context={"task_id": task.task_id},
                    recoverable=True
                )
            )

    # ==================== Helper Methods ====================

    def _extract_report_context(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured context from report data."""
        return {
            'report_type': report_data.get('report_type'),
            'severity': report_data.get('severity'),
            'timestamp': report_data.get('timestamp'),
            'project_id': report_data.get('project_id'),
            'version': report_data.get('version'),
            'raw_report': report_data
        }

    def _create_crash_fix_tasks(self, context: Dict[str, Any]) -> List[TaskDefinition]:
        """Create tasks for fixing crashes."""
        tasks = []
        crash_data = context['raw_report'].get('crash_report', {})

        # Extract relevant information
        exception_type = crash_data.get('exception_type', 'UnknownException')
        message = crash_data.get('message', '')
        stack_trace = crash_data.get('stack_trace', '')

        # Parse stack trace to find file and line
        file_location = self._parse_stack_trace(stack_trace)

        # Create fix task
        task = TaskDefinition(
            task_id=f"fix_crash_{int(datetime.utcnow().timestamp())}",
            task_type=TaskType.IMPLEMENTATION,
            priority=self._severity_to_priority(context.get('severity')),
            assigned_agent="ArchitectAgent",  # Will be assigned to Architect
            context={
                "description": f"Fix {exception_type}: {message}",
                "exception_type": exception_type,
                "error_message": message,
                "stack_trace": stack_trace,
                "file_location": file_location,
                "frequency": crash_data.get('frequency', 1),
                "affected_users": crash_data.get('affected_users', 0),
                "report_type": "crash",
                "severity": context.get('severity'),
                "source": "modl_ai"
            },
            requirements={
                "fix_type": "null_check" if "null" in exception_type.lower() else "general",
                "test_coverage_required": True,
                "verify_no_regression": True
            }
        )

        tasks.append(task)
        return tasks

    def _create_balance_fix_tasks(self, context: Dict[str, Any]) -> List[TaskDefinition]:
        """Create tasks for fixing game balance issues."""
        tasks = []
        balance_data = context['raw_report'].get('balance_issue', {})

        metric_name = balance_data.get('metric_name', 'unknown')
        expected = balance_data.get('expected_value')
        actual = balance_data.get('actual_value')
        suggested_fix = balance_data.get('suggested_fix')

        task = TaskDefinition(
            task_id=f"fix_balance_{int(datetime.utcnow().timestamp())}",
            task_type=TaskType.REFACTORING,
            priority=self._severity_to_priority(context.get('severity')),
            assigned_agent="ArchitectAgent",  # Will be assigned to Architect
            context={
                "description": f"Adjust {metric_name} balance (Expected: {expected}, Actual: {actual})",
                "metric_name": metric_name,
                "expected_value": expected,
                "actual_value": actual,
                "deviation": balance_data.get('deviation_percent'),
                "suggested_fix": suggested_fix,
                "context": balance_data.get('context', ''),
                "report_type": "balance",
                "severity": context.get('severity'),
                "source": "modl_ai"
            },
            requirements={
                "test_new_balance": True,
                "verify_player_feedback": True,
                "config_based_fix": True  # Prefer config changes over code
            }
        )

        tasks.append(task)
        return tasks

    def _create_performance_fix_tasks(self, context: Dict[str, Any]) -> List[TaskDefinition]:
        """Create tasks for fixing performance issues."""
        tasks = []
        perf_data = context['raw_report'].get('performance_issue', {})

        metric_type = perf_data.get('metric_type', 'unknown')
        threshold = perf_data.get('threshold')
        measured = perf_data.get('measured_value')
        location = perf_data.get('location', 'unknown')

        task = TaskDefinition(
            task_id=f"fix_performance_{int(datetime.utcnow().timestamp())}",
            task_type=TaskType.REFACTORING,
            priority=self._severity_to_priority(context.get('severity')),
            assigned_agent="ArchitectAgent",  # Will be assigned to Architect
            context={
                "description": f"Optimize {metric_type} in {location} (Target: {threshold}, Current: {measured})",
                "metric_type": metric_type,
                "threshold": threshold,
                "measured_value": measured,
                "location": location,
                "duration": perf_data.get('duration_seconds', 0),
                "report_type": "performance",
                "severity": context.get('severity'),
                "source": "modl_ai"
            },
            requirements={
                "performance_test_required": True,
                "profile_before_after": True,
                "verify_no_regression": True
            }
        )

        tasks.append(task)
        return tasks

    def _create_test_fix_tasks(self, context: Dict[str, Any]) -> List[TaskDefinition]:
        """Create tasks for fixing test failures."""
        tasks = []
        test_data = context['raw_report'].get('test_failure', {})

        test_name = test_data.get('test_name', 'unknown')
        failure_msg = test_data.get('failure_message', '')
        file_path = test_data.get('file_path', '')
        line_number = test_data.get('line_number')

        task = TaskDefinition(
            task_id=f"fix_test_{int(datetime.utcnow().timestamp())}",
            task_type=TaskType.IMPLEMENTATION,
            priority=self._severity_to_priority(context.get('severity')),
            assigned_agent="ArchitectAgent",  # Will be assigned to Architect
            context={
                "description": f"Fix failing test: {test_name}",
                "test_name": test_name,
                "test_type": test_data.get('test_type', 'unit'),
                "failure_message": failure_msg,
                "expected": test_data.get('expected'),
                "actual": test_data.get('actual'),
                "file_path": file_path,
                "line_number": line_number,
                "report_type": "test_failure",
                "severity": context.get('severity'),
                "source": "modl_ai"
            },
            requirements={
                "fix_test_or_code": True,  # Either test is wrong or code is wrong
                "verify_related_tests": True,
                "maintain_coverage": True
            }
        )

        tasks.append(task)
        return tasks

    def _calculate_confidence(self, task: TaskDefinition) -> float:
        """
        Calculate confidence score for auto-merge eligibility.

        Factors:
        - Severity (lower severity = higher confidence for auto-merge)
        - Task type (bug fixes < refactoring < config changes)
        - Frequency (higher frequency = more testing data)
        - Affected users (lower = higher confidence)
        - Stack trace quality (clear stack trace = higher confidence)

        Returns:
            Confidence score (0-100)
        """
        score = 50.0  # Base confidence

        # Severity factor
        severity = task.context.get('severity', Severity.MEDIUM)
        severity_scores = {
            Severity.CRITICAL: -20,
            Severity.HIGH: -10,
            Severity.MEDIUM: 0,
            Severity.LOW: +15,
            Severity.INFO: +25
        }
        score += severity_scores.get(severity, 0)

        # Task type factor
        task_type_scores = {
            TaskType.IMPLEMENTATION: +10,
            TaskType.REFACTORING: +5,
            TaskType.REFACTORING: 0,
            TaskType.IMPLEMENTATION: -10
        }
        score += task_type_scores.get(task.task_type, 0)

        # Frequency factor (for crashes)
        frequency = task.context.get('frequency', 1)
        if frequency == 1:
            score += 10  # Single occurrence = easier to fix
        elif frequency < 10:
            score += 5
        else:
            score -= 5  # High frequency might indicate complex issue

        # Affected users factor (for crashes)
        affected_users = task.context.get('affected_users', 0)
        if affected_users == 0:
            score += 10
        elif affected_users < 5:
            score += 5
        else:
            score -= 10  # Many affected users = critical

        # Stack trace quality (for crashes)
        if task.context.get('file_location'):
            score += 15  # Clear location = easier to fix

        # Config-based fix (for balance issues)
        if task.requirements.get('config_based_fix'):
            score += 20  # Config changes are safer

        # Suggested fix available
        if task.context.get('suggested_fix'):
            score += 10

        # Clamp to 0-100 range
        return max(0.0, min(100.0, score))

    def _parse_stack_trace(self, stack_trace: str) -> Optional[Dict[str, Any]]:
        """
        Parse stack trace to extract file location and line number.

        Supports multiple formats:
        - Python: 'at File.py:line 42'
        - C#: 'at GameManager.cs:line 42'
        - Generic: 'File.ext:123'
        """
        if not stack_trace:
            return None

        # Try to find file:line patterns
        patterns = [
            r'in\s+([\w/\\.]+\.[\w]+):line\s+(\d+)',  # C# style with "in"
            r'at\s+([\w/\\.]+\.[\w]+):line\s+(\d+)',  # C# style
            r'File\s+"([\w/\\.]+\.[\w]+)",\s+line\s+(\d+)',  # Python style
            r'([\w/\\.]+\.[\w]+):(\d+)',  # Generic style
        ]

        for pattern in patterns:
            match = re.search(pattern, stack_trace)
            if match:
                return {
                    "file": match.group(1),
                    "line": int(match.group(2))
                }

        return None

    def _severity_to_priority(self, severity: str) -> int:
        """Convert severity to task priority (1-10)."""
        severity_map = {
            Severity.CRITICAL: 1,
            Severity.HIGH: 3,
            Severity.MEDIUM: 5,
            Severity.LOW: 7,
            Severity.INFO: 9
        }
        return severity_map.get(severity, 5)

    def analyze_report(self, report: ModlAiPayload) -> List[TaskDefinition]:
        """
        High-level method to analyze a report and create tasks.

        This is a convenience method that combines all three phases.

        Args:
            report: Modl.ai report payload

        Returns:
            List of TaskDefinitions ready for Closed Loop processing
        """
        logger.info(f"🔍 Analyzing {report.report_type.value} report...")

        # Create temporary task to hold the report
        temp_task = TaskDefinition(
            task_id=f"qa_analysis_{int(datetime.utcnow().timestamp())}",
            task_type=TaskType.TESTING,
            priority=5,
            assigned_agent=self.agent_id,
            context={
                "description": "QA Report Analysis",
                "report": report.model_dump()
            },
            requirements={}
        )

        # Phase 1: Gather context
        context_response = self.gather_context(temp_task)
        if not context_response.success:
            logger.error(f"Failed to gather context: {context_response.error}")
            return []

        # Phase 2: Create tasks
        action_response = self.take_action(temp_task, context_response.data)
        if not action_response.success:
            logger.error(f"Failed to create tasks: {action_response.error}")
            return []

        # Phase 3: Verify and score
        verify_response = self.verify_work(temp_task, action_response.data)
        if not verify_response.success:
            logger.error(f"Failed to verify tasks: {verify_response.error}")
            return action_response.data.get('tasks', [])

        tasks = verify_response.data.get('tasks', [])
        avg_confidence = verify_response.data.get('average_confidence', 0)
        auto_merge_count = verify_response.data.get('auto_merge_count', 0)

        logger.info(
            f"✅ Generated {len(tasks)} tasks "
            f"(avg confidence: {avg_confidence:.1f}%, "
            f"auto-merge eligible: {auto_merge_count})"
        )

        return tasks
