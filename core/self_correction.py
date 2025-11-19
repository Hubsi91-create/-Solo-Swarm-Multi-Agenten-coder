"""
Self-Correction Manager - Analyzes test failures and generates bugfix tasks
Implements iterative self-correction loop for code quality
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.tdf_schema import TaskDefinition, TaskType


logger = logging.getLogger(__name__)


class FailureInfo:
    """
    Represents a single test failure with extracted information.
    """

    def __init__(
        self,
        test_name: str,
        error_type: str,
        error_message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        traceback: Optional[str] = None
    ):
        self.test_name = test_name
        self.error_type = error_type
        self.error_message = error_message
        self.file_path = file_path
        self.line_number = line_number
        self.traceback = traceback

    def __repr__(self) -> str:
        return (
            f"TestFailure(test='{self.test_name}', "
            f"error='{self.error_type}', "
            f"file='{self.file_path}:{self.line_number}')"
        )


class SelfCorrectionManager:
    """
    Manages self-correction loops for code quality.

    Responsibilities:
    - Parse test output from pytest
    - Extract failure information
    - Generate TaskDefinition objects for bugfixes
    - Track correction iterations
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize SelfCorrectionManager.

        Args:
            max_retries: Maximum number of correction attempts
        """
        self.max_retries = max_retries
        self.correction_history: List[Dict[str, Any]] = []

        logger.info(f"SelfCorrectionManager initialized with max_retries={max_retries}")

    def analyze_test_failures(
        self,
        test_output: str,
        original_task_id: Optional[str] = None
    ) -> List[TaskDefinition]:
        """
        Analyze test failure output and generate bugfix tasks.

        Parses pytest output to extract:
        - Failed test names
        - Error types (AssertionError, TypeError, etc.)
        - Error messages
        - File paths and line numbers
        - Tracebacks

        Args:
            test_output: Raw pytest output string
            original_task_id: Optional ID of the task that produced failing code

        Returns:
            List of TaskDefinition objects for bugfixes
        """
        logger.info("Analyzing test failures from output...")

        # Parse failures from test output
        failures = self._parse_pytest_output(test_output)

        if not failures:
            logger.info("No test failures found in output")
            return []

        logger.info(f"Found {len(failures)} test failures")

        # Generate bugfix tasks
        bugfix_tasks = []
        for idx, failure in enumerate(failures):
            task = self._create_bugfix_task(failure, idx, original_task_id)
            bugfix_tasks.append(task)

        # Record in history
        self.correction_history.append({
            "timestamp": datetime.utcnow(),
            "failure_count": len(failures),
            "tasks_generated": len(bugfix_tasks),
            "original_task_id": original_task_id
        })

        logger.info(f"Generated {len(bugfix_tasks)} bugfix tasks")
        return bugfix_tasks

    def _parse_pytest_output(self, test_output: str) -> List[FailureInfo]:
        """
        Parse pytest output to extract test failures.

        Pytest output format:
        ```
        FAILED tests/test_file.py::TestClass::test_method - AssertionError: ...
        ...
        tests/test_file.py:42: AssertionError
        ```

        Args:
            test_output: Raw pytest output

        Returns:
            List of FailureInfo objects
        """
        failures = []

        # Pattern for FAILED lines: "FAILED path::test_name - ErrorType: message"
        failed_pattern = r'FAILED\s+([\w/._-]+)::([\w:]+)\s*-\s*([\w]+):\s*(.*?)(?=\n|$)'

        # Pattern for traceback file location: "path/to/file.py:42: ErrorType"
        traceback_pattern = r'([\w/._-]+\.py):(\d+):\s*([\w]+)'

        # Find all FAILED lines
        failed_matches = re.finditer(failed_pattern, test_output, re.MULTILINE)

        for match in failed_matches:
            file_path = match.group(1)
            test_name = match.group(2)
            error_type = match.group(3)
            error_message = match.group(4).strip()

            # Try to find line number from traceback
            line_number = None
            traceback = None

            # Look for traceback info after this FAILED line
            start_pos = match.end()
            next_section = test_output[start_pos:start_pos + 500]  # Next 500 chars

            traceback_match = re.search(traceback_pattern, next_section)
            if traceback_match:
                line_number = int(traceback_match.group(2))
                traceback = next_section[:traceback_match.end()]

            failure = FailureInfo(
                test_name=test_name,
                error_type=error_type,
                error_message=error_message,
                file_path=file_path,
                line_number=line_number,
                traceback=traceback
            )

            failures.append(failure)
            logger.debug(f"Parsed failure: {failure}")

        # Alternative parsing if no FAILED lines found (simpler format)
        if not failures:
            failures = self._parse_simple_format(test_output)

        return failures

    def _parse_simple_format(self, test_output: str) -> List[FailureInfo]:
        """
        Parse simpler test output format (fallback).

        Args:
            test_output: Raw test output

        Returns:
            List of FailureInfo objects
        """
        failures = []

        # Look for common error patterns
        error_patterns = [
            r'(AssertionError|TypeError|ValueError|AttributeError):\s*(.*?)(?=\n|$)',
            r'Error:\s*(.*?)(?=\n|$)',
            r'FAIL:\s*(.*?)(?=\n|$)'
        ]

        for pattern in error_patterns:
            matches = re.finditer(pattern, test_output, re.MULTILINE)
            for idx, match in enumerate(matches):
                if len(match.groups()) >= 2:
                    error_type = match.group(1)
                    error_message = match.group(2).strip()
                else:
                    error_type = "Error"
                    error_message = match.group(1).strip()

                failure = FailureInfo(
                    test_name=f"unknown_test_{idx}",
                    error_type=error_type,
                    error_message=error_message
                )
                failures.append(failure)

            if failures:
                break  # Stop after first successful pattern

        return failures

    def _create_bugfix_task(
        self,
        failure: FailureInfo,
        index: int,
        original_task_id: Optional[str] = None
    ) -> TaskDefinition:
        """
        Create a TaskDefinition for fixing a test failure.

        Args:
            failure: FailureInfo object with error details
            index: Index of this failure in the list
            original_task_id: Optional ID of the original task

        Returns:
            TaskDefinition for bugfix
        """
        # Generate task ID
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        task_id = f"bugfix_{timestamp}_{index}"

        # Build context with failure details
        context = {
            "bug_type": "test_failure",
            "test_name": failure.test_name,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
            "language": "python"
        }

        if failure.file_path:
            context["file_path"] = failure.file_path
        if failure.line_number:
            context["line_number"] = failure.line_number
        if failure.traceback:
            context["traceback"] = failure.traceback
        if original_task_id:
            context["original_task_id"] = original_task_id

        # Build requirements
        requirements = {
            "must_pass_tests": True,
            "preserve_existing_functionality": True,
            "timeout": 600  # 10 minutes for bugfix
        }

        # Create TaskDefinition
        task = TaskDefinition(
            task_id=task_id,
            task_type=TaskType.IMPLEMENTATION,  # Bugfix is implementation
            priority=1,  # High priority for bugfixes
            assigned_agent="coder_agent",
            context=context,
            requirements=requirements
        )

        logger.debug(f"Created bugfix task: {task}")
        return task

    def should_retry(self, iteration: int) -> bool:
        """
        Determine if another correction iteration should be attempted.

        Args:
            iteration: Current iteration number (0-indexed)

        Returns:
            True if should retry, False if max retries reached
        """
        return iteration < self.max_retries

    def get_correction_history(self) -> List[Dict[str, Any]]:
        """
        Get the correction history.

        Returns:
            List of correction attempt records
        """
        return self.correction_history

    def reset_history(self) -> None:
        """Reset correction history."""
        self.correction_history = []
        logger.info("Correction history reset")

    def __repr__(self) -> str:
        return (
            f"SelfCorrectionManager(max_retries={self.max_retries}, "
            f"history_count={len(self.correction_history)})"
        )
