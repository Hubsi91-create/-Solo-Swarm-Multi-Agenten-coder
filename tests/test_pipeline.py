"""
Pipeline Tests - End-to-End Integration Testing
Tests the complete pipeline from feature request to tested code
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agents.orchestrator.architect_agent import ArchitectAgent
from core.tdf_schema import TaskDefinition, TaskType
from core.self_correction import SelfCorrectionManager, FailureInfo


class TestSelfCorrectionManager:
    """Test suite for SelfCorrectionManager"""

    def test_initialization(self):
        """Test SelfCorrectionManager initialization"""
        manager = SelfCorrectionManager(max_retries=3)

        assert manager.max_retries == 3
        assert len(manager.correction_history) == 0

    def test_analyze_test_failures_with_pytest_output(self):
        """Test parsing pytest output with failures"""
        manager = SelfCorrectionManager()

        # Simulated pytest output with failure
        test_output = """
============================= test session starts ==============================
FAILED tests/test_calculator.py::TestCalculator::test_add - AssertionError: Expected 5, got 4
FAILED tests/test_calculator.py::TestCalculator::test_subtract - TypeError: unsupported operand type(s) for -: 'str' and 'int'
=========================== short test summary info ============================
2 failed, 0 passed in 0.12s
"""

        tasks = manager.analyze_test_failures(test_output, original_task_id="calc_001")

        assert len(tasks) == 2
        assert all(isinstance(task, TaskDefinition) for task in tasks)
        assert all(task.task_type == TaskType.IMPLEMENTATION for task in tasks)
        assert all(task.priority == 1 for task in tasks)
        assert all(task.assigned_agent == "coder_agent" for task in tasks)

        # Check first task
        task1 = tasks[0]
        assert "test_add" in task1.context["test_name"]
        assert task1.context["error_type"] == "AssertionError"
        assert "Expected 5, got 4" in task1.context["error_message"]

        # Check second task
        task2 = tasks[1]
        assert "test_subtract" in task2.context["test_name"]
        assert task2.context["error_type"] == "TypeError"

    def test_analyze_test_failures_no_failures(self):
        """Test parsing pytest output with no failures"""
        manager = SelfCorrectionManager()

        # Simulated pytest output with all passing
        test_output = """
============================= test session starts ==============================
collected 5 items

tests/test_calculator.py .....                                           [100%]

============================== 5 passed in 0.05s ===============================
"""

        tasks = manager.analyze_test_failures(test_output)

        assert len(tasks) == 0

    def test_analyze_test_failures_simple_format(self):
        """Test parsing simple error format (fallback)"""
        manager = SelfCorrectionManager()

        # Simple error output
        test_output = """
AssertionError: Values do not match
TypeError: Cannot add string and int
ValueError: Invalid input value
"""

        tasks = manager.analyze_test_failures(test_output)

        # Should detect at least some errors
        assert len(tasks) > 0

    def test_should_retry(self):
        """Test retry logic"""
        manager = SelfCorrectionManager(max_retries=3)

        assert manager.should_retry(0) is True
        assert manager.should_retry(1) is True
        assert manager.should_retry(2) is True
        assert manager.should_retry(3) is False
        assert manager.should_retry(4) is False

    def test_correction_history(self):
        """Test correction history tracking"""
        manager = SelfCorrectionManager()

        test_output = "FAILED test.py::test_func - AssertionError: fail"

        manager.analyze_test_failures(test_output, original_task_id="task_001")
        manager.analyze_test_failures(test_output, original_task_id="task_002")

        history = manager.get_correction_history()

        assert len(history) == 2
        assert all("timestamp" in record for record in history)
        assert all("failure_count" in record for record in history)

    def test_reset_history(self):
        """Test resetting correction history"""
        manager = SelfCorrectionManager()

        test_output = "FAILED test.py::test_func - AssertionError: fail"
        manager.analyze_test_failures(test_output)

        assert len(manager.correction_history) > 0

        manager.reset_history()

        assert len(manager.correction_history) == 0


class TestFailureInfo:
    """Test suite for FailureInfo class"""

    def test_failure_info_creation(self):
        """Test creating FailureInfo object"""
        failure = FailureInfo(
            test_name="test_addition",
            error_type="AssertionError",
            error_message="Expected 5, got 4",
            file_path="tests/test_math.py",
            line_number=42,
            traceback="Traceback info here"
        )

        assert failure.test_name == "test_addition"
        assert failure.error_type == "AssertionError"
        assert failure.error_message == "Expected 5, got 4"
        assert failure.file_path == "tests/test_math.py"
        assert failure.line_number == 42
        assert failure.traceback == "Traceback info here"

    def test_failure_info_repr(self):
        """Test FailureInfo repr"""
        failure = FailureInfo(
            test_name="test_func",
            error_type="TypeError",
            error_message="Type mismatch",
            file_path="test.py",
            line_number=10
        )

        repr_str = repr(failure)

        assert "test_func" in repr_str
        assert "TypeError" in repr_str


class TestIntegrationPipeline:
    """Test suite for end-to-end pipeline integration"""

    def test_integrate_and_test_with_passing_code(self):
        """Test integration pipeline with code that passes tests"""
        architect = ArchitectAgent(
            agent_id="test_architect",
            agent_name="TestArchitect"
        )

        try:
            # Worker results with correct Calculator implementation AND tests together
            worker_results = [
                {
                    "code": """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

# Tests
def test_calculator_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_calculator_subtract():
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
"""
                }
            ]

            # Run integration (without separate test file - tests are in the code)
            response = architect.integrate_and_test(
                worker_results=worker_results
            )

            # Should succeed
            assert response.success is True
            assert response.data["success"] is True
            assert response.data["iterations"] >= 1
            assert "code_file" in response.data

        finally:
            # Cleanup
            if response.data and "code_file" in response.data:
                code_file = response.data["code_file"]
                if os.path.exists(code_file):
                    os.unlink(code_file)

    def test_integrate_and_test_with_failing_code(self):
        """Test integration pipeline with code that fails tests"""
        architect = ArchitectAgent(
            agent_id="test_architect",
            agent_name="TestArchitect"
        )

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
# Test file for Calculator
def test_calculator_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5
""")
            test_file = f.name

        try:
            # Worker results with INCORRECT Calculator implementation
            worker_results = [
                {
                    "code": """
class Calculator:
    def add(self, a, b):
        return a + b + 1  # BUG: Adding 1 extra!
"""
                }
            ]

            # Run integration
            response = architect.integrate_and_test(
                worker_results=worker_results,
                test_file_path=test_file
            )

            # Should fail after max retries
            assert response.success is False
            assert response.error is not None
            assert response.error.error_type == "tests_failed"
            assert response.data["success"] is False
            assert response.data["bugfix_tasks_generated"] > 0

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.unlink(test_file)
            if response.data and "code_file" in response.data:
                code_file = response.data["code_file"]
                if os.path.exists(code_file):
                    os.unlink(code_file)

    def test_integrate_and_test_retry_loop(self):
        """Test that retry loop is triggered and generates bugfix tasks"""
        architect = ArchitectAgent(
            agent_id="test_architect",
            agent_name="TestArchitect"
        )

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    result = broken_function()
    assert result == 42
""")
            test_file = f.name

        try:
            # Worker results with broken function
            worker_results = [
                {
                    "code": """
def broken_function():
    return 41  # Wrong value!
"""
                }
            ]

            # Run integration
            response = architect.integrate_and_test(
                worker_results=worker_results,
                test_file_path=test_file
            )

            # Should fail and generate bugfix tasks
            assert response.data["bugfix_tasks_generated"] > 0
            assert len(response.data["bugfix_tasks"]) > 0

            # Check bugfix task structure
            bugfix_task = response.data["bugfix_tasks"][0]
            assert bugfix_task["task_type"] == "implementation"
            assert bugfix_task["priority"] == 1
            assert bugfix_task["assigned_agent"] == "coder_agent"
            assert "bug_type" in bugfix_task["context"]
            assert bugfix_task["context"]["bug_type"] == "test_failure"

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.unlink(test_file)
            if response.data and "code_file" in response.data:
                code_file = response.data["code_file"]
                if os.path.exists(code_file):
                    os.unlink(code_file)

    def test_integrate_and_test_with_empty_worker_results(self):
        """Test integration with empty worker results"""
        architect = ArchitectAgent(
            agent_id="test_architect",
            agent_name="TestArchitect"
        )

        # Empty worker results
        worker_results = []

        # Run integration
        response = architect.integrate_and_test(worker_results=worker_results)

        # Should fail with merge error
        assert response.success is False
        assert response.error is not None
        assert response.error.error_type == "merge_failed"

    def test_merge_worker_results(self):
        """Test merging multiple worker results"""
        architect = ArchitectAgent(
            agent_id="test_architect",
            agent_name="TestArchitect"
        )

        worker_results = [
            {"code": "def func1():\n    return 1\n"},
            {"code": "def func2():\n    return 2\n"},
            {"output": "def func3():\n    return 3\n"}  # Using 'output' key
        ]

        merged = architect._merge_worker_results(worker_results)

        assert "func1" in merged
        assert "func2" in merged
        assert "func3" in merged
        assert "Worker 1 Output" in merged
        assert "Worker 2 Output" in merged
        assert "Worker 3 Output" in merged

    def test_feature_request_calculator_simulation(self):
        """
        End-to-end simulation: Feature Request for Calculator class

        Simulates:
        1. User requests: "Create Calculator class with add/subtract"
        2. Worker delivers faulty code first (add returns wrong value)
        3. Architect detects failure
        4. SelfCorrectionManager generates bugfix task
        5. (In real system: worker would fix, we just verify task generation)
        """
        architect = ArchitectAgent(
            agent_id="test_architect",
            agent_name="TestArchitect"
        )

        # Create test file for Calculator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_calculator_add():
    calc = Calculator()
    result = calc.add(10, 5)
    assert result == 15, f"Expected 15, got {result}"

def test_calculator_subtract():
    calc = Calculator()
    result = calc.subtract(10, 5)
    assert result == 5, f"Expected 5, got {result}"
""")
            test_file = f.name

        try:
            # Simulate worker response with FAULTY code
            faulty_worker_results = [
                {
                    "code": """
class Calculator:
    '''Calculator class with basic operations'''

    def add(self, a, b):
        # BUG: Returns wrong value!
        return a + b - 1

    def subtract(self, a, b):
        return a - b
"""
                }
            ]

            # Run pipeline with faulty code
            response = architect.integrate_and_test(
                worker_results=faulty_worker_results,
                test_file_path=test_file
            )

            # Assertions
            assert response.success is False  # Tests should fail
            assert response.data["success"] is False
            assert response.data["iterations"] == 4  # Max iterations (1 initial + 3 retries)
            assert response.data["bugfix_tasks_generated"] > 0  # Should generate bugfix tasks

            # Verify bugfix tasks were generated correctly
            bugfix_tasks = response.data["bugfix_tasks"]
            assert len(bugfix_tasks) > 0

            # Check first bugfix task
            first_bugfix = bugfix_tasks[0]
            assert first_bugfix["task_type"] == "implementation"
            assert first_bugfix["context"]["bug_type"] == "test_failure"
            assert first_bugfix["requirements"]["must_pass_tests"] is True

            # Verify that test_calculator_add failure was detected
            found_add_failure = any(
                "test_calculator_add" in task["context"].get("test_name", "")
                for task in bugfix_tasks
            )
            assert found_add_failure, "Should detect test_calculator_add failure"

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.unlink(test_file)
            if response.data and "code_file" in response.data:
                code_file = response.data["code_file"]
                if os.path.exists(code_file):
                    os.unlink(code_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
