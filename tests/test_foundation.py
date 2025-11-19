"""
Foundation Tests - Testing core functionality
Tests for TaskDefinition validation and TokenTracker calculations
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType


class TestTaskDefinition:
    """Test suite for TaskDefinition Pydantic model"""

    def test_valid_task_creation(self):
        """Test creating a valid TaskDefinition"""
        task = TaskDefinition(
            task_id="test_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="test_agent",
            context={"language": "python"},
            requirements={"timeout": 300}
        )

        assert task.task_id == "test_001"
        assert task.task_type == TaskType.IMPLEMENTATION
        assert task.priority == 1
        assert task.assigned_agent == "test_agent"
        assert task.context == {"language": "python"}
        assert task.requirements == {"timeout": 300}
        assert task.status == "pending"
        assert isinstance(task.created_at, datetime)

    def test_task_with_defaults(self):
        """Test TaskDefinition with default values"""
        task = TaskDefinition(
            task_id="test_002",
            task_type=TaskType.ANALYSIS,
            priority=5,
            assigned_agent="analyzer"
        )

        assert task.context == {}
        assert task.requirements == {}
        assert task.status == "pending"
        assert task.result is None

    def test_invalid_task_id_pattern(self):
        """Test that invalid task_id pattern raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            TaskDefinition(
                task_id="test 003",  # Contains space - invalid
                task_type=TaskType.TESTING,
                priority=3,
                assigned_agent="tester"
            )

        assert "task_id" in str(exc_info.value)

    def test_invalid_priority_range(self):
        """Test that priority outside valid range raises ValidationError"""
        # Priority too low (< 1)
        with pytest.raises(ValidationError):
            TaskDefinition(
                task_id="test_004",
                task_type=TaskType.REVIEW,
                priority=0,
                assigned_agent="reviewer"
            )

        # Priority too high (> 10)
        with pytest.raises(ValidationError):
            TaskDefinition(
                task_id="test_005",
                task_type=TaskType.REVIEW,
                priority=11,
                assigned_agent="reviewer"
            )

    def test_invalid_status_pattern(self):
        """Test that invalid status raises ValidationError"""
        with pytest.raises(ValidationError):
            TaskDefinition(
                task_id="test_006",
                task_type=TaskType.DEPLOYMENT,
                priority=2,
                assigned_agent="deployer",
                status="invalid_status"
            )

    def test_task_type_enum(self):
        """Test all TaskType enum values"""
        valid_types = [
            TaskType.ANALYSIS,
            TaskType.IMPLEMENTATION,
            TaskType.REVIEW,
            TaskType.TESTING,
            TaskType.DOCUMENTATION,
            TaskType.DEPLOYMENT,
            TaskType.RESEARCH,
            TaskType.REFACTORING
        ]

        for task_type in valid_types:
            task = TaskDefinition(
                task_id=f"test_{task_type.value}",
                task_type=task_type,
                priority=5,
                assigned_agent="agent"
            )
            assert task.task_type == task_type

    def test_strict_json_output(self):
        """Test to_strict_json method"""
        task = TaskDefinition(
            task_id="test_007",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="coder",
            context={"framework": "fastapi"},
            requirements={"max_retries": 3}
        )

        json_output = task.to_strict_json()

        assert isinstance(json_output, dict)
        assert json_output["task_id"] == "test_007"
        assert json_output["task_type"] == "implementation"
        assert json_output["priority"] == 1
        assert json_output["assigned_agent"] == "coder"
        assert json_output["context"] == {"framework": "fastapi"}
        assert json_output["requirements"] == {"max_retries": 3}
        assert "created_at" in json_output

    def test_update_status_method(self):
        """Test update_status method"""
        task = TaskDefinition(
            task_id="test_008",
            task_type=TaskType.TESTING,
            priority=3,
            assigned_agent="tester"
        )

        assert task.status == "pending"

        task.update_status("in_progress")
        assert task.status == "in_progress"

        task.update_status("completed")
        assert task.status == "completed"

        # Invalid status should raise error
        with pytest.raises(ValueError) as exc_info:
            task.update_status("invalid")
        assert "Invalid status" in str(exc_info.value)

    def test_set_result_method(self):
        """Test set_result method"""
        task = TaskDefinition(
            task_id="test_009",
            task_type=TaskType.ANALYSIS,
            priority=2,
            assigned_agent="analyzer"
        )

        assert task.result is None
        assert task.status == "pending"

        result_data = {"findings": ["item1", "item2"], "score": 95}
        task.set_result(result_data)

        assert task.result == result_data
        assert task.status == "completed"

    def test_context_validation(self):
        """Test context field validation"""
        # Valid context
        task = TaskDefinition(
            task_id="test_010",
            task_type=TaskType.RESEARCH,
            priority=4,
            assigned_agent="researcher",
            context={"topic": "AI", "depth": "deep"}
        )
        assert task.context == {"topic": "AI", "depth": "deep"}

    def test_requirements_validation(self):
        """Test requirements field validation"""
        # Valid requirements
        task = TaskDefinition(
            task_id="test_011",
            task_type=TaskType.REFACTORING,
            priority=6,
            assigned_agent="refactorer",
            requirements={"preserve_tests": True, "max_complexity": 10}
        )
        assert task.requirements == {"preserve_tests": True, "max_complexity": 10}


class TestTokenTracker:
    """Test suite for TokenTracker"""

    def test_tracker_initialization(self):
        """Test TokenTracker initialization"""
        tracker = TokenTracker()

        assert len(tracker.usage_records) == 0
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_track_usage_haiku(self):
        """Test tracking usage for Haiku model"""
        tracker = TokenTracker()

        # Haiku: $1.00 input, $5.00 output per million
        # 1000 input tokens = $0.001
        # 2000 output tokens = $0.010
        # Total = $0.011
        cost = tracker.track_usage(
            model=ModelType.HAIKU_3_5,
            input_tokens=1000,
            output_tokens=2000
        )

        expected_cost = (1000 / 1_000_000) * 1.00 + (2000 / 1_000_000) * 5.00
        assert cost == pytest.approx(expected_cost, rel=1e-9)
        assert cost == pytest.approx(0.011, rel=1e-9)

        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 2000
        assert tracker.total_cost_usd == pytest.approx(0.011, rel=1e-9)
        assert len(tracker.usage_records) == 1

    def test_track_usage_sonnet(self):
        """Test tracking usage for Sonnet model"""
        tracker = TokenTracker()

        # Sonnet: $3.00 input, $15.00 output per million
        # 5000 input tokens = $0.015
        # 3000 output tokens = $0.045
        # Total = $0.060
        cost = tracker.track_usage(
            model=ModelType.SONNET_3_5,
            input_tokens=5000,
            output_tokens=3000
        )

        expected_cost = (5000 / 1_000_000) * 3.00 + (3000 / 1_000_000) * 15.00
        assert cost == pytest.approx(expected_cost, rel=1e-9)
        assert cost == pytest.approx(0.060, rel=1e-9)

        assert tracker.total_input_tokens == 5000
        assert tracker.total_output_tokens == 3000
        assert tracker.total_cost_usd == pytest.approx(0.060, rel=1e-9)

    def test_track_multiple_usages(self):
        """Test tracking multiple API calls"""
        tracker = TokenTracker()

        # First call - Haiku
        cost1 = tracker.track_usage(ModelType.HAIKU_3_5, 1000, 500)

        # Second call - Sonnet
        cost2 = tracker.track_usage(ModelType.SONNET_3_5, 2000, 1000)

        # Third call - Haiku
        cost3 = tracker.track_usage(ModelType.HAIKU_3_5, 500, 250)

        assert len(tracker.usage_records) == 3
        assert tracker.total_input_tokens == 3500
        assert tracker.total_output_tokens == 1750
        assert tracker.total_cost_usd == pytest.approx(cost1 + cost2 + cost3, rel=1e-9)

    def test_invalid_model(self):
        """Test that invalid model raises ValueError"""
        tracker = TokenTracker()

        with pytest.raises(ValueError) as exc_info:
            tracker.track_usage("invalid-model", 1000, 500)

        assert "Unsupported model" in str(exc_info.value)

    def test_negative_tokens(self):
        """Test that negative token counts raise ValueError"""
        tracker = TokenTracker()

        with pytest.raises(ValueError) as exc_info:
            tracker.track_usage(ModelType.HAIKU_3_5, -100, 500)
        assert "non-negative" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            tracker.track_usage(ModelType.HAIKU_3_5, 100, -500)
        assert "non-negative" in str(exc_info.value)

    def test_get_summary(self):
        """Test get_summary method"""
        tracker = TokenTracker()

        tracker.track_usage(ModelType.HAIKU_3_5, 1000, 500)
        tracker.track_usage(ModelType.SONNET_3_5, 2000, 1000)
        tracker.track_usage(ModelType.HAIKU_3_5, 500, 250)

        summary = tracker.get_summary()

        assert summary.total_input_tokens == 3500
        assert summary.total_output_tokens == 1750
        assert summary.total_tokens == 5250
        assert summary.record_count == 3

        # Check by-model breakdown
        assert ModelType.HAIKU_3_5 in summary.by_model
        assert ModelType.SONNET_3_5 in summary.by_model

        haiku_stats = summary.by_model[ModelType.HAIKU_3_5]
        assert haiku_stats["input_tokens"] == 1500  # 1000 + 500
        assert haiku_stats["output_tokens"] == 750  # 500 + 250
        assert haiku_stats["call_count"] == 2

        sonnet_stats = summary.by_model[ModelType.SONNET_3_5]
        assert sonnet_stats["input_tokens"] == 2000
        assert sonnet_stats["output_tokens"] == 1000
        assert sonnet_stats["call_count"] == 1

    def test_calculate_estimated_cost(self):
        """Test calculate_estimated_cost without tracking"""
        tracker = TokenTracker()

        # Calculate cost for Haiku without tracking
        estimated = tracker.calculate_estimated_cost(
            ModelType.HAIKU_3_5,
            1000,
            2000
        )

        expected = (1000 / 1_000_000) * 1.00 + (2000 / 1_000_000) * 5.00
        assert estimated == pytest.approx(expected, rel=1e-9)

        # Verify nothing was tracked
        assert len(tracker.usage_records) == 0
        assert tracker.total_cost_usd == 0.0

    def test_reset(self):
        """Test reset method"""
        tracker = TokenTracker()

        # Add some usage
        tracker.track_usage(ModelType.HAIKU_3_5, 1000, 500)
        tracker.track_usage(ModelType.SONNET_3_5, 2000, 1000)

        assert len(tracker.usage_records) == 2
        assert tracker.total_cost_usd > 0

        # Reset
        tracker.reset()

        assert len(tracker.usage_records) == 0
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_usage_with_metadata(self):
        """Test tracking usage with metadata"""
        tracker = TokenTracker()

        metadata = {"agent_id": "test_agent", "task_id": "task_001"}
        tracker.track_usage(
            ModelType.HAIKU_3_5,
            1000,
            500,
            metadata=metadata
        )

        record = tracker.usage_records[0]
        assert record.metadata == metadata

    def test_cost_precision(self):
        """Test cost calculation precision"""
        tracker = TokenTracker()

        # Small token counts should still calculate correctly
        cost = tracker.track_usage(ModelType.HAIKU_3_5, 1, 1)

        expected = (1 / 1_000_000) * 1.00 + (1 / 1_000_000) * 5.00
        assert cost == pytest.approx(expected, rel=1e-9)
        assert cost == pytest.approx(0.000006, rel=1e-9)

    def test_export_records(self):
        """Test exporting usage records"""
        tracker = TokenTracker()

        tracker.track_usage(ModelType.HAIKU_3_5, 1000, 500)
        tracker.track_usage(ModelType.SONNET_3_5, 2000, 1000)

        exported = tracker.export_records()

        assert len(exported) == 2
        assert all(isinstance(record, dict) for record in exported)
        assert all("timestamp" in record for record in exported)
        assert all("model" in record for record in exported)
        assert all("cost_usd" in record for record in exported)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
