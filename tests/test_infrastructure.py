"""
Infrastructure Tests - Testing Queue Manager, Agent Pool, and Worker Agents
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from core.tdf_schema import TaskDefinition, TaskType
from infrastructure.agent_pool import AgentPoolManager, SlotAllocationError
from agents.workers.coder_agent import CoderAgent


class TestAgentPoolManager:
    """Test suite for AgentPoolManager"""

    def test_pool_initialization(self):
        """Test AgentPoolManager initialization"""
        pool = AgentPoolManager(max_slots=100)

        assert pool.max_slots == 100
        assert pool.used_slots == 0
        assert pool.available_slots == 100
        assert pool.utilization_percentage == 0.0

    def test_pool_initialization_invalid_slots(self):
        """Test that invalid max_slots raises ValueError"""
        with pytest.raises(ValueError):
            AgentPoolManager(max_slots=0)

        with pytest.raises(ValueError):
            AgentPoolManager(max_slots=-10)

    def test_allocate_slot(self):
        """Test successful slot allocation"""
        pool = AgentPoolManager(max_slots=10)

        slot_id = pool.allocate_slot(agent_type="coder")

        assert pool.used_slots == 1
        assert pool.available_slots == 9
        assert pool.utilization_percentage == 10.0
        assert slot_id in pool.active_slots

        allocation = pool.get_slot_info(slot_id)
        assert allocation is not None
        assert allocation.agent_type == "coder"
        assert isinstance(allocation.allocated_at, datetime)

    def test_allocate_multiple_slots(self):
        """Test allocating multiple slots"""
        pool = AgentPoolManager(max_slots=100)

        slot_ids = []
        for i in range(50):
            slot_id = pool.allocate_slot(agent_type=f"agent_{i}")
            slot_ids.append(slot_id)

        assert pool.used_slots == 50
        assert pool.available_slots == 50
        assert pool.utilization_percentage == 50.0

        # All slot IDs should be unique
        assert len(set(slot_ids)) == 50

    def test_allocate_slot_with_metadata(self):
        """Test slot allocation with metadata"""
        pool = AgentPoolManager(max_slots=10)

        metadata = {"task_id": "task_001", "priority": 1}
        slot_id = pool.allocate_slot(
            agent_type="coder",
            metadata=metadata
        )

        allocation = pool.get_slot_info(slot_id)
        assert allocation.metadata == metadata

    def test_allocate_slot_limit_reached(self):
        """Test that allocation fails when limit is reached"""
        pool = AgentPoolManager(max_slots=5)

        # Allocate all slots
        for i in range(5):
            pool.allocate_slot(agent_type="agent")

        assert pool.used_slots == 5
        assert pool.available_slots == 0

        # Attempting to allocate another should raise error
        with pytest.raises(SlotAllocationError) as exc_info:
            pool.allocate_slot(agent_type="agent")

        assert "No slots available" in str(exc_info.value)
        assert pool.allocation_failures == 1

    def test_release_slot(self):
        """Test slot release"""
        pool = AgentPoolManager(max_slots=10)

        slot_id = pool.allocate_slot(agent_type="coder")
        assert pool.used_slots == 1

        released = pool.release_slot(slot_id)
        assert released is True
        assert pool.used_slots == 0
        assert pool.available_slots == 10

        # Slot should no longer be in active slots
        assert slot_id not in pool.active_slots

    def test_release_non_existent_slot(self):
        """Test releasing a slot that doesn't exist"""
        pool = AgentPoolManager(max_slots=10)

        released = pool.release_slot("non_existent_slot")
        assert released is False

    def test_release_slot_empty_id(self):
        """Test that empty slot_id raises ValueError"""
        pool = AgentPoolManager(max_slots=10)

        with pytest.raises(ValueError):
            pool.release_slot("")

        with pytest.raises(ValueError):
            pool.release_slot(None)

    def test_is_slot_available(self):
        """Test is_slot_available method"""
        pool = AgentPoolManager(max_slots=3)

        assert pool.is_slot_available() is True

        # Allocate all slots
        pool.allocate_slot(agent_type="agent1")
        pool.allocate_slot(agent_type="agent2")
        pool.allocate_slot(agent_type="agent3")

        assert pool.is_slot_available() is False

        # Release one slot
        slot_id = list(pool.active_slots.keys())[0]
        pool.release_slot(slot_id)

        assert pool.is_slot_available() is True

    def test_get_active_slots(self):
        """Test get_active_slots method"""
        pool = AgentPoolManager(max_slots=10)

        slot1 = pool.allocate_slot(agent_type="coder")
        slot2 = pool.allocate_slot(agent_type="reviewer")

        active = pool.get_active_slots()

        assert len(active) == 2
        assert slot1 in active
        assert slot2 in active

    def test_get_slots_by_agent_type(self):
        """Test filtering slots by agent type"""
        pool = AgentPoolManager(max_slots=10)

        pool.allocate_slot(agent_type="coder")
        pool.allocate_slot(agent_type="coder")
        pool.allocate_slot(agent_type="reviewer")

        coder_slots = pool.get_slots_by_agent_type("coder")
        reviewer_slots = pool.get_slots_by_agent_type("reviewer")

        assert len(coder_slots) == 2
        assert len(reviewer_slots) == 1

    def test_clear_all_slots(self):
        """Test clearing all slots"""
        pool = AgentPoolManager(max_slots=10)

        pool.allocate_slot(agent_type="agent1")
        pool.allocate_slot(agent_type="agent2")
        pool.allocate_slot(agent_type="agent3")

        assert pool.used_slots == 3

        cleared = pool.clear_all_slots()

        assert cleared == 3
        assert pool.used_slots == 0
        assert pool.available_slots == 10

    def test_get_statistics(self):
        """Test get_statistics method"""
        pool = AgentPoolManager(max_slots=100)

        pool.allocate_slot(agent_type="coder")
        pool.allocate_slot(agent_type="coder")
        pool.allocate_slot(agent_type="reviewer")

        stats = pool.get_statistics()

        assert stats["max_slots"] == 100
        assert stats["used_slots"] == 3
        assert stats["available_slots"] == 97
        assert stats["utilization_percentage"] == 3.0
        assert stats["total_allocations"] == 3
        assert stats["active_agent_types"]["coder"] == 2
        assert stats["active_agent_types"]["reviewer"] == 1

    def test_pool_context_manager(self):
        """Test AgentPoolManager as context manager"""
        with AgentPoolManager(max_slots=10) as pool:
            pool.allocate_slot(agent_type="agent1")
            pool.allocate_slot(agent_type="agent2")
            assert pool.used_slots == 2

        # After exiting context, all slots should be cleared
        assert pool.used_slots == 0


class TestCoderAgent:
    """Test suite for CoderAgent"""

    def test_coder_agent_initialization(self):
        """Test CoderAgent initialization"""
        agent = CoderAgent(
            agent_id="coder_001",
            agent_name="TestCoder"
        )

        assert agent.agent_id == "coder_001"
        assert agent.agent_name == "TestCoder"
        assert agent.default_language == "python"
        assert agent.token_tracker is not None

    def test_coder_agent_with_config(self):
        """Test CoderAgent initialization with custom config"""
        config = {
            "default_language": "javascript",
            "max_code_lines": 1000,
            "temperature": 0.5
        }

        agent = CoderAgent(
            agent_id="coder_002",
            config=config
        )

        assert agent.default_language == "javascript"
        assert agent.max_code_lines == 1000
        assert agent.temperature == 0.5

    def test_gather_context_implementation_task(self):
        """Test gather_context for IMPLEMENTATION task"""
        agent = CoderAgent(agent_id="coder_003")

        task = TaskDefinition(
            task_id="impl_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="coder_003",
            context={
                "language": "python",
                "framework": "fastapi",
                "specifications": "Create REST API endpoint"
            },
            requirements={
                "include_tests": True,
                "max_lines": 200
            }
        )

        response = agent.gather_context(task)

        assert response.success is True
        assert response.data["language"] == "python"
        assert response.data["framework"] == "fastapi"
        assert response.data["include_tests"] is True
        assert response.data["max_lines"] == 200

    def test_gather_context_unsupported_task_type(self):
        """Test gather_context with unsupported task type"""
        agent = CoderAgent(agent_id="coder_004")

        task = TaskDefinition(
            task_id="review_001",
            task_type=TaskType.REVIEW,  # CoderAgent doesn't handle REVIEW
            priority=1,
            assigned_agent="coder_004"
        )

        response = agent.gather_context(task)

        assert response.success is False
        assert response.error is not None
        assert response.error.error_type == "invalid_task_type"

    def test_take_action_generates_code(self):
        """Test take_action generates code"""
        agent = CoderAgent(agent_id="coder_005")

        task = TaskDefinition(
            task_id="impl_002",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="coder_005",
            context={"language": "python"}
        )

        context = {
            "language": "python",
            "framework": "fastapi",
            "specifications": "Create API endpoint",
            "max_lines": 500
        }

        response = agent.take_action(task, context)

        assert response.success is True
        assert "generated_code" in response.data
        assert response.data["language"] == "python"
        assert "token_usage" in response.data
        assert response.data["token_usage"]["cost_usd"] > 0

    def test_verify_work_success(self):
        """Test verify_work with valid code"""
        agent = CoderAgent(agent_id="coder_006")

        task = TaskDefinition(
            task_id="impl_003",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="coder_006",
            requirements={"max_lines": 500}
        )

        action_result = {
            "generated_code": 'def hello():\n    print("Hello")\n',
            "language": "python",
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "cost_usd": 0.001
            }
        }

        response = agent.verify_work(task, action_result)

        assert response.success is True
        assert response.data["verification_passed"] is True
        assert "validation_results" in response.data

    def test_full_execution_lifecycle(self):
        """Test complete task execution lifecycle"""
        agent = CoderAgent(agent_id="coder_007")

        task = TaskDefinition(
            task_id="impl_004",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="coder_007",
            context={
                "language": "python",
                "framework": "fastapi",
                "specifications": "Create user registration endpoint"
            },
            requirements={
                "include_tests": False,
                "max_lines": 300
            }
        )

        # Execute full lifecycle
        response = agent.execute_task(task)

        assert response.success is True
        assert response.data["verification_passed"] is True
        assert "generated_code" in response.data

        # Check execution history
        history = agent.get_execution_history()
        assert len(history) > 0

    def test_usage_summary(self):
        """Test get_usage_summary method"""
        agent = CoderAgent(agent_id="coder_008")

        task = TaskDefinition(
            task_id="impl_005",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="coder_008",
            context={"language": "python"}
        )

        # Execute task to generate usage
        agent.execute_task(task)

        summary = agent.get_usage_summary()

        assert "agent_id" in summary
        assert "usage_summary" in summary
        assert summary["agent_id"] == "coder_008"


class TestQueueManagerMocked:
    """Test suite for QueueManager with mocked Redis"""

    @pytest.fixture
    def mock_redis(self):
        """Fixture to mock Redis"""
        with patch('infrastructure.queue_manager.REDIS_AVAILABLE', True):
            with patch('infrastructure.queue_manager.Redis') as mock_redis_class:
                with patch('infrastructure.queue_manager.Queue') as mock_queue_class:
                    mock_redis_instance = MagicMock()
                    mock_redis_instance.ping.return_value = True
                    mock_redis_class.return_value = mock_redis_instance

                    mock_queue_instance = MagicMock()
                    mock_queue_instance.name = "test_queue"
                    mock_queue_instance.__len__ = MagicMock(return_value=0)
                    mock_queue_class.return_value = mock_queue_instance

                    yield {
                        'redis_class': mock_redis_class,
                        'redis_instance': mock_redis_instance,
                        'queue_class': mock_queue_class,
                        'queue_instance': mock_queue_instance
                    }

    def test_queue_manager_initialization_without_redis(self):
        """Test that QueueManager requires Redis"""
        # Temporarily set REDIS_AVAILABLE to False
        with patch('infrastructure.queue_manager.REDIS_AVAILABLE', False):
            from infrastructure.queue_manager import QueueManager

            with pytest.raises(ImportError) as exc_info:
                QueueManager()

            assert "Redis and RQ are required" in str(exc_info.value)

    def test_queue_manager_health_check(self, mock_redis):
        """Test health check method"""
        from infrastructure.queue_manager import QueueManager

        manager = QueueManager()
        health = manager.health_check()

        assert health is True
        mock_redis['redis_instance'].ping.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
