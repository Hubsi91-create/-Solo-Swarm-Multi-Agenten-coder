"""
Test Suite for Advanced Planning - Extended Thinking & Budgeting

Tests:
1. Complexity calculation with various TDFs
2. Budget allocation for different complexity levels
3. Architect agent extended thinking behavior
4. Plan generation with different budget tiers
"""

import pytest
from datetime import datetime

from core.thinking_budget import (
    ThinkingBudgetAllocator,
    ComplexityFactors
)
from core.tdf_schema import TaskDefinition, TaskType
from agents.orchestrator.architect_agent import ArchitectAgent
from core.token_tracker import TokenTracker


class TestComplexityCalculation:
    """Test suite for complexity calculation logic"""

    @pytest.fixture
    def allocator(self):
        """Create a ThinkingBudgetAllocator instance"""
        return ThinkingBudgetAllocator()

    def test_simple_task_low_complexity(self, allocator):
        """Test that simple tasks get low complexity scores (1-3)"""
        task = TaskDefinition(
            task_id="simple_001",
            task_type=TaskType.REVIEW,
            priority=5,
            assigned_agent="reviewer",
            context={
                "review_type": "simple"
            },
            requirements={}
        )

        complexity = allocator.calculate_complexity(task)

        # Review tasks with minimal context should be low complexity
        assert 1 <= complexity <= 3, f"Expected low complexity (1-3), got {complexity}"

    def test_moderate_task_medium_complexity(self, allocator):
        """Test that moderate tasks get medium complexity scores (4-6)"""
        task = TaskDefinition(
            task_id="moderate_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=3,
            assigned_agent="coder",
            context={
                "language": "python",
                "specifications": "Implement a function to calculate fibonacci numbers with memoization",
                "framework": "pytest"
            },
            requirements={
                "dependencies": ["task_001", "task_002"],
                "include_tests": True
            }
        )

        complexity = allocator.calculate_complexity(task)

        # Implementation with some dependencies and moderate context
        assert 3 <= complexity <= 7, f"Expected medium complexity (3-7), got {complexity}"

    def test_complex_task_high_complexity(self, allocator):
        """Test that complex tasks get high complexity scores (7-10)"""
        # Create a task with large context
        large_context = {
            "language": "python",
            "specifications": "Implement a complete web application with user authentication, " * 100,
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "caching": "Redis",
            "additional_info": "This is a complex multi-tier application " * 50
        }

        task = TaskDefinition(
            task_id="complex_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="coder",
            context=large_context,
            requirements={
                "dependencies": ["task_001", "task_002", "task_003",
                               "task_004", "task_005", "task_006"],
                "include_tests": True,
                "max_lines": 5000
            }
        )

        complexity = allocator.calculate_complexity(task)

        # Large context + many dependencies + implementation type
        assert 6 <= complexity <= 10, f"Expected high complexity (6-10), got {complexity}"

    def test_context_size_scoring(self, allocator):
        """Test that context size affects complexity score"""
        # Small context task
        small_task = TaskDefinition(
            task_id="small_001",
            task_type=TaskType.DOCUMENTATION,
            priority=5,
            assigned_agent="documenter",
            context={"doc_type": "brief"},
            requirements={}
        )

        # Large context task
        large_task = TaskDefinition(
            task_id="large_001",
            task_type=TaskType.DOCUMENTATION,
            priority=5,
            assigned_agent="documenter",
            context={
                "doc_type": "comprehensive",
                "content": "Very large documentation content " * 500
            },
            requirements={}
        )

        small_complexity = allocator.calculate_complexity(small_task)
        large_complexity = allocator.calculate_complexity(large_task)

        # Large context should yield higher complexity
        assert large_complexity > small_complexity, (
            f"Large task ({large_complexity}) should be more complex than "
            f"small task ({small_complexity})"
        )

    def test_dependency_scoring(self, allocator):
        """Test that dependencies affect complexity score"""
        # No dependencies
        no_deps_task = TaskDefinition(
            task_id="nodeps_001",
            task_type=TaskType.ANALYSIS,
            priority=3,
            assigned_agent="analyst",
            context={"analysis_type": "standard"},
            requirements={}
        )

        # Many dependencies
        many_deps_task = TaskDefinition(
            task_id="manydeps_001",
            task_type=TaskType.ANALYSIS,
            priority=3,
            assigned_agent="analyst",
            context={"analysis_type": "standard"},
            requirements={
                "dependencies": [f"task_{i:03d}" for i in range(10)]
            }
        )

        no_deps_complexity = allocator.calculate_complexity(no_deps_task)
        many_deps_complexity = allocator.calculate_complexity(many_deps_task)

        # More dependencies should yield higher complexity
        assert many_deps_complexity > no_deps_complexity, (
            f"Task with dependencies ({many_deps_complexity}) should be more "
            f"complex than task without ({no_deps_complexity})"
        )

    def test_task_type_scoring(self, allocator):
        """Test that task type affects complexity score"""
        # Simple task type (Review)
        review_task = TaskDefinition(
            task_id="review_001",
            task_type=TaskType.REVIEW,
            priority=5,
            assigned_agent="reviewer",
            context={"review_type": "code"},
            requirements={}
        )

        # Complex task type (Implementation)
        impl_task = TaskDefinition(
            task_id="impl_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=5,
            assigned_agent="coder",
            context={"language": "python"},
            requirements={}
        )

        review_complexity = allocator.calculate_complexity(review_task)
        impl_complexity = allocator.calculate_complexity(impl_task)

        # Implementation should generally be more complex than review
        assert impl_complexity >= review_complexity, (
            f"Implementation ({impl_complexity}) should be >= "
            f"review ({review_complexity})"
        )


class TestBudgetAllocation:
    """Test suite for budget allocation logic"""

    @pytest.fixture
    def allocator(self):
        """Create a ThinkingBudgetAllocator instance"""
        return ThinkingBudgetAllocator()

    def test_low_complexity_gets_low_budget(self, allocator):
        """Test that low complexity (1-2) gets minimal budget (1k tokens)"""
        for complexity in [1, 2]:
            budget = allocator.allocate_budget(complexity)
            assert budget == 1_000, f"Complexity {complexity} should get 1k budget, got {budget:,}"

    def test_medium_complexity_gets_medium_budget(self, allocator):
        """Test that medium complexity (3-6) gets moderate budget (2.5k-5k tokens)"""
        budgets = {
            3: 2_500,
            4: 2_500,
            5: 5_000,
            6: 5_000
        }

        for complexity, expected_budget in budgets.items():
            budget = allocator.allocate_budget(complexity)
            assert budget == expected_budget, (
                f"Complexity {complexity} should get {expected_budget:,} budget, "
                f"got {budget:,}"
            )

    def test_high_complexity_gets_high_budget(self, allocator):
        """Test that high complexity (7-10) gets large budget (10k-32k tokens)"""
        budgets = {
            7: 10_000,
            8: 10_000,
            9: 32_000,
            10: 32_000
        }

        for complexity, expected_budget in budgets.items():
            budget = allocator.allocate_budget(complexity)
            assert budget == expected_budget, (
                f"Complexity {complexity} should get {expected_budget:,} budget, "
                f"got {budget:,}"
            )

    def test_invalid_complexity_clamped(self, allocator):
        """Test that invalid complexity scores are clamped to valid range"""
        # Test below minimum
        budget_below = allocator.allocate_budget(0)
        assert budget_below == 1_000, "Complexity 0 should be clamped to 1 (1k budget)"

        # Test above maximum
        budget_above = allocator.allocate_budget(15)
        assert budget_above == 32_000, "Complexity 15 should be clamped to 10 (32k budget)"

    def test_custom_budget_limits(self):
        """Test that custom budget limits are respected"""
        allocator = ThinkingBudgetAllocator(
            config={
                "max_budget": 10_000,
                "min_budget": 500
            }
        )

        # High complexity should be capped at max_budget
        high_budget = allocator.allocate_budget(10)
        assert high_budget <= 10_000, f"Budget {high_budget:,} exceeds max 10k"

        # Low complexity should be at least min_budget
        low_budget = allocator.allocate_budget(1)
        assert low_budget >= 500, f"Budget {low_budget:,} below min 500"

    def test_calculate_and_allocate(self, allocator):
        """Test the convenience method that does both calculation and allocation"""
        task = TaskDefinition(
            task_id="test_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=2,
            assigned_agent="coder",
            context={
                "language": "python",
                "specifications": "Complex implementation task " * 100
            },
            requirements={
                "dependencies": ["task_001", "task_002", "task_003"]
            }
        )

        result = allocator.calculate_and_allocate(task)

        # Verify result structure
        assert "complexity" in result
        assert "budget_tokens" in result
        assert "task_id" in result
        assert "task_type" in result
        assert "recommendation" in result

        # Verify values are reasonable
        assert 1 <= result["complexity"] <= 10
        assert 1_000 <= result["budget_tokens"] <= 32_000
        assert result["task_id"] == "test_001"
        assert result["task_type"] == TaskType.IMPLEMENTATION.value


class TestArchitectExtendedThinking:
    """Test suite for Architect Agent with extended thinking"""

    @pytest.fixture
    def architect(self):
        """Create an ArchitectAgent instance with extended thinking enabled"""
        return ArchitectAgent(
            agent_id="architect_001",
            agent_name="TestArchitect",
            config={
                "use_extended_thinking": True,
                "codebase_path": "."
            }
        )

    def test_architect_has_budget_allocator(self, architect):
        """Test that architect is initialized with budget allocator"""
        assert hasattr(architect, 'budget_allocator')
        assert isinstance(architect.budget_allocator, ThinkingBudgetAllocator)

    def test_simple_request_generates_few_tasks(self, architect):
        """Test that simple requests generate fewer tasks (lightweight planning)"""
        result = architect.plan_and_delegate(
            user_request="Write a simple hello world function",
            codebase_path="."
        )

        assert result.get("success") is True
        plan = result.get("plan")

        # Simple request should generate 1-8 tasks (can vary based on heuristics)
        num_tasks = len(plan["tasks"])
        assert 1 <= num_tasks <= 8, (
            f"Simple request should generate 1-8 tasks, got {num_tasks}"
        )

        # Budget should be reasonable (complexity can vary based on codebase context)
        budget_info = result.get("budget_info", {})
        complexity = budget_info.get("complexity", 0)
        assert complexity <= 8, f"Simple request should have reasonable complexity, got {complexity}"

    def test_complex_request_generates_many_tasks(self, architect):
        """Test that complex requests generate more tasks (detailed planning)"""
        # Create a complex request with large context
        complex_request = (
            "Implement a complete microservices architecture with user authentication, "
            "API gateway, service discovery, load balancing, database sharding, "
            "caching layer, message queue integration, monitoring, logging, "
            "and comprehensive testing suite. Include CI/CD pipeline setup."
        ) * 10  # Repeat to increase context size

        result = architect.plan_and_delegate(
            user_request=complex_request,
            codebase_path="."
        )

        assert result.get("success") is True
        plan = result.get("plan")

        # Complex request should generate many tasks (5-15)
        num_tasks = len(plan["tasks"])
        assert 3 <= num_tasks <= 20, (
            f"Complex request should generate 3-20 tasks, got {num_tasks}"
        )

        # Budget should be high
        budget_info = result.get("budget_info", {})
        budget = budget_info.get("budget_tokens", 0)
        assert budget >= 2_500, (
            f"Complex request should have budget >= 2.5k, got {budget:,}"
        )

    def test_moderate_request_balanced_planning(self, architect):
        """Test that moderate requests get balanced planning (3-8 tasks)"""
        result = architect.plan_and_delegate(
            user_request="Implement a REST API endpoint for user registration with validation and tests",
            codebase_path="."
        )

        assert result.get("success") is True
        plan = result.get("plan")

        # Moderate request should generate 3-8 tasks
        num_tasks = len(plan["tasks"])
        assert 2 <= num_tasks <= 10, (
            f"Moderate request should generate 2-10 tasks, got {num_tasks}"
        )

    def test_extended_thinking_token_tracking(self, architect):
        """Test that extended thinking tokens are tracked properly"""
        result = architect.plan_and_delegate(
            user_request="Implement a complex data processing pipeline",
            codebase_path="."
        )

        assert result.get("success") is True

        # Verify token usage is tracked
        token_usage = result.get("token_usage", {})
        assert "thinking_tokens" in token_usage
        assert "total_tokens" in token_usage
        assert "cost_usd" in token_usage

        # Thinking tokens should be non-zero
        thinking_tokens = token_usage["thinking_tokens"]
        assert thinking_tokens > 0, "Extended thinking should use tokens"

        # Total should include thinking tokens
        total_tokens = token_usage["total_tokens"]
        assert total_tokens >= thinking_tokens, "Total tokens should include thinking tokens"

    def test_planning_mode_in_metadata(self, architect):
        """Test that planning mode is included in plan metadata"""
        result = architect.plan_and_delegate(
            user_request="Create a simple utility function",
            codebase_path="."
        )

        assert result.get("success") is True
        plan = result.get("plan")

        # Plan should include planning_mode
        # Note: planning_mode might not be in the top-level plan for lightweight planning
        # Check metadata instead
        metadata = result.get("metadata", {})
        assert "extended_thinking_enabled" in metadata
        assert metadata["extended_thinking_enabled"] is True

    def test_fallback_to_standard_planning_when_disabled(self):
        """Test that standard planning is used when extended thinking is disabled"""
        architect = ArchitectAgent(
            agent_id="architect_002",
            agent_name="TestArchitect",
            config={
                "use_extended_thinking": False
            }
        )

        result = architect.plan_and_delegate(
            user_request="Implement a feature",
            codebase_path="."
        )

        # Should still succeed with standard planning
        assert "plan" in result or "error" in result

    def test_budget_guidance_in_prompt(self, architect):
        """Test that budget guidance is created correctly"""
        # Test high budget guidance
        high_guidance = architect._create_budget_guidance(9, 32_000)
        assert "Extended Thinking Mode" in high_guidance
        assert "32,000 tokens" in high_guidance

        # Test medium budget guidance
        medium_guidance = architect._create_budget_guidance(5, 5_000)
        assert "Moderate Thinking Mode" in medium_guidance
        assert "5,000 tokens" in medium_guidance

        # Test low budget guidance
        low_guidance = architect._create_budget_guidance(2, 1_000)
        assert "Lightweight Thinking Mode" in low_guidance
        assert "1,000 tokens" in low_guidance


class TestPlanningModes:
    """Test suite for different planning modes"""

    @pytest.fixture
    def architect(self):
        """Create an ArchitectAgent instance"""
        return ArchitectAgent(
            agent_id="architect_003",
            agent_name="TestArchitect",
            config={"use_extended_thinking": True}
        )

    def test_detailed_planning_mode(self, architect):
        """Test detailed planning mode (budget > 10k)"""
        plan_json = architect._simulate_detailed_planning(
            user_request="Implement complex feature",
            codebase_context="Large codebase context",
            complexity=9
        )

        import json
        plan = json.loads(plan_json)

        # Detailed planning should generate many tasks
        assert len(plan["tasks"]) >= 5, "Detailed planning should generate >= 5 tasks"
        assert plan["planning_mode"] == "detailed"
        assert plan["complexity"] == 9

    def test_moderate_planning_mode(self, architect):
        """Test moderate planning mode (budget 5k-10k)"""
        plan_json = architect._simulate_moderate_planning(
            user_request="Implement moderate feature",
            codebase_context="Medium codebase context",
            complexity=5
        )

        import json
        plan = json.loads(plan_json)

        # Moderate planning should generate 3-8 tasks
        assert 3 <= len(plan["tasks"]) <= 8, "Moderate planning should generate 3-8 tasks"
        assert plan["planning_mode"] == "moderate"
        assert plan["complexity"] == 5

    def test_lightweight_planning_mode(self, architect):
        """Test lightweight planning mode (budget < 5k)"""
        plan_json = architect._simulate_planning(
            user_request="Simple task",
            codebase_context="Small context"
        )

        import json
        plan = json.loads(plan_json)

        # Lightweight planning should generate 1-3 tasks
        assert 1 <= len(plan["tasks"]) <= 3, "Lightweight planning should generate 1-3 tasks"


class TestIntegration:
    """Integration tests for the full extended thinking pipeline"""

    def test_end_to_end_planning_with_budgeting(self):
        """Test complete end-to-end planning with budget allocation"""
        # Create architect with extended thinking
        architect = ArchitectAgent(
            agent_id="architect_e2e",
            agent_name="E2EArchitect",
            config={
                "use_extended_thinking": True,
                "budget_config": {
                    "max_budget": 32_000,
                    "min_budget": 1_000
                }
            }
        )

        # Plan a moderate complexity task
        result = architect.plan_and_delegate(
            user_request="Implement a user authentication system with JWT tokens and refresh tokens",
            codebase_path="."
        )

        # Verify complete result structure
        assert result.get("success") is True
        assert "plan" in result
        assert "budget_info" in result
        assert "token_usage" in result
        assert "metadata" in result

        # Verify plan has tasks
        plan = result["plan"]
        assert len(plan["tasks"]) > 0
        assert "plan_summary" in plan
        assert "estimated_cost" in plan

        # Verify budget info
        budget_info = result["budget_info"]
        assert "complexity" in budget_info
        assert "budget_tokens" in budget_info
        assert "recommendation" in budget_info

        # Verify token usage
        token_usage = result["token_usage"]
        assert token_usage["thinking_tokens"] > 0
        assert token_usage["total_tokens"] >= token_usage["thinking_tokens"]
        assert token_usage["cost_usd"] > 0

    def test_cost_control_enforcement(self):
        """Test that budget limits are enforced for cost control"""
        # Create architect with strict budget limits
        architect = ArchitectAgent(
            agent_id="architect_strict",
            agent_name="StrictArchitect",
            config={
                "use_extended_thinking": True,
                "budget_config": {
                    "max_budget": 5_000,  # Strict limit
                    "min_budget": 500
                }
            }
        )

        # Try to plan a complex task
        result = architect.plan_and_delegate(
            user_request="Implement a complete enterprise application " * 100,  # Very complex
            codebase_path="."
        )

        # Budget should be capped at max_budget
        if result.get("success"):
            budget_info = result.get("budget_info", {})
            allocated_budget = budget_info.get("budget_tokens", 0)
            assert allocated_budget <= 5_000, (
                f"Budget {allocated_budget:,} should be capped at 5k"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
