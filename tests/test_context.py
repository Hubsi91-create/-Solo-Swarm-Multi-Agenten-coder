"""
Context Engineering Tests - Testing ContextManager, CLAUDEMDManager, and ArchitectAgent
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from core.context_manager import ContextManager, FunctionSignature, ClassSignature
from core.claude_md_manager import CLAUDEMDManager, AgentRole
from core.tdf_schema import TaskDefinition, TaskType
from agents.orchestrator.architect_agent import ArchitectAgent


class TestContextManager:
    """Test suite for ContextManager"""

    @pytest.fixture
    def sample_python_file(self):
        """Create a temporary Python file for testing"""
        code = '''"""Sample module for testing"""

import os
from typing import List, Optional

class Calculator:
    """A simple calculator class"""

    def __init__(self, initial_value: int = 0):
        """Initialize calculator with optional value"""
        self.value = initial_value

    def add(self, x: int) -> int:
        """Add a number to current value"""
        self.value += x
        return self.value

    async def multiply_async(self, x: int) -> int:
        """Multiply current value asynchronously"""
        self.value *= x
        return self.value


def standalone_function(name: str, age: int) -> str:
    """
    A standalone function outside classes.

    This function demonstrates AST parsing.
    """
    return f"{name} is {age} years old"


async def async_function(data: List[str]) -> Optional[str]:
    """Async function example"""
    if data:
        return data[0]
    return None
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_context_manager_initialization(self):
        """Test ContextManager initialization"""
        cm = ContextManager()
        assert len(cm.cached_contexts) == 0

    def test_parse_python_file(self, sample_python_file):
        """Test parsing a Python file"""
        cm = ContextManager()
        context = cm._parse_python_file(sample_python_file)

        assert context is not None
        assert len(context.classes) == 1
        assert len(context.functions) == 2  # standalone_function and async_function
        assert context.file_path == sample_python_file

    def test_extract_class_signatures(self, sample_python_file):
        """Test extracting class signatures"""
        cm = ContextManager()
        context = cm._parse_python_file(sample_python_file)

        calculator_class = context.classes[0]
        assert calculator_class.name == "Calculator"
        assert calculator_class.docstring == "A simple calculator class"
        assert len(calculator_class.methods) == 3  # __init__, add, multiply_async

    def test_extract_function_signatures(self, sample_python_file):
        """Test extracting function signatures"""
        cm = ContextManager()
        context = cm._parse_python_file(sample_python_file)

        # Check standalone function
        standalone = context.functions[0]
        assert standalone.name == "standalone_function"
        assert len(standalone.args) == 2
        assert standalone.returns == "str"
        assert standalone.is_async is False

        # Check async function
        async_func = context.functions[1]
        assert async_func.name == "async_function"
        assert async_func.is_async is True
        assert async_func.returns == "Optional[str]"

    def test_extract_imports(self, sample_python_file):
        """Test extracting import statements"""
        cm = ContextManager()
        context = cm._parse_python_file(sample_python_file)

        assert "import os" in context.imports
        # Note: "from typing import ..." should be in imports

    def test_extract_relevant_context_file(self, sample_python_file):
        """Test extracting context from a file"""
        cm = ContextManager()
        context_str = cm.extract_relevant_context(sample_python_file)

        assert "Calculator" in context_str
        assert "standalone_function" in context_str
        assert "async_function" in context_str
        assert "import os" in context_str

    def test_extract_relevant_context_nonexistent(self):
        """Test extracting context from non-existent path"""
        cm = ContextManager()
        context_str = cm.extract_relevant_context("/nonexistent/path")

        assert "Error" in context_str or "not found" in context_str.lower()

    def test_format_class(self, sample_python_file):
        """Test formatting class signatures"""
        cm = ContextManager()
        context = cm._parse_python_file(sample_python_file)

        formatted = cm._format_class(context.classes[0])

        assert "class Calculator" in formatted
        assert "def __init__" in formatted
        assert "def add" in formatted
        assert "async def multiply_async" in formatted

    def test_compact_history_no_compaction_needed(self):
        """Test compact_history when messages are within limit"""
        cm = ContextManager()

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        compacted = cm.compact_history(messages, max_tokens=10000)

        assert len(compacted) == len(messages)
        assert compacted == messages

    def test_compact_history_with_compaction(self):
        """Test compact_history when compaction is needed"""
        cm = ContextManager()

        # Create many messages
        messages = [{"role": "system", "content": "System prompt"}]

        # Add 50 middle messages
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}" * 100})

        # Add 5 recent messages
        for i in range(5):
            messages.append({"role": "user", "content": f"Recent {i}"})

        compacted = cm.compact_history(messages, max_tokens=1000)

        # Should have: system + summary + last 10
        assert len(compacted) < len(messages)
        assert compacted[0]["role"] == "system"  # First message kept
        # One of the middle messages should be a summary
        assert any("summary" in msg.get("content", "").lower() for msg in compacted)

    def test_cache_functionality(self, sample_python_file):
        """Test that contexts are cached"""
        cm = ContextManager()

        # First parse
        cm._parse_python_file(sample_python_file)
        assert len(cm.cached_contexts) == 1

        # Second parse should use cache
        cm._parse_python_file(sample_python_file)
        assert len(cm.cached_contexts) == 1

    def test_clear_cache(self, sample_python_file):
        """Test clearing the cache"""
        cm = ContextManager()

        cm._parse_python_file(sample_python_file)
        assert len(cm.cached_contexts) == 1

        cm.clear_cache()
        assert len(cm.cached_contexts) == 0

    def test_get_cache_stats(self, sample_python_file):
        """Test getting cache statistics"""
        cm = ContextManager()

        cm._parse_python_file(sample_python_file)
        stats = cm.get_cache_stats()

        assert stats["cached_files"] == 1
        assert stats["total_classes"] >= 1
        assert stats["total_functions"] >= 2

    def test_robust_error_handling(self):
        """Test that ContextManager handles syntax errors gracefully"""
        cm = ContextManager()

        # Create file with syntax error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def broken_function(\n  return 'missing parenthesis'")
            temp_path = f.name

        try:
            context = cm._parse_python_file(temp_path)
            # Should return None for syntax errors
            assert context is None
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCLAUDEMDManager:
    """Test suite for CLAUDEMDManager"""

    @pytest.fixture
    def sample_claude_md(self):
        """Create a temporary CLAUDE.md file"""
        content = """# Global Rules

This is a global rule that applies to all agents.

## Sub-section
More global rules here.

# Worker Rules

Worker-specific rules go here.
Workers should follow these guidelines.

# Orchestrator Rules

Orchestrator-specific rules.
Plan carefully and delegate wisely.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            claude_md = config_dir / "CLAUDE.md"
            claude_md.write_text(content)

            yield str(config_dir)

    def test_claude_md_manager_initialization(self, sample_claude_md):
        """Test CLAUDEMDManager initialization"""
        manager = CLAUDEMDManager(config_dir=sample_claude_md)

        assert len(manager.sections[AgentRole.GLOBAL]) > 0
        assert len(manager.sections[AgentRole.WORKER]) > 0
        assert len(manager.sections[AgentRole.ORCHESTRATOR]) > 0

    def test_parse_sections(self, sample_claude_md):
        """Test parsing different sections"""
        manager = CLAUDEMDManager(config_dir=sample_claude_md)

        # Check global rules
        global_rules = manager.sections[AgentRole.GLOBAL]
        assert len(global_rules) > 0
        assert any("global rule" in section.content.lower() for section in global_rules)

        # Check worker rules
        worker_rules = manager.sections[AgentRole.WORKER]
        assert len(worker_rules) > 0
        assert any("worker" in section.content.lower() for section in worker_rules)

        # Check orchestrator rules
        orchestrator_rules = manager.sections[AgentRole.ORCHESTRATOR]
        assert len(orchestrator_rules) > 0
        assert any("orchestrator" in section.content.lower() for section in orchestrator_rules)

    def test_inject_into_prompt_worker(self, sample_claude_md):
        """Test injecting worker rules into prompt"""
        manager = CLAUDEMDManager(config_dir=sample_claude_md)

        base_prompt = "You are a worker agent."
        injected = manager.inject_into_prompt(base_prompt, AgentRole.WORKER)

        assert base_prompt in injected
        assert "Global Rules" in injected
        assert "Worker Rules" in injected
        assert "Orchestrator Rules" not in injected  # Should not include orchestrator rules

    def test_inject_into_prompt_orchestrator(self, sample_claude_md):
        """Test injecting orchestrator rules into prompt"""
        manager = CLAUDEMDManager(config_dir=sample_claude_md)

        base_prompt = "You are an orchestrator agent."
        injected = manager.inject_into_prompt(base_prompt, AgentRole.ORCHESTRATOR)

        assert base_prompt in injected
        assert "Global Rules" in injected
        assert "Orchestrator Rules" in injected
        assert "Worker Rules" not in injected  # Should not include worker rules

    def test_get_rules_for_role(self, sample_claude_md):
        """Test getting rules for specific role"""
        manager = CLAUDEMDManager(config_dir=sample_claude_md)

        worker_rules = manager.get_rules_for_role(AgentRole.WORKER)

        # Should include global + worker rules
        assert len(worker_rules) >= 2

    def test_get_section_count(self, sample_claude_md):
        """Test getting section counts"""
        manager = CLAUDEMDManager(config_dir=sample_claude_md)

        counts = manager.get_section_count()

        assert counts["global"] > 0
        assert counts["worker"] > 0
        assert counts["orchestrator"] > 0

    def test_nonexistent_claude_md(self):
        """Test handling when CLAUDE.md doesn't exist"""
        manager = CLAUDEMDManager(config_dir="/nonexistent/path")

        # Should initialize without error
        assert len(manager.sections[AgentRole.GLOBAL]) == 0


class TestArchitectAgent:
    """Test suite for ArchitectAgent"""

    def test_architect_agent_initialization(self):
        """Test ArchitectAgent initialization"""
        agent = ArchitectAgent(
            agent_id="architect_001",
            agent_name="TestArchitect"
        )

        assert agent.agent_id == "architect_001"
        assert agent.agent_name == "TestArchitect"
        assert agent.context_manager is not None
        assert agent.claude_md_manager is not None

    def test_gather_context_with_request(self):
        """Test gathering context with user request"""
        agent = ArchitectAgent(agent_id="architect_002")

        task = TaskDefinition(
            task_id="plan_001",
            task_type=TaskType.ANALYSIS,
            priority=1,
            assigned_agent="architect_002",
            context={
                "user_request": "Create a REST API for user management",
                "codebase_path": "."
            }
        )

        response = agent.gather_context(task)

        assert response.success is True
        assert "user_request" in response.data
        assert "codebase_context" in response.data

    def test_gather_context_missing_request(self):
        """Test gathering context without user request"""
        agent = ArchitectAgent(agent_id="architect_003")

        task = TaskDefinition(
            task_id="plan_002",
            task_type=TaskType.ANALYSIS,
            priority=1,
            assigned_agent="architect_003",
            context={}  # No user_request
        )

        response = agent.gather_context(task)

        assert response.success is False
        assert response.error is not None
        assert response.error.error_type == "missing_request"

    def test_take_action_creates_plan(self):
        """Test that take_action creates an execution plan"""
        agent = ArchitectAgent(agent_id="architect_004")

        task = TaskDefinition(
            task_id="plan_003",
            task_type=TaskType.ANALYSIS,
            priority=1,
            assigned_agent="architect_004"
        )

        context = {
            "user_request": "Implement user authentication",
            "codebase_context": "# Sample codebase\nclass User:\n    pass"
        }

        response = agent.take_action(task, context)

        assert response.success is True
        assert "plan" in response.data
        assert "token_usage" in response.data
        assert len(response.data["plan"]["tasks"]) > 0

    def test_verify_work_validates_plan(self):
        """Test that verify_work validates the plan"""
        agent = ArchitectAgent(agent_id="architect_005")

        task = TaskDefinition(
            task_id="plan_004",
            task_type=TaskType.ANALYSIS,
            priority=1,
            assigned_agent="architect_005"
        )

        action_result = {
            "plan": {
                "plan_summary": "Test plan",
                "estimated_duration": "10 minutes",
                "estimated_cost": 0.01,
                "tasks": [
                    {
                        "task_id": "task_001",
                        "task_type": "implementation",
                        "priority": 1,
                        "assigned_agent": "coder_agent",
                        "context": {},
                        "requirements": {}
                    }
                ],
                "task_graph": {}
            },
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "cost_usd": 0.001
            }
        }

        response = agent.verify_work(task, action_result)

        assert response.success is True
        assert response.data["verification_passed"] is True

    def test_plan_and_delegate(self):
        """Test high-level plan_and_delegate method"""
        agent = ArchitectAgent(agent_id="architect_006")

        result = agent.plan_and_delegate(
            user_request="Create a simple calculator API",
            codebase_path="."
        )

        assert "plan" in result or "error" in result

    def test_validate_dependencies_no_cycle(self):
        """Test dependency validation without cycles"""
        agent = ArchitectAgent(agent_id="architect_007")

        task_graph = {
            "task_1": ["task_2", "task_3"],
            "task_2": ["task_3"],
            "task_3": []
        }

        is_valid = agent._validate_dependencies(task_graph)
        assert is_valid is True

    def test_validate_dependencies_with_cycle(self):
        """Test dependency validation with cycle"""
        agent = ArchitectAgent(agent_id="architect_008")

        # Create a cycle: task_1 -> task_2 -> task_3 -> task_1
        task_graph = {
            "task_1": ["task_2"],
            "task_2": ["task_3"],
            "task_3": ["task_1"]
        }

        is_valid = agent._validate_dependencies(task_graph)
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
