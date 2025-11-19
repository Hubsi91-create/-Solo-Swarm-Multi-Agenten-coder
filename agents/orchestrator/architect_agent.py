"""
Architect Agent - High-level planning and task orchestration
Uses Sonnet 3.5 for complex reasoning and task decomposition
"""

from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType
from core.context_manager import ContextManager
from core.claude_md_manager import CLAUDEMDManager, AgentRole


logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):
    """
    Architect Agent - Master planner and orchestrator.

    Responsibilities:
    - Parse and understand user requests
    - Analyze codebase structure using ContextManager
    - Decompose requests into atomic TaskDefinition objects
    - Validate and optimize task plans
    - Use Sonnet 3.5 for complex reasoning

    This agent does NOT execute tasks - it only plans them.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "ArchitectAgent",
        config: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[TokenTracker] = None,
        context_manager: Optional[ContextManager] = None,
        claude_md_manager: Optional[CLAUDEMDManager] = None
    ):
        """
        Initialize the Architect Agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_name: Human-readable name for the agent
            config: Optional configuration dictionary
            token_tracker: Optional TokenTracker instance
            context_manager: Optional ContextManager instance
            claude_md_manager: Optional CLAUDEMDManager instance
        """
        super().__init__(agent_id, agent_name, config)

        # Initialize components
        self.token_tracker = token_tracker or TokenTracker()
        self.context_manager = context_manager or ContextManager()
        self.claude_md_manager = claude_md_manager or CLAUDEMDManager()

        # Agent-specific configuration
        self.max_tasks_per_plan = self.config.get("max_tasks_per_plan", 50)
        self.temperature = self.config.get("temperature", 0.7)
        self.codebase_path = self.config.get("codebase_path", ".")

        logger.info(
            f"ArchitectAgent initialized: {agent_name} (ID: {agent_id}), "
            f"max_tasks={self.max_tasks_per_plan}"
        )

    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather context for planning.

        Extracts:
        - User request from task context
        - Codebase structure
        - Relevant files and signatures
        - CLAUDE.md rules

        Args:
            task: The TaskDefinition containing the user request

        Returns:
            AgentResponse with gathered context or error
        """
        logger.info(f"ArchitectAgent gathering context for planning: {task.task_id}")

        try:
            # Extract user request
            user_request = task.context.get("user_request", "")
            if not user_request:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="missing_request",
                        message="No user_request found in task context",
                        timestamp=datetime.utcnow(),
                        context={"task_id": task.task_id},
                        recoverable=False
                    )
                )

            # Extract codebase context (signatures only)
            codebase_path = task.context.get("codebase_path", self.codebase_path)
            codebase_context = self.context_manager.extract_relevant_context(
                codebase_path=codebase_path,
                task_description=user_request,
                include_bodies=False  # Only signatures for token efficiency
            )

            # Get CLAUDE.md rules for orchestrator
            rules = self.claude_md_manager.get_rules_for_role(AgentRole.ORCHESTRATOR)

            # Prepare context data
            context_data = {
                "user_request": user_request,
                "codebase_context": codebase_context,
                "codebase_path": codebase_path,
                "rules_count": len(rules),
                "context_size_chars": len(codebase_context)
            }

            logger.info(
                f"Context gathered: {context_data['context_size_chars']} chars, "
                f"{context_data['rules_count']} rules"
            )

            return AgentResponse(
                success=True,
                data=context_data,
                metadata={"agent_id": self.agent_id, "phase": "gather_context"}
            )

        except Exception as e:
            error = AgentError(
                error_type="context_gathering_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            logger.error(f"Error gathering context: {error}")
            return AgentResponse(success=False, error=error)

    def take_action(self, task: TaskDefinition, context: Dict[str, Any]) -> AgentResponse:
        """
        Phase 2: Create execution plan.

        Uses Sonnet 3.5 to:
        - Decompose user request into tasks
        - Create TaskDefinition objects
        - Establish task dependencies
        - Estimate costs and duration

        Args:
            task: The original planning TaskDefinition
            context: Context data from gather_context

        Returns:
            AgentResponse with execution plan or error
        """
        logger.info(f"ArchitectAgent creating execution plan for: {task.task_id}")

        try:
            user_request = context["user_request"]
            codebase_context = context["codebase_context"]

            # Create planning prompt with CLAUDE.md rules
            planning_prompt = self._build_planning_prompt(
                user_request=user_request,
                codebase_context=codebase_context
            )

            # Call Sonnet 3.5 for planning (simulated for now)
            # In production: plan_json = self.call_sonnet(planning_prompt)
            plan_json = self._simulate_planning(user_request, codebase_context)

            # Simulate token usage (Sonnet model)
            simulated_input_tokens = len(planning_prompt) // 4  # Rough estimate
            simulated_output_tokens = len(plan_json) // 4

            # Track usage
            cost = self.token_tracker.track_usage(
                model=ModelType.SONNET_3_5,
                input_tokens=simulated_input_tokens,
                output_tokens=simulated_output_tokens,
                metadata={
                    "task_id": task.task_id,
                    "agent_id": self.agent_id,
                    "operation": "planning"
                }
            )

            # Parse and validate plan
            plan = self._parse_and_validate_plan(plan_json)

            result = {
                "plan": plan,
                "token_usage": {
                    "input_tokens": simulated_input_tokens,
                    "output_tokens": simulated_output_tokens,
                    "cost_usd": cost
                },
                "metadata": {
                    "model": ModelType.SONNET_3_5,
                    "temperature": self.temperature,
                    "agent_id": self.agent_id
                }
            }

            logger.info(
                f"Plan created: {len(plan['tasks'])} tasks, "
                f"estimated cost: ${plan['estimated_cost']}"
            )

            return AgentResponse(
                success=True,
                data=result,
                metadata={"agent_id": self.agent_id, "phase": "take_action"}
            )

        except Exception as e:
            error = AgentError(
                error_type="planning_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=True
            )
            logger.error(f"Error creating plan: {error}")
            return AgentResponse(success=False, error=error)

    def verify_work(self, task: TaskDefinition, action_result: Dict[str, Any]) -> AgentResponse:
        """
        Phase 3: Validate execution plan.

        Checks:
        - Task count within limits
        - All tasks have valid schema
        - Dependencies are resolvable
        - Cost estimates are reasonable

        Args:
            task: The original planning TaskDefinition
            action_result: Results from take_action phase

        Returns:
            AgentResponse indicating validation success or failure
        """
        logger.info(f"ArchitectAgent validating plan for: {task.task_id}")

        try:
            plan = action_result["plan"]

            validation_results = {
                "task_count_valid": len(plan["tasks"]) <= self.max_tasks_per_plan,
                "all_tasks_valid_schema": True,  # Validated in parse
                "dependencies_resolvable": self._validate_dependencies(plan.get("task_graph", {})),
                "cost_reasonable": plan["estimated_cost"] < 100.0,  # $100 limit
                "task_count": len(plan["tasks"])
            }

            # Check if validation passed
            validation_passed = all([
                validation_results["task_count_valid"],
                validation_results["all_tasks_valid_schema"],
                validation_results["dependencies_resolvable"],
                validation_results["cost_reasonable"]
            ])

            verification_data = {
                "verification_passed": validation_passed,
                "validation_results": validation_results,
                "plan": plan,
                "token_usage": action_result.get("token_usage")
            }

            if not validation_passed:
                error = AgentError(
                    error_type="plan_validation_failed",
                    message="Plan did not pass validation checks",
                    timestamp=datetime.utcnow(),
                    context={
                        "task_id": task.task_id,
                        "validation_results": validation_results
                    },
                    recoverable=True
                )
                logger.warning(f"Plan validation failed for {task.task_id}: {error}")
                return AgentResponse(success=False, error=error, data=verification_data)

            logger.info(f"Plan validation passed for {task.task_id}")

            return AgentResponse(
                success=True,
                data=verification_data,
                metadata={"agent_id": self.agent_id, "phase": "verify_work"}
            )

        except Exception as e:
            error = AgentError(
                error_type="validation_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            logger.error(f"Error during validation: {error}")
            return AgentResponse(success=False, error=error)

    def plan_and_delegate(self, user_request: str, codebase_path: Optional[str] = None) -> Dict[str, Any]:
        """
        High-level method to plan and delegate a user request.

        This is a convenience method that wraps the full lifecycle.

        Args:
            user_request: User's natural language request
            codebase_path: Optional path to codebase (uses default if None)

        Returns:
            Dictionary with plan and execution details
        """
        # Create planning task
        planning_task = TaskDefinition(
            task_id=f"plan_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            task_type=TaskType.ANALYSIS,
            priority=1,
            assigned_agent=self.agent_id,
            context={
                "user_request": user_request,
                "codebase_path": codebase_path or self.codebase_path
            }
        )

        # Execute planning lifecycle
        result = self.execute_task(planning_task)

        if result.success:
            return result.data
        else:
            return {
                "success": False,
                "error": str(result.error) if result.error else "Unknown error"
            }

    def _build_planning_prompt(self, user_request: str, codebase_context: str) -> str:
        """Build the planning prompt for Sonnet"""

        base_prompt = f"""You are the Architect Agent in the Solo-Swarm Multi-Agent System.

Your task is to analyze the user request and create a detailed execution plan.

# User Request
{user_request}

# Codebase Context
{codebase_context}

# Instructions
1. Decompose the user request into atomic, executable tasks
2. Create a TaskDefinition for each task with proper schema
3. Assign appropriate priorities (1=highest, 10=lowest)
4. Estimate cost and duration
5. Define task dependencies

Output the plan as JSON following this structure:
{{
    "plan_summary": "Brief description",
    "estimated_duration": "Time estimate",
    "estimated_cost": "Cost in USD",
    "tasks": [...TaskDefinition objects...],
    "task_graph": {{"task_id": ["dependent_task_id", ...]}}
}}
"""

        # Inject CLAUDE.md rules
        enhanced_prompt = self.claude_md_manager.inject_into_prompt(
            base_prompt,
            AgentRole.ORCHESTRATOR
        )

        return enhanced_prompt

    def _simulate_planning(self, user_request: str, codebase_context: str) -> str:
        """
        Simulate Sonnet planning response.

        In production, this would be replaced with actual call_sonnet().

        Args:
            user_request: User's request
            codebase_context: Codebase context

        Returns:
            JSON string with simulated plan
        """
        # Create simulated tasks based on request keywords
        tasks = []

        # Simple heuristic: if request mentions "implement", create implementation task
        if "implement" in user_request.lower() or "create" in user_request.lower():
            tasks.append({
                "task_id": "impl_001",
                "task_type": TaskType.IMPLEMENTATION.value,
                "priority": 1,
                "assigned_agent": "coder_agent",
                "context": {
                    "language": "python",
                    "specifications": user_request
                },
                "requirements": {
                    "include_tests": True,
                    "max_lines": 500
                }
            })

        # If request mentions "test", add testing task
        if "test" in user_request.lower():
            tasks.append({
                "task_id": "test_001",
                "task_type": TaskType.TESTING.value,
                "priority": 2,
                "assigned_agent": "tester_agent",
                "context": {
                    "test_type": "unit",
                    "coverage_target": 80
                },
                "requirements": {
                    "timeout": 300
                }
            })

        # Default: create at least one analysis task
        if not tasks:
            tasks.append({
                "task_id": "analysis_001",
                "task_type": TaskType.ANALYSIS.value,
                "priority": 1,
                "assigned_agent": "coder_agent",
                "context": {
                    "analysis_type": "requirement_analysis",
                    "scope": user_request
                },
                "requirements": {}
            })

        plan = {
            "plan_summary": f"Plan for: {user_request[:100]}",
            "estimated_duration": "30 minutes",
            "estimated_cost": 0.05,  # $0.05 estimate
            "tasks": tasks,
            "task_graph": {
                tasks[0]["task_id"]: [t["task_id"] for t in tasks[1:]] if len(tasks) > 1 else []
            }
        }

        return json.dumps(plan, indent=2)

    def _parse_and_validate_plan(self, plan_json: str) -> Dict[str, Any]:
        """
        Parse and validate a plan JSON response.

        Args:
            plan_json: JSON string from Sonnet

        Returns:
            Validated plan dictionary

        Raises:
            ValueError: If plan is invalid
        """
        try:
            plan = json.loads(plan_json)

            # Validate required fields
            required_fields = ["plan_summary", "estimated_duration", "estimated_cost", "tasks"]
            for field in required_fields:
                if field not in plan:
                    raise ValueError(f"Missing required field: {field}")

            # Validate and convert tasks to TaskDefinition objects
            validated_tasks = []
            for task_data in plan["tasks"]:
                # Convert task_type string to TaskType enum if needed
                if isinstance(task_data.get("task_type"), str):
                    task_data["task_type"] = TaskType(task_data["task_type"])

                # Create TaskDefinition to validate schema
                task = TaskDefinition(**task_data)
                validated_tasks.append(task.to_strict_json())

            plan["tasks"] = validated_tasks

            return plan

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in plan: {e}")
        except Exception as e:
            raise ValueError(f"Plan validation failed: {e}")

    def _validate_dependencies(self, task_graph: Dict[str, List[str]]) -> bool:
        """
        Validate that task dependencies are resolvable (no cycles).

        Args:
            task_graph: Dictionary mapping task_id to list of dependent task_ids

        Returns:
            True if dependencies are valid, False if cycles detected
        """
        # Simple cycle detection using DFS
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in task_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for node in task_graph.keys():
            if node not in visited:
                if has_cycle(node, visited, set()):
                    logger.error(f"Cycle detected in task graph at node: {node}")
                    return False

        return True

    def __repr__(self) -> str:
        return (
            f"ArchitectAgent(id='{self.agent_id}', "
            f"name='{self.agent_name}', "
            f"max_tasks={self.max_tasks_per_plan})"
        )
