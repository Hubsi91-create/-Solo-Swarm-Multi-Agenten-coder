"""
Architect Agent - High-level planning and task orchestration
Uses Sonnet 3.5 for complex reasoning and task decomposition
"""

from typing import Dict, Any, Optional, List
import json
import logging
import subprocess
import tempfile
import os
from datetime import datetime
from pathlib import Path

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType
from core.context_manager import ContextManager
from core.claude_md_manager import CLAUDEMDManager, AgentRole
from core.self_correction import SelfCorrectionManager
from core.thinking_budget import ThinkingBudgetAllocator


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
        self.self_correction_manager = SelfCorrectionManager(max_retries=3)
        self.budget_allocator = ThinkingBudgetAllocator(
            config=self.config.get("budget_config", {})
        )

        # Agent-specific configuration
        self.max_tasks_per_plan = self.config.get("max_tasks_per_plan", 50)
        self.temperature = self.config.get("temperature", 0.7)
        self.codebase_path = self.config.get("codebase_path", ".")
        self.use_extended_thinking = self.config.get("use_extended_thinking", True)

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

    def plan_with_extended_thinking(
        self,
        request: str,
        planning_task: TaskDefinition
    ) -> Dict[str, Any]:
        """
        Plan with Extended Thinking - Uses dynamic budgeting for intelligent planning.

        This method implements the core extended thinking logic:
        1. Calculate task complexity based on context, dependencies, and type
        2. Allocate appropriate token budget (1k-32k tokens)
        3. Execute planning with budget-appropriate depth:
           - Budget > 10k: Detailed planning with many sub-tasks
           - Budget 5k-10k: Moderate planning with balanced decomposition
           - Budget < 5k: Lightweight planning with direct execution

        Args:
            request: User request string
            planning_task: TaskDefinition for planning

        Returns:
            Dictionary with plan, budget info, and complexity analysis
        """
        logger.info(f"Planning with extended thinking for: {request[:100]}...")

        try:
            # Step 1: Calculate complexity and allocate budget
            budget_info = self.budget_allocator.calculate_and_allocate(planning_task)
            complexity = budget_info["complexity"]
            allocated_budget = budget_info["budget_tokens"]

            logger.info(
                f"Complexity: {complexity}/10, "
                f"Allocated budget: {allocated_budget:,} tokens, "
                f"Recommendation: {budget_info['recommendation']}"
            )

            # Step 2: Create planning prompt with budget context
            context_data = planning_task.context
            codebase_context = context_data.get("codebase_context", "")

            planning_prompt = self._build_planning_prompt(
                user_request=request,
                codebase_context=codebase_context
            )

            # Add budget guidance to prompt
            budget_guidance = self._create_budget_guidance(complexity, allocated_budget)
            enhanced_prompt = f"{planning_prompt}\n\n{budget_guidance}"

            # Step 3: Execute planning with appropriate depth
            if allocated_budget > 10_000:
                # High budget: Detailed planning
                logger.info("Using detailed planning mode (budget > 10k)")
                plan_json = self._simulate_detailed_planning(
                    request, codebase_context, complexity
                )
            elif allocated_budget >= 5_000:
                # Medium budget: Moderate planning
                logger.info("Using moderate planning mode (budget 5k-10k)")
                plan_json = self._simulate_moderate_planning(
                    request, codebase_context, complexity
                )
            else:
                # Low budget: Lightweight planning
                logger.info("Using lightweight planning mode (budget < 5k)")
                plan_json = self._simulate_planning(request, codebase_context)

            # Step 4: Track token usage (simulated based on budget)
            # In production, actual token usage would be measured
            simulated_input_tokens = len(enhanced_prompt) // 4
            simulated_thinking_tokens = allocated_budget  # Extended thinking tokens
            simulated_output_tokens = len(plan_json) // 4

            cost = self.token_tracker.track_usage(
                model=ModelType.SONNET_3_5,
                input_tokens=simulated_input_tokens,
                output_tokens=simulated_output_tokens,
                metadata={
                    "task_id": planning_task.task_id,
                    "agent_id": self.agent_id,
                    "operation": "extended_thinking_planning",
                    "thinking_tokens": simulated_thinking_tokens,
                    "complexity": complexity,
                    "allocated_budget": allocated_budget
                }
            )

            # Step 5: Parse and validate plan
            plan = self._parse_and_validate_plan(plan_json)

            result = {
                "plan": plan,
                "budget_info": budget_info,
                "token_usage": {
                    "input_tokens": simulated_input_tokens,
                    "thinking_tokens": simulated_thinking_tokens,
                    "output_tokens": simulated_output_tokens,
                    "total_tokens": (
                        simulated_input_tokens +
                        simulated_thinking_tokens +
                        simulated_output_tokens
                    ),
                    "cost_usd": cost
                },
                "metadata": {
                    "model": ModelType.SONNET_3_5,
                    "extended_thinking_enabled": True,
                    "complexity": complexity,
                    "agent_id": self.agent_id
                }
            }

            logger.info(
                f"Extended thinking plan created: "
                f"{len(plan['tasks'])} tasks, "
                f"complexity {complexity}, "
                f"thinking tokens: {simulated_thinking_tokens:,}, "
                f"cost: ${cost:.4f}"
            )

            return result

        except Exception as e:
            error_msg = f"Error in extended thinking planning: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _create_budget_guidance(self, complexity: int, budget: int) -> str:
        """
        Create budget-specific guidance for the planning prompt.

        Args:
            complexity: Complexity score
            budget: Allocated token budget

        Returns:
            Guidance string to append to planning prompt
        """
        if budget > 10_000:
            return f"""
# Extended Thinking Mode (Budget: {budget:,} tokens)

You have a large thinking budget for this complex task (complexity: {complexity}/10).
Use extensive reasoning to:
- Break down the task into 5-15 atomic sub-tasks
- Create detailed dependencies and execution order
- Consider edge cases and potential issues
- Provide comprehensive cost and time estimates
"""
        elif budget >= 5_000:
            return f"""
# Moderate Thinking Mode (Budget: {budget:,} tokens)

You have a moderate thinking budget (complexity: {complexity}/10).
Focus on:
- Breaking down the task into 3-8 sub-tasks
- Establishing clear dependencies
- Providing reasonable estimates
"""
        else:
            return f"""
# Lightweight Thinking Mode (Budget: {budget:,} tokens)

You have a minimal thinking budget (complexity: {complexity}/10).
Focus on:
- Creating 1-3 direct execution tasks
- Simple, straightforward planning
- Efficient task decomposition
"""

    def _simulate_detailed_planning(
        self,
        user_request: str,
        codebase_context: str,
        complexity: int
    ) -> str:
        """
        Simulate detailed planning with many sub-tasks (high budget).

        Creates a comprehensive plan with 5-15 tasks for complex requests.

        Args:
            user_request: User's request
            codebase_context: Codebase context
            complexity: Complexity score

        Returns:
            JSON string with detailed plan
        """
        tasks = []
        base_id = datetime.utcnow().strftime('%Y%m%d%H%M%S')

        # High complexity: Create many detailed sub-tasks
        num_tasks = min(15, max(5, complexity + 5))  # 5-15 tasks

        # Analysis phase
        tasks.append({
            "task_id": f"analysis_{base_id}_001",
            "task_type": TaskType.ANALYSIS.value,
            "priority": 1,
            "assigned_agent": "architect_agent",
            "context": {
                "analysis_type": "requirement_analysis",
                "scope": user_request,
                "depth": "comprehensive"
            },
            "requirements": {"include_edge_cases": True}
        })

        # Research phase (for complex tasks)
        if "implement" in user_request.lower() or "create" in user_request.lower():
            tasks.append({
                "task_id": f"research_{base_id}_002",
                "task_type": TaskType.RESEARCH.value,
                "priority": 1,
                "assigned_agent": "researcher_agent",
                "context": {
                    "research_topic": user_request,
                    "depth": "detailed"
                },
                "requirements": {}
            })

        # Implementation tasks (multiple phases)
        impl_count = max(1, num_tasks // 3)
        for i in range(impl_count):
            tasks.append({
                "task_id": f"impl_{base_id}_{i+3:03d}",
                "task_type": TaskType.IMPLEMENTATION.value,
                "priority": 2 + (i % 3),
                "assigned_agent": "coder_agent",
                "context": {
                    "language": "python",
                    "specifications": f"{user_request} - Phase {i+1}",
                    "phase": i + 1
                },
                "requirements": {
                    "include_tests": True,
                    "max_lines": 500
                }
            })

        # Testing tasks
        test_count = max(1, impl_count // 2)
        for i in range(test_count):
            tasks.append({
                "task_id": f"test_{base_id}_{i+10:03d}",
                "task_type": TaskType.TESTING.value,
                "priority": 3,
                "assigned_agent": "tester_agent",
                "context": {
                    "test_type": "unit" if i == 0 else "integration",
                    "coverage_target": 80
                },
                "requirements": {"timeout": 300}
            })

        # Review task
        tasks.append({
            "task_id": f"review_{base_id}_020",
            "task_type": TaskType.REVIEW.value,
            "priority": 4,
            "assigned_agent": "reviewer_agent",
            "context": {
                "review_type": "comprehensive",
                "focus_areas": ["correctness", "performance", "security"]
            },
            "requirements": {}
        })

        # Documentation task
        tasks.append({
            "task_id": f"docs_{base_id}_021",
            "task_type": TaskType.DOCUMENTATION.value,
            "priority": 5,
            "assigned_agent": "documenter_agent",
            "context": {
                "doc_type": "comprehensive",
                "include_examples": True
            },
            "requirements": {}
        })

        # Build task graph with dependencies
        task_graph = {}
        for i, task in enumerate(tasks):
            if i == 0:
                task_graph[task["task_id"]] = []
            elif i < len(tasks) - 2:
                # Most tasks depend on previous task
                task_graph[task["task_id"]] = [tasks[i-1]["task_id"]]
            else:
                # Last two tasks (review, docs) depend on all implementation/testing
                deps = [t["task_id"] for t in tasks[1:i]]
                task_graph[task["task_id"]] = deps

        plan = {
            "plan_summary": f"Detailed plan for: {user_request[:100]} (Extended Thinking)",
            "estimated_duration": f"{len(tasks) * 15} minutes",
            "estimated_cost": 0.05 * len(tasks),
            "tasks": tasks,
            "task_graph": task_graph,
            "planning_mode": "detailed",
            "complexity": complexity
        }

        return json.dumps(plan, indent=2)

    def _simulate_moderate_planning(
        self,
        user_request: str,
        codebase_context: str,
        complexity: int
    ) -> str:
        """
        Simulate moderate planning with balanced task decomposition (medium budget).

        Creates a plan with 3-8 tasks for moderately complex requests.

        Args:
            user_request: User's request
            codebase_context: Codebase context
            complexity: Complexity score

        Returns:
            JSON string with moderate plan
        """
        tasks = []
        base_id = datetime.utcnow().strftime('%Y%m%d%H%M%S')

        # Moderate complexity: Create 3-8 tasks
        num_tasks = min(8, max(3, complexity + 1))

        # Analysis task
        tasks.append({
            "task_id": f"analysis_{base_id}_001",
            "task_type": TaskType.ANALYSIS.value,
            "priority": 1,
            "assigned_agent": "architect_agent",
            "context": {
                "analysis_type": "requirement_analysis",
                "scope": user_request
            },
            "requirements": {}
        })

        # Implementation tasks (1-3)
        impl_count = min(3, max(1, num_tasks - 3))
        for i in range(impl_count):
            tasks.append({
                "task_id": f"impl_{base_id}_{i+2:03d}",
                "task_type": TaskType.IMPLEMENTATION.value,
                "priority": 2,
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

        # Testing task
        tasks.append({
            "task_id": f"test_{base_id}_010",
            "task_type": TaskType.TESTING.value,
            "priority": 3,
            "assigned_agent": "tester_agent",
            "context": {
                "test_type": "unit",
                "coverage_target": 80
            },
            "requirements": {"timeout": 300}
        })

        # Review task
        tasks.append({
            "task_id": f"review_{base_id}_020",
            "task_type": TaskType.REVIEW.value,
            "priority": 4,
            "assigned_agent": "reviewer_agent",
            "context": {
                "review_type": "standard"
            },
            "requirements": {}
        })

        # Build simple task graph
        task_graph = {
            tasks[0]["task_id"]: [],
        }
        for i in range(1, len(tasks)):
            task_graph[tasks[i]["task_id"]] = [tasks[i-1]["task_id"]]

        plan = {
            "plan_summary": f"Moderate plan for: {user_request[:100]}",
            "estimated_duration": f"{len(tasks) * 10} minutes",
            "estimated_cost": 0.03 * len(tasks),
            "tasks": tasks,
            "task_graph": task_graph,
            "planning_mode": "moderate",
            "complexity": complexity
        }

        return json.dumps(plan, indent=2)

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

        This method now uses extended thinking with dynamic budgeting:
        - Calculates task complexity
        - Allocates appropriate token budget
        - Plans with budget-appropriate depth

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

        # Gather context first (needed for complexity calculation)
        context_response = self.gather_context(planning_task)

        if not context_response.success:
            return {
                "success": False,
                "error": str(context_response.error) if context_response.error else "Failed to gather context"
            }

        # Add codebase context to planning task for complexity calculation
        planning_task.context["codebase_context"] = context_response.data.get("codebase_context", "")

        # Use extended thinking if enabled
        if self.use_extended_thinking:
            try:
                result = self.plan_with_extended_thinking(user_request, planning_task)
                return {
                    "success": True,
                    "plan": result["plan"],
                    "budget_info": result["budget_info"],
                    "token_usage": result["token_usage"],
                    "metadata": result["metadata"]
                }
            except Exception as e:
                logger.error(f"Extended thinking failed, falling back to standard planning: {e}")
                # Fall through to standard planning

        # Fallback: Standard planning (without extended thinking)
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

    def integrate_and_test(
        self,
        worker_results: List[Dict[str, Any]],
        test_file_path: Optional[str] = None,
        code_file_path: Optional[str] = None
    ) -> AgentResponse:
        """
        Integrate worker results and run tests with self-correction loop.

        This method implements the end-to-end pipeline:
        1. Merge code snippets from worker results
        2. Write merged code to file
        3. Run pytest on the generated code
        4. If tests fail: Use SelfCorrectionManager to generate bugfix tasks
        5. Retry up to max_retries times
        6. Return success or failure

        Args:
            worker_results: List of worker execution results
            test_file_path: Optional path to test file (for validation)
            code_file_path: Optional path to write generated code

        Returns:
            AgentResponse with integration results
        """
        logger.info("Starting integrate_and_test pipeline...")

        try:
            # Step 1: Merge worker results
            merged_code = self._merge_worker_results(worker_results)

            if not merged_code:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="merge_failed",
                        message="Failed to merge worker results - no code generated",
                        timestamp=datetime.utcnow(),
                        context={"worker_count": len(worker_results)},
                        recoverable=False
                    )
                )

            logger.info(f"Merged code: {len(merged_code)} characters")

            # Step 2: Write code to temporary or specified file
            if code_file_path:
                target_file = code_file_path
            else:
                # Create temporary file
                temp_fd, target_file = tempfile.mkstemp(suffix=".py", prefix="generated_")
                os.close(temp_fd)

            with open(target_file, 'w') as f:
                f.write(merged_code)

            logger.info(f"Code written to: {target_file}")

            # Step 3: Run tests with self-correction loop
            max_iterations = self.self_correction_manager.max_retries + 1
            iteration = 0
            test_passed = False
            test_output = ""
            bugfix_tasks: List[TaskDefinition] = []

            while iteration < max_iterations and not test_passed:
                logger.info(f"Test iteration {iteration + 1}/{max_iterations}")

                # Run tests
                test_result = self._run_tests(target_file, test_file_path)
                test_output = test_result["output"]
                test_passed = test_result["success"]

                if test_passed:
                    logger.info(f"Tests passed on iteration {iteration + 1}")
                    break

                # Tests failed - analyze and generate bugfix tasks
                logger.warning(f"Tests failed on iteration {iteration + 1}")

                if iteration < max_iterations - 1:
                    # Generate bugfix tasks
                    new_bugfix_tasks = self.self_correction_manager.analyze_test_failures(
                        test_output=test_output,
                        original_task_id=f"integrate_{iteration}"
                    )
                    bugfix_tasks.extend(new_bugfix_tasks)

                    logger.info(f"Generated {len(new_bugfix_tasks)} bugfix tasks for iteration {iteration + 1}")

                    # In a real system, these tasks would be sent to workers
                    # For now, we simulate the correction by noting the issues
                    # In production: dispatch tasks, wait for results, merge again

                iteration += 1

            # Step 4: Prepare response
            # iteration is 0-indexed, so add 1 to get actual count
            actual_iterations = iteration + 1 if test_passed else iteration

            integration_data = {
                "success": test_passed,
                "iterations": actual_iterations,
                "code_file": target_file,
                "test_output": test_output,
                "merged_code_length": len(merged_code),
                "bugfix_tasks_generated": len(bugfix_tasks),
                "bugfix_tasks": [task.to_strict_json() for task in bugfix_tasks]
            }

            if test_file_path:
                integration_data["test_file"] = test_file_path

            if test_passed:
                logger.info(
                    f"Integration successful after {iteration} iteration(s), "
                    f"generated {len(bugfix_tasks)} bugfix tasks total"
                )
                return AgentResponse(
                    success=True,
                    data=integration_data,
                    metadata={"agent_id": self.agent_id, "phase": "integrate_and_test"}
                )
            else:
                error = AgentError(
                    error_type="tests_failed",
                    message=f"Tests failed after {max_iterations} iterations",
                    timestamp=datetime.utcnow(),
                    context={
                        "iterations": iteration,
                        "bugfix_tasks_count": len(bugfix_tasks),
                        "last_output": test_output[:500]  # First 500 chars
                    },
                    recoverable=True
                )
                logger.error(f"Integration failed: {error}")
                return AgentResponse(
                    success=False,
                    error=error,
                    data=integration_data
                )

        except Exception as e:
            error = AgentError(
                error_type="integration_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"worker_results_count": len(worker_results)},
                recoverable=False
            )
            logger.error(f"Error during integration: {error}")
            return AgentResponse(success=False, error=error)

    def _merge_worker_results(self, worker_results: List[Dict[str, Any]]) -> str:
        """
        Merge code snippets from multiple worker results.

        Args:
            worker_results: List of worker execution results

        Returns:
            Merged code as string
        """
        code_parts = []

        for idx, result in enumerate(worker_results):
            # Extract code from result
            code = result.get("code", result.get("output", ""))

            if code:
                # Add comment separator
                code_parts.append(f"# ===== Worker {idx + 1} Output =====\n")
                code_parts.append(code)
                code_parts.append("\n\n")

        merged = "".join(code_parts)
        logger.debug(f"Merged {len(worker_results)} worker results into {len(merged)} chars")

        return merged

    def _run_tests(
        self,
        code_file: str,
        test_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run pytest on generated code.

        Executes tests in isolated subprocess for safety.

        Args:
            code_file: Path to generated code file
            test_file: Optional path to test file

        Returns:
            Dictionary with 'success' (bool) and 'output' (str)
        """
        logger.info(f"Running tests on {code_file}...")

        try:
            # Determine test target
            if test_file:
                test_target = test_file
            else:
                # If no test file specified, try to run pytest on the code file
                # This assumes the code file contains test functions
                test_target = code_file

            # Build pytest command
            cmd = [
                "python", "-m", "pytest",
                test_target,
                "-v",  # Verbose
                "--tb=short",  # Short traceback
                "--no-header",  # No header
                "--color=no"  # No color codes
            ]

            # Run pytest in subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=os.path.dirname(code_file) or "."
            )

            # Combine stdout and stderr
            output = result.stdout + "\n" + result.stderr

            # pytest returns 0 on success, non-zero on failure
            success = result.returncode == 0

            logger.info(f"Test result: {'PASS' if success else 'FAIL'} (return code: {result.returncode})")

            return {
                "success": success,
                "output": output,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out")
            return {
                "success": False,
                "output": "ERROR: Test execution timed out after 30 seconds",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {
                "success": False,
                "output": f"ERROR: Failed to run tests: {str(e)}",
                "return_code": -1
            }

    def __repr__(self) -> str:
        return (
            f"ArchitectAgent(id='{self.agent_id}', "
            f"name='{self.agent_name}', "
            f"max_tasks={self.max_tasks_per_plan})"
        )
