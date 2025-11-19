"""
Meta-Agent - Self-Optimization Through Performance Analysis

The Meta-Agent represents the pinnacle of system autonomy: an agent that
optimizes other agents by analyzing their performance and improving their
system prompts through empirical data and shadow testing.

This is the "brain" that makes the swarm smarter over time.

Key Capabilities:
- Performance monitoring across all agents
- Failure pattern analysis
- System prompt optimization
- A/B testing through shadow runs
- Autonomous deployment of improvements

Uses Claude Sonnet for deep reasoning and Extended Thinking for complex analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType
from core.thinking_budget import ThinkingBudgetAllocator


logger = logging.getLogger(__name__)


class AgentPerformanceMetric(str, Enum):
    """Performance metrics for agent evaluation"""
    SUCCESS_RATE = "success_rate"
    AVERAGE_DURATION = "average_duration"
    RETRY_COUNT = "retry_count"
    ERROR_RATE = "error_rate"
    COST_EFFICIENCY = "cost_efficiency"


class PromptImprovementStrategy(str, Enum):
    """Strategies for prompt optimization"""
    CLARITY = "clarity"  # Make instructions clearer
    EXAMPLES = "examples"  # Add more examples
    CONSTRAINTS = "constraints"  # Add better guardrails
    REASONING = "reasoning"  # Encourage better thinking
    SPECIFICITY = "specificity"  # Be more specific


class MetaAgent(BaseAgent):
    """
    Meta-Agent: The agent that optimizes other agents.

    This is the highest-level agent in the system, responsible for
    continuous improvement through performance analysis and prompt optimization.

    Workflow:
    1. Monitor agent performance metrics
    2. Identify underperforming agents
    3. Analyze failure patterns
    4. Generate improved prompt variants
    5. Shadow test new prompts
    6. Deploy successful improvements

    Uses:
    - Claude Sonnet for deep reasoning
    - Extended Thinking for complex analysis
    - Shadow Testing for safe experimentation
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "MetaAgent",
        config: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[TokenTracker] = None,
        db_session: Optional[Any] = None
    ):
        """
        Initialize the Meta-Agent.

        Args:
            agent_id: Unique identifier for this meta-agent
            agent_name: Human-readable name
            config: Configuration dictionary
            token_tracker: Token usage tracker
            db_session: Database session for metrics queries
        """
        super().__init__(agent_id, agent_name, config)

        self.token_tracker = token_tracker or TokenTracker()
        self.db_session = db_session
        self.thinking_allocator = ThinkingBudgetAllocator()

        # Meta-agent configuration
        self.model = ModelType.SONNET_3_5  # Needs high intelligence
        self.min_sample_size = self.config.get("min_sample_size", 20)  # Min tasks before optimization
        self.success_threshold = self.config.get("success_threshold", 0.80)  # 80% success rate
        self.improvement_target = self.config.get("improvement_target", 0.10)  # +10% target
        self.shadow_test_count = self.config.get("shadow_test_count", 10)  # Test on 10 historical tasks

        # Performance tracking
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
        self.optimization_history: List[Dict[str, Any]] = []

        logger.info(
            f"MetaAgent initialized: {agent_name} (ID: {agent_id}), "
            f"model: {self.model}, success threshold: {self.success_threshold}"
        )

    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather performance metrics for agent optimization.

        Queries the database for task execution history and calculates
        performance metrics for each agent type.

        Args:
            task: Task definition containing target agent type

        Returns:
            AgentResponse with performance metrics
        """
        logger.info(f"MetaAgent gathering performance metrics for task: {task.task_id}")

        try:
            # Get target agent from task context
            target_agent = task.context.get("target_agent_type", "all")

            # Get time window for analysis
            days_back = task.context.get("analysis_days", 7)

            # Query performance metrics
            metrics = self._query_agent_performance(target_agent, days_back)

            # Identify underperforming agents
            underperforming = self._identify_underperformers(metrics)

            return AgentResponse(
                success=True,
                data={
                    "agent_metrics": metrics,
                    "underperforming_agents": underperforming,
                    "analysis_period_days": days_back,
                    "timestamp": datetime.utcnow().isoformat()
                },
                metadata={
                    "agent_id": self.agent_id,
                    "task_id": task.task_id
                }
            )

        except Exception as e:
            logger.error(f"Error gathering performance metrics: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=AgentError(
                    error_type="metrics_error",
                    message=str(e),
                    timestamp=datetime.utcnow(),
                    context={"task_id": task.task_id},
                    recoverable=True
                )
            )

    def take_action(self, task: TaskDefinition, context: Dict[str, Any]) -> AgentResponse:
        """
        Phase 2: Generate optimized prompt variants for underperforming agents.

        Uses Extended Thinking to deeply analyze failure patterns and create
        improved system prompts through empirical reasoning.

        Args:
            task: Original task definition
            context: Context from gather_context phase

        Returns:
            AgentResponse with generated prompt variants
        """
        logger.info(f"MetaAgent generating prompt optimizations for task: {task.task_id}")

        try:
            underperforming = context.get("underperforming_agents", [])

            if not underperforming:
                logger.info("No underperforming agents found - system is healthy!")
                return AgentResponse(
                    success=True,
                    data={
                        "optimization_needed": False,
                        "message": "All agents meeting performance targets"
                    },
                    metadata={"agent_id": self.agent_id, "task_id": task.task_id}
                )

            # Generate optimizations for each underperforming agent
            optimizations = []

            for agent_info in underperforming:
                agent_type = agent_info["agent_type"]
                current_success_rate = agent_info["success_rate"]

                logger.info(
                    f"🧠 Optimizing {agent_type} "
                    f"(current success rate: {current_success_rate:.1%})"
                )

                # Analyze failure patterns
                failures = self._analyze_failure_patterns(agent_type)

                # Allocate thinking budget based on complexity
                thinking_budget = self._allocate_thinking_budget(agent_type, failures)

                # Generate improved prompts (THIS IS WHERE THE MAGIC HAPPENS)
                prompt_variants = self._generate_prompt_variants(
                    agent_type=agent_type,
                    failures=failures,
                    thinking_budget=thinking_budget
                )

                optimizations.append({
                    "agent_type": agent_type,
                    "current_success_rate": current_success_rate,
                    "target_success_rate": current_success_rate + self.improvement_target,
                    "failure_patterns": failures,
                    "prompt_variants": prompt_variants,
                    "thinking_budget_used": thinking_budget,
                    "generated_at": datetime.utcnow().isoformat()
                })

            return AgentResponse(
                success=True,
                data={
                    "optimizations": optimizations,
                    "agent_count": len(optimizations)
                },
                metadata={"agent_id": self.agent_id, "task_id": task.task_id}
            )

        except Exception as e:
            logger.error(f"Error generating optimizations: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=AgentError(
                    error_type="optimization_error",
                    message=str(e),
                    timestamp=datetime.utcnow(),
                    context={"task_id": task.task_id},
                    recoverable=True
                )
            )

    def verify_work(self, task: TaskDefinition, result: Any) -> AgentResponse:
        """
        Phase 3: Validate optimizations through shadow testing.

        Tests new prompt variants against historical tasks in a sandbox
        to ensure they actually improve performance before deployment.

        Args:
            task: Original task definition
            result: Results from take_action phase

        Returns:
            AgentResponse with validation results
        """
        logger.info(f"MetaAgent validating optimizations for task: {task.task_id}")

        try:
            optimizations = result.get("optimizations", [])

            if not optimizations:
                return AgentResponse(
                    success=True,
                    data={"validated": True, "no_optimizations_needed": True},
                    metadata={"agent_id": self.agent_id, "task_id": task.task_id}
                )

            validated_optimizations = []

            for opt in optimizations:
                agent_type = opt["agent_type"]
                prompt_variants = opt["prompt_variants"]

                logger.info(
                    f"🧪 Shadow testing {len(prompt_variants)} prompt variants "
                    f"for {agent_type}..."
                )

                # Shadow test each variant
                # (Integration with ShadowTester will be in next step)
                test_results = self._shadow_test_prompts(agent_type, prompt_variants)

                # Select best performing variant
                best_variant = self._select_best_variant(test_results)

                validated_optimizations.append({
                    "agent_type": agent_type,
                    "original_success_rate": opt["current_success_rate"],
                    "target_success_rate": opt["target_success_rate"],
                    "best_variant": best_variant,
                    "test_results": test_results,
                    "ready_for_deployment": best_variant is not None
                })

            return AgentResponse(
                success=True,
                data={
                    "validated_optimizations": validated_optimizations,
                    "deployment_ready_count": sum(
                        1 for o in validated_optimizations
                        if o["ready_for_deployment"]
                    )
                },
                metadata={"agent_id": self.agent_id, "task_id": task.task_id}
            )

        except Exception as e:
            logger.error(f"Error validating optimizations: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=AgentError(
                    error_type="validation_error",
                    message=str(e),
                    timestamp=datetime.utcnow(),
                    context={"task_id": task.task_id},
                    recoverable=True
                )
            )

    # ==================== Helper Methods ====================

    def _query_agent_performance(
        self,
        agent_type: str,
        days_back: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Query database for agent performance metrics.

        Args:
            agent_type: Type of agent to query ("all" for all agents)
            days_back: Number of days to look back

        Returns:
            Dictionary mapping agent types to their metrics
        """
        # Simulate database query (would use self.db_session in production)
        # Returns aggregated metrics per agent type

        metrics = {
            "CoderAgent": {
                "total_tasks": 156,
                "successful_tasks": 142,
                "failed_tasks": 14,
                "success_rate": 0.91,  # 91%
                "avg_duration_seconds": 45.3,
                "avg_retries": 0.2,
                "error_rate": 0.09,
                "cost_per_task_usd": 0.025
            },
            "AssetAgent": {
                "total_tasks": 89,
                "successful_tasks": 67,
                "failed_tasks": 22,
                "success_rate": 0.75,  # 75% - UNDERPERFORMING!
                "avg_duration_seconds": 78.5,
                "avg_retries": 0.5,
                "error_rate": 0.25,
                "cost_per_task_usd": 0.045
            },
            "VerifierAgent": {
                "total_tasks": 203,
                "successful_tasks": 195,
                "failed_tasks": 8,
                "success_rate": 0.96,  # 96% - Excellent!
                "avg_duration_seconds": 12.1,
                "avg_retries": 0.1,
                "error_rate": 0.04,
                "cost_per_task_usd": 0.008
            },
            "QAAgent": {
                "total_tasks": 45,
                "successful_tasks": 38,
                "failed_tasks": 7,
                "success_rate": 0.84,  # 84%
                "avg_duration_seconds": 23.7,
                "avg_retries": 0.3,
                "error_rate": 0.16,
                "cost_per_task_usd": 0.012
            }
        }

        if agent_type != "all" and agent_type in metrics:
            return {agent_type: metrics[agent_type]}

        return metrics

    def _identify_underperformers(
        self,
        metrics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify agents that are underperforming.

        Args:
            metrics: Agent performance metrics

        Returns:
            List of underperforming agents with their metrics
        """
        underperforming = []

        for agent_type, agent_metrics in metrics.items():
            total_tasks = agent_metrics["total_tasks"]
            success_rate = agent_metrics["success_rate"]

            # Check if agent has enough data and is underperforming
            if total_tasks >= self.min_sample_size and success_rate < self.success_threshold:
                underperforming.append({
                    "agent_type": agent_type,
                    "success_rate": success_rate,
                    "total_tasks": total_tasks,
                    "failed_tasks": agent_metrics["failed_tasks"],
                    "error_rate": agent_metrics["error_rate"],
                    "gap_to_target": self.success_threshold - success_rate
                })

                logger.warning(
                    f"📉 {agent_type} underperforming: "
                    f"{success_rate:.1%} success rate "
                    f"(target: {self.success_threshold:.1%})"
                )

        return underperforming

    def _analyze_failure_patterns(self, agent_type: str) -> Dict[str, Any]:
        """
        Analyze failure patterns for an agent type.

        In production, this would query failed tasks from the database
        and use LLM analysis to identify common patterns.

        Args:
            agent_type: Type of agent to analyze

        Returns:
            Dictionary with failure pattern analysis
        """
        # Simulate failure pattern analysis
        # In production: query DB for failed tasks, analyze with LLM

        patterns = {
            "AssetAgent": {
                "common_errors": [
                    "API timeout when fetching external assets",
                    "Invalid file format detection",
                    "Missing validation constraints"
                ],
                "failure_categories": {
                    "timeout": 0.40,  # 40% of failures
                    "validation": 0.35,  # 35% of failures
                    "api_error": 0.25  # 25% of failures
                },
                "suggested_improvements": [
                    "Add retry logic with exponential backoff",
                    "Improve file format detection heuristics",
                    "Add explicit validation examples in prompt"
                ]
            },
            "CoderAgent": {
                "common_errors": [
                    "Syntax errors in generated code",
                    "Missing import statements"
                ],
                "failure_categories": {
                    "syntax": 0.60,
                    "imports": 0.40
                },
                "suggested_improvements": [
                    "Add more code examples in prompt",
                    "Emphasize import management"
                ]
            }
        }

        return patterns.get(agent_type, {
            "common_errors": [],
            "failure_categories": {},
            "suggested_improvements": []
        })

    def _allocate_thinking_budget(
        self,
        agent_type: str,
        failures: Dict[str, Any]
    ) -> int:
        """
        Allocate Extended Thinking budget for prompt optimization.

        More complex failure patterns get more thinking tokens.

        Args:
            agent_type: Type of agent being optimized
            failures: Failure pattern analysis

        Returns:
            Thinking token budget (1000-32000)
        """
        # Base complexity on number of failure categories
        failure_categories = len(failures.get("failure_categories", {}))

        # Complexity score: 0-10
        complexity_score = min(10, failure_categories + 3)  # Prompt opt is complex

        budget = self.thinking_allocator.allocate_budget(complexity=complexity_score)

        logger.info(
            f"Allocated {budget} thinking tokens for {agent_type} optimization "
            f"(complexity: {complexity_score}/10)"
        )

        return budget

    def _generate_prompt_variants(
        self,
        agent_type: str,
        failures: Dict[str, Any],
        thinking_budget: int
    ) -> List[Dict[str, Any]]:
        """
        Generate improved system prompt variants.

        THIS IS THE CORE INTELLIGENCE: Use LLM with Extended Thinking to
        analyze failures and create better prompts.

        In production, this would call Claude Sonnet with extended thinking.
        For now, we'll generate plausible variants based on failure patterns.

        Args:
            agent_type: Type of agent to optimize
            failures: Failure pattern analysis
            thinking_budget: Allocated thinking tokens

        Returns:
            List of prompt variant dictionaries
        """
        # In production: Call Claude Sonnet with extended thinking
        # Prompt: "Given these failure patterns, generate 3 improved system prompts..."

        suggested_improvements = failures.get("suggested_improvements", [])

        # Generate 3 variants with different strategies
        variants = []

        # Variant 1: Clarity-focused (clearer instructions)
        variants.append({
            "variant_id": 1,
            "strategy": PromptImprovementStrategy.CLARITY,
            "prompt_additions": [
                "Be explicit about error handling requirements",
                "Clarify expected output format with examples",
                "Define success criteria upfront"
            ],
            "changes_from_baseline": suggested_improvements[:1] if suggested_improvements else [],
            "estimated_improvement": 0.12  # +12%
        })

        # Variant 2: Examples-focused (more examples)
        variants.append({
            "variant_id": 2,
            "strategy": PromptImprovementStrategy.EXAMPLES,
            "prompt_additions": [
                "Add 3 success examples",
                "Add 2 failure examples with corrections",
                "Include edge case handling examples"
            ],
            "changes_from_baseline": suggested_improvements[:2] if len(suggested_improvements) >= 2 else [],
            "estimated_improvement": 0.15  # +15%
        })

        # Variant 3: Constraints-focused (better guardrails)
        variants.append({
            "variant_id": 3,
            "strategy": PromptImprovementStrategy.CONSTRAINTS,
            "prompt_additions": [
                "Add explicit timeout handling requirements",
                "Define validation checkpoints",
                "Specify retry strategies"
            ],
            "changes_from_baseline": suggested_improvements,
            "estimated_improvement": 0.18  # +18%
        })

        logger.info(f"Generated {len(variants)} prompt variants for {agent_type}")

        return variants

    def _shadow_test_prompts(
        self,
        agent_type: str,
        prompt_variants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Shadow test prompt variants against historical tasks.

        This would integrate with ShadowTester to run tests in isolation.
        For now, we simulate the results.

        Args:
            agent_type: Agent type being tested
            prompt_variants: Prompt variants to test

        Returns:
            List of test result dictionaries
        """
        # In production: Use ShadowTester to run variants on historical tasks
        # For now: Simulate based on estimated improvement

        results = []

        for variant in prompt_variants:
            variant_id = variant["variant_id"]
            estimated_improvement = variant["estimated_improvement"]

            # Simulate testing on N historical tasks
            simulated_success_rate = min(1.0, 0.75 + estimated_improvement)  # AssetAgent baseline: 75%

            results.append({
                "variant_id": variant_id,
                "strategy": variant["strategy"],
                "tasks_tested": self.shadow_test_count,
                "success_rate": simulated_success_rate,
                "improvement_over_baseline": simulated_improvement,
                "confidence": 0.85 if simulated_success_rate >= 0.85 else 0.70,
                "tested_at": datetime.utcnow().isoformat()
            })

        return results

    def _select_best_variant(
        self,
        test_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best performing prompt variant.

        Args:
            test_results: Shadow test results

        Returns:
            Best variant dict or None if no improvement
        """
        if not test_results:
            return None

        # Sort by success rate
        sorted_results = sorted(
            test_results,
            key=lambda r: r["success_rate"],
            reverse=True
        )

        best = sorted_results[0]

        # Only select if it actually improves
        if best["success_rate"] > 0.80:  # Better than threshold
            logger.info(
                f"✅ Best variant: {best['strategy']} "
                f"(success rate: {best['success_rate']:.1%})"
            )
            return best

        logger.warning("No variant achieved sufficient improvement")
        return None

    def monitor_performance(self) -> Dict[str, Any]:
        """
        High-level method to monitor system-wide performance.

        Returns:
            Performance summary for all agents
        """
        logger.info("🔍 MetaAgent monitoring system-wide performance...")

        # Query all agent metrics
        metrics = self._query_agent_performance("all", days_back=7)

        # Identify underperformers
        underperforming = self._identify_underperformers(metrics)

        # Calculate system health
        avg_success_rate = sum(
            m["success_rate"] for m in metrics.values()
        ) / len(metrics) if metrics else 0

        return {
            "system_health": "healthy" if avg_success_rate >= self.success_threshold else "needs_optimization",
            "average_success_rate": avg_success_rate,
            "total_agents_monitored": len(metrics),
            "underperforming_agents": len(underperforming),
            "metrics_by_agent": metrics,
            "underperforming_details": underperforming,
            "monitoring_timestamp": datetime.utcnow().isoformat()
        }

    def optimize_prompts(self, agent_type: str) -> Dict[str, Any]:
        """
        High-level method to optimize prompts for a specific agent.

        This orchestrates the full optimization cycle:
        1. Analyze performance
        2. Generate variants
        3. Shadow test
        4. Select best
        5. (Ready for deployment)

        Args:
            agent_type: Type of agent to optimize

        Returns:
            Optimization results
        """
        logger.info(f"🚀 Starting prompt optimization for {agent_type}...")

        # Create task for optimization
        task = TaskDefinition(
            task_id=f"meta_optimize_{agent_type}_{int(datetime.utcnow().timestamp())}",
            task_type=TaskType.ANALYSIS,
            priority=3,
            assigned_agent=self.agent_id,
            context={
                "target_agent_type": agent_type,
                "analysis_days": 7,
                "optimization_goal": "improve_success_rate"
            },
            requirements={}
        )

        # Phase 1: Gather metrics
        context_response = self.gather_context(task)
        if not context_response.success:
            return {"error": str(context_response.error)}

        # Phase 2: Generate optimizations
        action_response = self.take_action(task, context_response.data)
        if not action_response.success:
            return {"error": str(action_response.error)}

        # Phase 3: Validate through shadow testing
        verify_response = self.verify_work(task, action_response.data)
        if not verify_response.success:
            return {"error": str(action_response.error)}

        return verify_response.data
