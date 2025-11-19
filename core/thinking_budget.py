"""
Thinking Budget Allocator - Manages Extended Thinking Token Budgets

This module implements intelligent budget allocation for extended thinking operations.
Extended thinking (Sonnet 3.5+) is expensive, so we must allocate budgets wisely based
on task complexity.

Key Features:
- Calculate task complexity from various factors
- Allocate token budgets based on complexity scores
- Cost control and optimization
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from core.tdf_schema import TaskDefinition, TaskType


logger = logging.getLogger(__name__)


@dataclass
class ComplexityFactors:
    """
    Breakdown of factors contributing to task complexity.

    Attributes:
        context_size_score: Score based on context size (0-4)
        dependency_score: Score based on task dependencies (0-3)
        task_type_score: Score based on task type complexity (0-3)
        total_score: Combined complexity score (1-10)
    """
    context_size_score: int
    dependency_score: int
    task_type_score: int
    total_score: int

    def __repr__(self) -> str:
        return (
            f"ComplexityFactors(context={self.context_size_score}, "
            f"deps={self.dependency_score}, "
            f"type={self.task_type_score}, "
            f"total={self.total_score})"
        )


class ThinkingBudgetAllocator:
    """
    Allocator for Extended Thinking Token Budgets.

    This class calculates task complexity and allocates appropriate token budgets
    for extended thinking operations. The goal is to use expensive extended thinking
    only when truly needed.

    Complexity Scoring (1-10):
    - 1-3: Simple tasks (direct execution, no deep thinking needed)
    - 4-6: Moderate tasks (some planning required)
    - 7-10: Complex tasks (extensive planning and reasoning required)

    Budget Allocation:
    - Score 1-2: 1,000 tokens (minimal thinking)
    - Score 3-4: 2,500 tokens (basic planning)
    - Score 5-6: 5,000 tokens (moderate planning)
    - Score 7-8: 10,000 tokens (detailed planning)
    - Score 9-10: 32,000 tokens (maximum extended thinking)
    """

    # Budget mapping: complexity score -> token budget
    BUDGET_TIERS = {
        1: 1_000,
        2: 1_000,
        3: 2_500,
        4: 2_500,
        5: 5_000,
        6: 5_000,
        7: 10_000,
        8: 10_000,
        9: 32_000,
        10: 32_000
    }

    # Task type complexity scores
    TASK_TYPE_COMPLEXITY = {
        TaskType.ANALYSIS: 2,
        TaskType.IMPLEMENTATION: 3,
        TaskType.REVIEW: 1,
        TaskType.TESTING: 2,
        TaskType.DOCUMENTATION: 1,
        TaskType.DEPLOYMENT: 2,
        TaskType.RESEARCH: 3,
        TaskType.REFACTORING: 3
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ThinkingBudgetAllocator.

        Args:
            config: Optional configuration dictionary for custom settings
        """
        self.config = config or {}
        self.max_budget = self.config.get("max_budget", 32_000)
        self.min_budget = self.config.get("min_budget", 1_000)

        logger.info(
            f"ThinkingBudgetAllocator initialized: "
            f"budget range {self.min_budget}-{self.max_budget} tokens"
        )

    def calculate_complexity(self, task: TaskDefinition) -> int:
        """
        Calculate task complexity score (1-10).

        Complexity is determined by:
        1. Context size (larger context = more complex)
        2. Dependencies (more dependencies = more complex)
        3. Task type (some types are inherently more complex)

        Args:
            task: TaskDefinition to analyze

        Returns:
            Complexity score from 1 (simplest) to 10 (most complex)
        """
        # Factor 1: Context Size Score (0-4 points)
        context_size = self._calculate_context_size(task.context)
        context_score = self._score_context_size(context_size)

        # Factor 2: Dependency Score (0-3 points)
        dependency_count = self._count_dependencies(task)
        dependency_score = self._score_dependencies(dependency_count)

        # Factor 3: Task Type Score (0-3 points)
        task_type_score = self._score_task_type(task.task_type)

        # Calculate total score (1-10)
        raw_score = context_score + dependency_score + task_type_score
        # Ensure score is between 1 and 10
        complexity_score = max(1, min(10, raw_score))

        factors = ComplexityFactors(
            context_size_score=context_score,
            dependency_score=dependency_score,
            task_type_score=task_type_score,
            total_score=complexity_score
        )

        logger.debug(
            f"Complexity calculated for task {task.task_id}: "
            f"score={complexity_score}, factors={factors}"
        )

        return complexity_score

    def allocate_budget(self, complexity: int) -> int:
        """
        Allocate token budget based on complexity score.

        Maps complexity scores to appropriate token budgets:
        - Low complexity (1-2): Minimal thinking (1k tokens)
        - Medium complexity (3-6): Moderate thinking (2.5k-5k tokens)
        - High complexity (7-10): Extended thinking (10k-32k tokens)

        Args:
            complexity: Complexity score from calculate_complexity (1-10)

        Returns:
            Token budget allocation
        """
        if complexity < 1 or complexity > 10:
            logger.warning(
                f"Invalid complexity score {complexity}, clamping to 1-10"
            )
            complexity = max(1, min(10, complexity))

        budget = self.BUDGET_TIERS.get(complexity, self.min_budget)

        # Ensure budget is within configured limits
        budget = max(self.min_budget, min(self.max_budget, budget))

        logger.info(
            f"Allocated budget: {budget:,} tokens for complexity {complexity}"
        )

        return budget

    def calculate_and_allocate(
        self,
        task: TaskDefinition
    ) -> Dict[str, Any]:
        """
        Convenience method: calculate complexity and allocate budget in one call.

        Args:
            task: TaskDefinition to analyze

        Returns:
            Dictionary with complexity, budget, and metadata
        """
        complexity = self.calculate_complexity(task)
        budget = self.allocate_budget(complexity)

        return {
            "complexity": complexity,
            "budget_tokens": budget,
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "recommendation": self._get_recommendation(complexity, budget)
        }

    def _calculate_context_size(self, context: Dict[str, Any]) -> int:
        """
        Calculate approximate context size in characters.

        Args:
            context: Task context dictionary

        Returns:
            Approximate context size in characters
        """
        # Simple heuristic: count characters in all context values
        size = 0

        for key, value in context.items():
            if isinstance(value, str):
                size += len(value)
            elif isinstance(value, (list, dict)):
                # Rough estimate for complex types
                size += len(str(value))

        return size

    def _score_context_size(self, size: int) -> int:
        """
        Convert context size to score (0-4).

        Scoring:
        - < 500 chars: 0 points (minimal context)
        - 500-2000 chars: 1 point (small context)
        - 2000-5000 chars: 2 points (medium context)
        - 5000-10000 chars: 3 points (large context)
        - > 10000 chars: 4 points (very large context)

        Args:
            size: Context size in characters

        Returns:
            Score from 0 to 4
        """
        if size < 500:
            return 0
        elif size < 2000:
            return 1
        elif size < 5000:
            return 2
        elif size < 10000:
            return 3
        else:
            return 4

    def _count_dependencies(self, task: TaskDefinition) -> int:
        """
        Count task dependencies.

        Dependencies can be specified in:
        - task.requirements['dependencies'] (list)
        - task.context['depends_on'] (list)

        Args:
            task: TaskDefinition to analyze

        Returns:
            Number of dependencies
        """
        count = 0

        # Check requirements
        if 'dependencies' in task.requirements:
            deps = task.requirements['dependencies']
            if isinstance(deps, list):
                count += len(deps)

        # Check context
        if 'depends_on' in task.context:
            deps = task.context['depends_on']
            if isinstance(deps, list):
                count += len(deps)

        return count

    def _score_dependencies(self, count: int) -> int:
        """
        Convert dependency count to score (0-3).

        Scoring:
        - 0 dependencies: 0 points (independent task)
        - 1-2 dependencies: 1 point (simple dependencies)
        - 3-5 dependencies: 2 points (moderate dependencies)
        - 6+ dependencies: 3 points (complex dependencies)

        Args:
            count: Number of dependencies

        Returns:
            Score from 0 to 3
        """
        if count == 0:
            return 0
        elif count <= 2:
            return 1
        elif count <= 5:
            return 2
        else:
            return 3

    def _score_task_type(self, task_type: TaskType) -> int:
        """
        Get complexity score for task type.

        Args:
            task_type: TaskType enum value

        Returns:
            Score from 0 to 3 based on inherent task complexity
        """
        return self.TASK_TYPE_COMPLEXITY.get(task_type, 2)

    def _get_recommendation(self, complexity: int, budget: int) -> str:
        """
        Get human-readable recommendation for budget usage.

        Args:
            complexity: Complexity score
            budget: Allocated budget

        Returns:
            Recommendation string
        """
        if complexity <= 2:
            return "Direct execution - minimal thinking needed"
        elif complexity <= 4:
            return "Basic planning - simple task breakdown"
        elif complexity <= 6:
            return "Moderate planning - detailed task analysis"
        elif complexity <= 8:
            return "Extended thinking - complex planning required"
        else:
            return "Maximum extended thinking - highly complex task"
