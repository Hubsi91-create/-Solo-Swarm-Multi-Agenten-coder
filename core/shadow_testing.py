"""
Shadow Testing - Safe Sandbox for Prompt Optimization

Shadow Testing allows the Meta-Agent to safely test new prompt variants
against historical tasks WITHOUT causing any side effects (no database writes,
no git operations, no external API calls).

This is the A/B testing infrastructure for continuous prompt improvement.

Key Principles:
- Complete isolation (no side effects)
- Historical task replay
- Baseline comparison
- Statistical significance testing
- Safe rollback capability

Think of this as a "time machine" - we replay past tasks with new prompts
to see if they would have performed better.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import copy

from core.tdf_schema import TaskDefinition


logger = logging.getLogger(__name__)


class ShadowTestStatus(str, Enum):
    """Status of shadow test execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ComparisonResult(str, Enum):
    """Result of comparing shadow test to baseline"""
    BETTER = "better"  # Shadow performed better
    SAME = "same"  # Same performance
    WORSE = "worse"  # Shadow performed worse


@dataclass
class ShadowTestResult:
    """Result from a single shadow test run"""

    test_id: str
    task_id: str
    variant_id: int
    status: ShadowTestStatus
    success: bool
    execution_time_seconds: float
    output_quality_score: float  # 0-1 score
    baseline_success: bool
    baseline_quality_score: float
    comparison: ComparisonResult
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_id": self.test_id,
            "task_id": self.task_id,
            "variant_id": self.variant_id,
            "status": self.status.value,
            "success": self.success,
            "execution_time_seconds": self.execution_time_seconds,
            "output_quality_score": self.output_quality_score,
            "baseline_success": self.baseline_success,
            "baseline_quality_score": self.baseline_quality_score,
            "comparison": self.comparison.value,
            "timestamp": self.timestamp.isoformat()
        }


class ShadowTester:
    """
    Shadow Testing Framework for Safe Prompt Optimization.

    This class provides a sandboxed environment for testing new agent prompts
    against historical tasks without any side effects.

    Workflow:
    1. Load historical tasks with known outcomes
    2. Execute tasks with new prompt variant (in isolation)
    3. Compare results to baseline (historical outcome)
    4. Calculate statistical significance
    5. Return recommendation (deploy or reject)

    Safety Guarantees:
    - No database writes
    - No git operations
    - No external API calls (mocked)
    - No file system changes (in-memory only)
    - Complete rollback capability
    """

    def __init__(
        self,
        baseline_db_session: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Shadow Tester.

        Args:
            baseline_db_session: Read-only DB session for loading historical data
            config: Configuration dictionary
        """
        self.baseline_db_session = baseline_db_session
        self.config = config or {}

        # Testing configuration
        self.timeout_seconds = self.config.get("timeout_seconds", 60)
        self.min_quality_threshold = self.config.get("min_quality_threshold", 0.7)
        self.significance_level = self.config.get("significance_level", 0.05)  # p < 0.05

        # Test history
        self.test_results: List[ShadowTestResult] = []

        logger.info(
            f"ShadowTester initialized - "
            f"timeout: {self.timeout_seconds}s, "
            f"quality threshold: {self.min_quality_threshold}"
        )

    def run_shadow_test(
        self,
        new_prompt_variant: Dict[str, Any],
        historical_tasks: List[Dict[str, Any]],
        agent_type: str
    ) -> Dict[str, Any]:
        """
        Run shadow test of a new prompt variant against historical tasks.

        This is the main entry point for shadow testing.

        Args:
            new_prompt_variant: New prompt to test (from MetaAgent)
            historical_tasks: List of historical tasks with known outcomes
            agent_type: Type of agent being tested

        Returns:
            Dictionary with test results and recommendation
        """
        variant_id = new_prompt_variant.get("variant_id")
        strategy = new_prompt_variant.get("strategy")

        logger.info(
            f"🧪 Starting shadow test - "
            f"Variant {variant_id} ({strategy}) on {len(historical_tasks)} historical tasks"
        )

        test_results = []
        start_time = datetime.utcnow()

        # Run each historical task with new prompt
        for task_data in historical_tasks:
            result = self._execute_shadow_task(
                task_data=task_data,
                prompt_variant=new_prompt_variant,
                agent_type=agent_type
            )
            test_results.append(result)

        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()

        # Analyze results
        analysis = self._analyze_test_results(test_results)

        # Make recommendation
        recommendation = self._make_deployment_recommendation(analysis)

        logger.info(
            f"✅ Shadow test completed in {total_duration:.1f}s - "
            f"Success rate: {analysis['success_rate']:.1%}, "
            f"Recommendation: {recommendation['decision']}"
        )

        return {
            "variant_id": variant_id,
            "strategy": strategy,
            "tasks_tested": len(historical_tasks),
            "test_results": [r.to_dict() for r in test_results],
            "analysis": analysis,
            "recommendation": recommendation,
            "total_duration_seconds": total_duration,
            "tested_at": start_time.isoformat()
        }

    def _execute_shadow_task(
        self,
        task_data: Dict[str, Any],
        prompt_variant: Dict[str, Any],
        agent_type: str
    ) -> ShadowTestResult:
        """
        Execute a single task in shadow mode.

        This simulates task execution with the new prompt in complete isolation.
        NO side effects are allowed.

        Args:
            task_data: Historical task data with baseline outcome
            prompt_variant: New prompt to test
            agent_type: Agent type being tested

        Returns:
            ShadowTestResult with execution outcome
        """
        test_id = f"shadow_{agent_type}_{int(datetime.utcnow().timestamp())}"
        task_id = task_data.get("task_id")

        # Get baseline (historical) outcome
        baseline_success = task_data.get("success", False)
        baseline_quality = task_data.get("quality_score", 0.5)

        try:
            # CRITICAL: Execute in complete isolation
            # In production, this would:
            # 1. Clone the agent instance
            # 2. Apply new prompt
            # 3. Mock all external dependencies
            # 4. Execute task
            # 5. Score output quality

            # For now, simulate execution
            shadow_result = self._simulate_shadow_execution(
                task_data=task_data,
                prompt_variant=prompt_variant,
                agent_type=agent_type
            )

            # Compare to baseline
            comparison = self._compare_to_baseline(
                shadow_success=shadow_result["success"],
                shadow_quality=shadow_result["quality_score"],
                baseline_success=baseline_success,
                baseline_quality=baseline_quality
            )

            return ShadowTestResult(
                test_id=test_id,
                task_id=task_id,
                variant_id=prompt_variant.get("variant_id"),
                status=ShadowTestStatus.COMPLETED,
                success=shadow_result["success"],
                execution_time_seconds=shadow_result["execution_time"],
                output_quality_score=shadow_result["quality_score"],
                baseline_success=baseline_success,
                baseline_quality_score=baseline_quality,
                comparison=comparison,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Shadow test failed for task {task_id}: {e}")

            return ShadowTestResult(
                test_id=test_id,
                task_id=task_id,
                variant_id=prompt_variant.get("variant_id"),
                status=ShadowTestStatus.FAILED,
                success=False,
                execution_time_seconds=0.0,
                output_quality_score=0.0,
                baseline_success=baseline_success,
                baseline_quality_score=baseline_quality,
                comparison=ComparisonResult.WORSE,
                timestamp=datetime.utcnow()
            )

    def _simulate_shadow_execution(
        self,
        task_data: Dict[str, Any],
        prompt_variant: Dict[str, Any],
        agent_type: str
    ) -> Dict[str, Any]:
        """
        Simulate shadow task execution.

        In production, this would actually execute the task with mocked dependencies.
        For now, we simulate based on the estimated improvement from the variant.

        Args:
            task_data: Task data
            prompt_variant: Prompt variant being tested
            agent_type: Agent type

        Returns:
            Simulated execution result
        """
        # Get baseline outcome
        baseline_success = task_data.get("success", False)
        baseline_quality = task_data.get("quality_score", 0.5)

        # Get estimated improvement from prompt variant
        estimated_improvement = prompt_variant.get("estimated_improvement", 0.10)

        # Simulate: new prompt has estimated improvement probability
        import random
        random.seed(task_data.get("task_id"))  # Deterministic based on task

        # If baseline failed, new prompt might fix it
        if not baseline_success:
            # Probability of fixing = estimated_improvement * 2
            success = random.random() < (estimated_improvement * 2)
            quality_score = baseline_quality + (estimated_improvement if success else 0)
        else:
            # If baseline succeeded, might do even better
            success = True
            quality_score = min(1.0, baseline_quality + (estimated_improvement * 0.5))

        return {
            "success": success,
            "quality_score": quality_score,
            "execution_time": random.uniform(1.0, 5.0),  # Simulated time
            "side_effects": []  # CRITICAL: No side effects in shadow mode!
        }

    def _compare_to_baseline(
        self,
        shadow_success: bool,
        shadow_quality: float,
        baseline_success: bool,
        baseline_quality: float
    ) -> ComparisonResult:
        """
        Compare shadow test result to baseline.

        Args:
            shadow_success: Did shadow test succeed?
            shadow_quality: Shadow output quality score
            baseline_success: Did baseline succeed?
            baseline_quality: Baseline output quality score

        Returns:
            ComparisonResult enum
        """
        # If shadow fixed a failure, it's better
        if not baseline_success and shadow_success:
            return ComparisonResult.BETTER

        # If shadow broke a success, it's worse
        if baseline_success and not shadow_success:
            return ComparisonResult.WORSE

        # Both succeeded or both failed - compare quality
        quality_diff = shadow_quality - baseline_quality

        if quality_diff > 0.05:  # 5% improvement threshold
            return ComparisonResult.BETTER
        elif quality_diff < -0.05:  # 5% degradation threshold
            return ComparisonResult.WORSE
        else:
            return ComparisonResult.SAME

    def _analyze_test_results(
        self,
        test_results: List[ShadowTestResult]
    ) -> Dict[str, Any]:
        """
        Analyze shadow test results statistically.

        Args:
            test_results: List of shadow test results

        Returns:
            Statistical analysis dictionary
        """
        if not test_results:
            return {}

        # Calculate metrics
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r.success)
        better_count = sum(1 for r in test_results if r.comparison == ComparisonResult.BETTER)
        worse_count = sum(1 for r in test_results if r.comparison == ComparisonResult.WORSE)
        same_count = sum(1 for r in test_results if r.comparison == ComparisonResult.SAME)

        # Calculate baseline metrics
        baseline_successes = sum(1 for r in test_results if r.baseline_success)

        # Calculate quality improvements
        avg_shadow_quality = sum(r.output_quality_score for r in test_results) / total_tests
        avg_baseline_quality = sum(r.baseline_quality_score for r in test_results) / total_tests
        quality_improvement = avg_shadow_quality - avg_baseline_quality

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "baseline_success_rate": baseline_successes / total_tests if total_tests > 0 else 0,
            "better_than_baseline": better_count,
            "worse_than_baseline": worse_count,
            "same_as_baseline": same_count,
            "improvement_rate": better_count / total_tests if total_tests > 0 else 0,
            "degradation_rate": worse_count / total_tests if total_tests > 0 else 0,
            "avg_shadow_quality": round(avg_shadow_quality, 3),
            "avg_baseline_quality": round(avg_baseline_quality, 3),
            "quality_improvement": round(quality_improvement, 3),
            "confidence_level": self._calculate_confidence(test_results)
        }

    def _calculate_confidence(self, test_results: List[ShadowTestResult]) -> float:
        """
        Calculate statistical confidence in the results.

        In production, this would use proper statistical tests (t-test, etc).
        For now, we use sample size and consistency as proxies.

        Args:
            test_results: List of test results

        Returns:
            Confidence score (0-1)
        """
        if not test_results:
            return 0.0

        sample_size = len(test_results)
        better_count = sum(1 for r in test_results if r.comparison == ComparisonResult.BETTER)
        worse_count = sum(1 for r in test_results if r.comparison == ComparisonResult.WORSE)

        # Sample size factor (more tests = more confidence)
        size_confidence = min(1.0, sample_size / 20)  # Max confidence at 20+ tests

        # Consistency factor (consistent direction = more confidence)
        if better_count > worse_count:
            consistency = better_count / (better_count + worse_count) if (better_count + worse_count) > 0 else 0.5
        else:
            consistency = worse_count / (better_count + worse_count) if (better_count + worse_count) > 0 else 0.5

        # Combined confidence
        confidence = (size_confidence * 0.4) + (consistency * 0.6)

        return round(confidence, 2)

    def _make_deployment_recommendation(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make deployment recommendation based on test analysis.

        Args:
            analysis: Statistical analysis of test results

        Returns:
            Recommendation dictionary
        """
        improvement_rate = analysis.get("improvement_rate", 0)
        degradation_rate = analysis.get("degradation_rate", 0)
        quality_improvement = analysis.get("quality_improvement", 0)
        confidence = analysis.get("confidence_level", 0)
        success_rate = analysis.get("success_rate", 0)

        # Decision criteria
        deploy = (
            improvement_rate >= 0.30 and  # 30%+ improvements
            degradation_rate <= 0.10 and  # Max 10% degradations
            quality_improvement > 0.05 and  # +5% quality
            confidence >= 0.70 and  # 70%+ confidence
            success_rate >= self.min_quality_threshold  # Meets minimum quality
        )

        if deploy:
            decision = "DEPLOY"
            reason = f"Strong evidence of improvement ({improvement_rate:.1%} improvement rate, {quality_improvement:+.1%} quality gain, {confidence:.0%} confidence)"
        elif degradation_rate > 0.20:
            decision = "REJECT"
            reason = f"Too many degradations ({degradation_rate:.1%})"
        elif confidence < 0.50:
            decision = "MORE_TESTING_NEEDED"
            reason = f"Insufficient confidence ({confidence:.0%}) - need more test cases"
        else:
            decision = "REJECT"
            reason = "Does not meet deployment criteria"

        return {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "expected_improvement": quality_improvement,
            "risk_level": "low" if degradation_rate < 0.10 else ("medium" if degradation_rate < 0.20 else "high")
        }

    def load_historical_tasks(
        self,
        agent_type: str,
        limit: int = 20,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load historical tasks for shadow testing.

        Args:
            agent_type: Type of agent
            limit: Maximum number of tasks to load
            success_only: Only load successful tasks

        Returns:
            List of historical task dictionaries
        """
        # In production: Query database for historical tasks
        # For now: Generate simulated historical tasks

        historical_tasks = []

        for i in range(limit):
            # Mix of successes and failures
            success = (i % 4 != 0) if not success_only else True  # 75% success rate

            historical_tasks.append({
                "task_id": f"historical_{agent_type}_{i}",
                "agent_type": agent_type,
                "success": success,
                "quality_score": 0.8 if success else 0.4,
                "execution_time": 3.5,
                "timestamp": datetime.utcnow().isoformat()
            })

        logger.info(
            f"Loaded {len(historical_tasks)} historical tasks for {agent_type} "
            f"(success rate: {sum(1 for t in historical_tasks if t['success']) / len(historical_tasks):.1%})"
        )

        return historical_tasks

    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get summary of all shadow tests run.

        Returns:
            Summary dictionary
        """
        if not self.test_results:
            return {"total_tests": 0, "message": "No tests run yet"}

        return {
            "total_tests": len(self.test_results),
            "by_status": {
                status.value: sum(1 for r in self.test_results if r.status == status)
                for status in ShadowTestStatus
            },
            "overall_success_rate": sum(1 for r in self.test_results if r.success) / len(self.test_results),
            "avg_execution_time": sum(r.execution_time_seconds for r in self.test_results) / len(self.test_results)
        }
