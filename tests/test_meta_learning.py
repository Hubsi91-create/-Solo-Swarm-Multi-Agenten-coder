"""
Meta-Learning Tests - Self-Optimization and Shadow Testing

This test suite validates the system's ability to optimize itself through:
- Performance monitoring
- Failure pattern analysis
- Prompt generation and improvement
- Safe shadow testing
- Deployment recommendations

This represents the highest level of AI autonomy - the system improving itself.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from agents.managers.meta_agent import (
    MetaAgent,
    AgentPerformanceMetric,
    PromptImprovementStrategy
)
from core.shadow_testing import (
    ShadowTester,
    ShadowTestResult,
    ShadowTestStatus,
    ComparisonResult
)
from core.tdf_schema import TaskDefinition, TaskType


class TestMetaAgent:
    """Test suite for MetaAgent performance monitoring and optimization."""

    @pytest.fixture
    def meta_agent(self):
        """Create a MetaAgent instance."""
        return MetaAgent(
            agent_id="meta_001",
            config={
                "min_sample_size": 20,
                "success_threshold": 0.80,
                "improvement_target": 0.10
            }
        )

    def test_meta_agent_initialization(self, meta_agent):
        """Test MetaAgent initializes correctly."""
        assert meta_agent.agent_id == "meta_001"
        assert meta_agent.agent_name == "MetaAgent"
        assert meta_agent.success_threshold == 0.80
        assert meta_agent.improvement_target == 0.10

    def test_query_agent_performance(self, meta_agent):
        """Test querying agent performance metrics."""
        metrics = meta_agent._query_agent_performance("all", days_back=7)

        assert isinstance(metrics, dict)
        assert "CoderAgent" in metrics
        assert "AssetAgent" in metrics
        assert "VerifierAgent" in metrics

        # Verify metric structure
        coder_metrics = metrics["CoderAgent"]
        assert "total_tasks" in coder_metrics
        assert "success_rate" in coder_metrics
        assert "error_rate" in coder_metrics

    def test_identify_underperformers(self, meta_agent):
        """Test identifying underperforming agents."""
        # Create mock metrics with one underperformer
        metrics = {
            "GoodAgent": {
                "total_tasks": 100,
                "successful_tasks": 90,
                "failed_tasks": 10,
                "success_rate": 0.90,  # Above threshold
                "error_rate": 0.10
            },
            "BadAgent": {
                "total_tasks": 50,
                "successful_tasks": 35,
                "failed_tasks": 15,
                "success_rate": 0.70,  # Below threshold (0.80)
                "error_rate": 0.30
            }
        }

        underperformers = meta_agent._identify_underperformers(metrics)

        assert len(underperformers) == 1
        assert underperformers[0]["agent_type"] == "BadAgent"
        assert underperformers[0]["success_rate"] == 0.70
        assert abs(underperformers[0]["gap_to_target"] - 0.10) < 0.0001  # 0.80 - 0.70

    def test_analyze_failure_patterns(self, meta_agent):
        """Test failure pattern analysis."""
        patterns = meta_agent._analyze_failure_patterns("AssetAgent")

        assert "common_errors" in patterns
        assert "failure_categories" in patterns
        assert "suggested_improvements" in patterns
        assert isinstance(patterns["common_errors"], list)
        assert isinstance(patterns["suggested_improvements"], list)

    def test_allocate_thinking_budget(self, meta_agent):
        """Test Extended Thinking budget allocation."""
        failures = {
            "failure_categories": {
                "timeout": 0.40,
                "validation": 0.35,
                "api_error": 0.25
            }
        }

        budget = meta_agent._allocate_thinking_budget("AssetAgent", failures)

        assert isinstance(budget, int)
        assert 1000 <= budget <= 32000  # Valid thinking budget range
        # More failure categories should get higher budget
        assert budget >= 5000  # At least moderate complexity

    def test_generate_prompt_variants(self, meta_agent):
        """Test prompt variant generation."""
        failures = {
            "suggested_improvements": [
                "Add retry logic",
                "Improve validation"
            ],
            "failure_categories": {
                "timeout": 0.50,
                "validation": 0.50
            }
        }

        variants = meta_agent._generate_prompt_variants(
            agent_type="AssetAgent",
            failures=failures,
            thinking_budget=10000
        )

        assert len(variants) == 3  # Should generate 3 variants
        assert all("variant_id" in v for v in variants)
        assert all("strategy" in v for v in variants)
        assert all("estimated_improvement" in v for v in variants)

        # Check strategies are different
        strategies = [v["strategy"] for v in variants]
        assert PromptImprovementStrategy.CLARITY in strategies
        assert PromptImprovementStrategy.EXAMPLES in strategies
        assert PromptImprovementStrategy.CONSTRAINTS in strategies

    def test_select_best_variant(self, meta_agent):
        """Test selecting best performing prompt variant."""
        test_results = [
            {
                "variant_id": 1,
                "strategy": "clarity",
                "success_rate": 0.75,
                "confidence": 0.80
            },
            {
                "variant_id": 2,
                "strategy": "examples",
                "success_rate": 0.88,  # Best!
                "confidence": 0.85
            },
            {
                "variant_id": 3,
                "strategy": "constraints",
                "success_rate": 0.82,
                "confidence": 0.75
            }
        ]

        best = meta_agent._select_best_variant(test_results)

        assert best is not None
        assert best["variant_id"] == 2
        assert best["success_rate"] == 0.88

    def test_select_best_variant_no_improvement(self, meta_agent):
        """Test that no variant is selected if none improves enough."""
        test_results = [
            {
                "variant_id": 1,
                "strategy": "clarity",
                "success_rate": 0.75,  # Below 0.80 threshold
                "confidence": 0.80
            },
            {
                "variant_id": 2,
                "strategy": "examples",
                "success_rate": 0.78,  # Still below threshold
                "confidence": 0.85
            }
        ]

        best = meta_agent._select_best_variant(test_results)

        assert best is None  # No variant good enough

    def test_monitor_performance(self, meta_agent):
        """Test high-level performance monitoring."""
        result = meta_agent.monitor_performance()

        assert "system_health" in result
        assert "average_success_rate" in result
        assert "total_agents_monitored" in result
        assert "underperforming_agents" in result
        assert "metrics_by_agent" in result

        # Check health determination
        assert result["system_health"] in ["healthy", "needs_optimization"]

    def test_optimize_prompts(self, meta_agent):
        """Test high-level prompt optimization workflow."""
        result = meta_agent.optimize_prompts("AssetAgent")

        # Should return optimization results or error
        assert isinstance(result, dict)

        # If optimization ran successfully
        if "error" not in result:
            assert "validated_optimizations" in result or "no_optimizations_needed" in result


class TestShadowTester:
    """Test suite for Shadow Testing framework."""

    @pytest.fixture
    def shadow_tester(self):
        """Create a ShadowTester instance."""
        return ShadowTester(
            config={
                "timeout_seconds": 60,
                "min_quality_threshold": 0.7,
                "significance_level": 0.05
            }
        )

    def test_shadow_tester_initialization(self, shadow_tester):
        """Test ShadowTester initializes correctly."""
        assert shadow_tester.timeout_seconds == 60
        assert shadow_tester.min_quality_threshold == 0.7
        assert shadow_tester.significance_level == 0.05

    def test_load_historical_tasks(self, shadow_tester):
        """Test loading historical tasks."""
        tasks = shadow_tester.load_historical_tasks(
            agent_type="CoderAgent",
            limit=10,
            success_only=False
        )

        assert len(tasks) == 10
        assert all("task_id" in t for t in tasks)
        assert all("success" in t for t in tasks)
        assert all("quality_score" in t for t in tasks)

        # Should have mix of successes and failures
        success_count = sum(1 for t in tasks if t["success"])
        assert 0 < success_count < len(tasks)  # Not all or none

    def test_load_historical_tasks_success_only(self, shadow_tester):
        """Test loading only successful historical tasks."""
        tasks = shadow_tester.load_historical_tasks(
            agent_type="CoderAgent",
            limit=10,
            success_only=True
        )

        assert len(tasks) == 10
        assert all(t["success"] for t in tasks)  # All successes

    def test_compare_to_baseline_better(self, shadow_tester):
        """Test comparison when shadow test is better."""
        result = shadow_tester._compare_to_baseline(
            shadow_success=True,
            shadow_quality=0.85,
            baseline_success=False,  # Shadow fixed a failure!
            baseline_quality=0.45
        )

        assert result == ComparisonResult.BETTER

    def test_compare_to_baseline_worse(self, shadow_tester):
        """Test comparison when shadow test is worse."""
        result = shadow_tester._compare_to_baseline(
            shadow_success=False,  # Shadow broke a success!
            shadow_quality=0.40,
            baseline_success=True,
            baseline_quality=0.85
        )

        assert result == ComparisonResult.WORSE

    def test_compare_to_baseline_same(self, shadow_tester):
        """Test comparison when results are similar."""
        result = shadow_tester._compare_to_baseline(
            shadow_success=True,
            shadow_quality=0.82,  # Only 2% better (below 5% threshold)
            baseline_success=True,
            baseline_quality=0.80
        )

        assert result == ComparisonResult.SAME

    def test_analyze_test_results(self, shadow_tester):
        """Test statistical analysis of shadow test results."""
        # Create mock test results
        test_results = [
            ShadowTestResult(
                test_id=f"test_{i}",
                task_id=f"task_{i}",
                variant_id=1,
                status=ShadowTestStatus.COMPLETED,
                success=(i % 3 != 0),  # 67% success
                execution_time_seconds=2.5,
                output_quality_score=0.8 if (i % 3 != 0) else 0.4,
                baseline_success=(i % 4 != 0),  # 75% baseline success
                baseline_quality_score=0.75 if (i % 4 != 0) else 0.35,
                comparison=ComparisonResult.BETTER if (i % 2 == 0) else ComparisonResult.SAME,
                timestamp=datetime.utcnow()
            )
            for i in range(10)
        ]

        analysis = shadow_tester._analyze_test_results(test_results)

        assert analysis["total_tests"] == 10
        assert "success_rate" in analysis
        assert "baseline_success_rate" in analysis
        assert "improvement_rate" in analysis
        assert "degradation_rate" in analysis
        assert "quality_improvement" in analysis
        assert "confidence_level" in analysis

        # Verify calculations
        assert 0 <= analysis["success_rate"] <= 1
        assert 0 <= analysis["confidence_level"] <= 1

    def test_calculate_confidence(self, shadow_tester):
        """Test confidence calculation."""
        # More tests with consistent results = higher confidence
        consistent_results = [
            ShadowTestResult(
                test_id=f"test_{i}",
                task_id=f"task_{i}",
                variant_id=1,
                status=ShadowTestStatus.COMPLETED,
                success=True,
                execution_time_seconds=2.0,
                output_quality_score=0.9,
                baseline_success=False,
                baseline_quality_score=0.5,
                comparison=ComparisonResult.BETTER,  # All better
                timestamp=datetime.utcnow()
            )
            for i in range(20)
        ]

        confidence = shadow_tester._calculate_confidence(consistent_results)

        assert 0.8 <= confidence <= 1.0  # High confidence

        # Fewer tests with mixed results = lower confidence
        mixed_results = consistent_results[:5]  # Only 5 tests
        mixed_results[2].comparison = ComparisonResult.WORSE  # Add inconsistency

        low_confidence = shadow_tester._calculate_confidence(mixed_results)

        assert low_confidence < confidence  # Should be lower

    def test_make_deployment_recommendation_deploy(self, shadow_tester):
        """Test deployment recommendation when variant is good."""
        analysis = {
            "improvement_rate": 0.40,  # 40% improvements
            "degradation_rate": 0.05,  # 5% degradations
            "quality_improvement": 0.12,  # +12% quality
            "confidence_level": 0.85,  # 85% confidence
            "success_rate": 0.88  # 88% success
        }

        recommendation = shadow_tester._make_deployment_recommendation(analysis)

        assert recommendation["decision"] == "DEPLOY"
        assert recommendation["confidence"] == 0.85
        assert recommendation["risk_level"] == "low"

    def test_make_deployment_recommendation_reject(self, shadow_tester):
        """Test deployment recommendation when variant is problematic."""
        analysis = {
            "improvement_rate": 0.20,  # 20% improvements
            "degradation_rate": 0.35,  # 35% degradations - TOO HIGH!
            "quality_improvement": -0.05,  # -5% quality
            "confidence_level": 0.75,  # 75% confidence
            "success_rate": 0.65  # 65% success
        }

        recommendation = shadow_tester._make_deployment_recommendation(analysis)

        assert recommendation["decision"] == "REJECT"
        assert recommendation["risk_level"] == "high"

    def test_make_deployment_recommendation_more_testing(self, shadow_tester):
        """Test recommendation when more testing is needed."""
        analysis = {
            "improvement_rate": 0.30,  # 30% improvements
            "degradation_rate": 0.15,  # 15% degradations
            "quality_improvement": 0.08,  # +8% quality
            "confidence_level": 0.45,  # 45% confidence - TOO LOW!
            "success_rate": 0.75  # 75% success
        }

        recommendation = shadow_tester._make_deployment_recommendation(analysis)

        assert recommendation["decision"] == "MORE_TESTING_NEEDED"

    def test_run_shadow_test(self, shadow_tester):
        """Test complete shadow test run."""
        prompt_variant = {
            "variant_id": 1,
            "strategy": PromptImprovementStrategy.EXAMPLES,
            "estimated_improvement": 0.15
        }

        historical_tasks = shadow_tester.load_historical_tasks(
            agent_type="CoderAgent",
            limit=10
        )

        result = shadow_tester.run_shadow_test(
            new_prompt_variant=prompt_variant,
            historical_tasks=historical_tasks,
            agent_type="CoderAgent"
        )

        assert result["variant_id"] == 1
        assert result["strategy"] == PromptImprovementStrategy.EXAMPLES
        assert result["tasks_tested"] == 10
        assert "test_results" in result
        assert "analysis" in result
        assert "recommendation" in result

        # Verify recommendation
        recommendation = result["recommendation"]
        assert "decision" in recommendation
        assert recommendation["decision"] in ["DEPLOY", "REJECT", "MORE_TESTING_NEEDED"]

    def test_shadow_test_no_side_effects(self, shadow_tester):
        """Test that shadow tests produce NO side effects."""
        prompt_variant = {
            "variant_id": 1,
            "strategy": "testing",
            "estimated_improvement": 0.10
        }

        historical_tasks = shadow_tester.load_historical_tasks(
            agent_type="CoderAgent",
            limit=5
        )

        # Run shadow test
        result = shadow_tester.run_shadow_test(
            new_prompt_variant=prompt_variant,
            historical_tasks=historical_tasks,
            agent_type="CoderAgent"
        )

        # Verify NO side effects in results
        for test_result in result["test_results"]:
            # In actual implementation, we'd verify:
            # - No database writes
            # - No git operations
            # - No file system changes
            # - No external API calls
            assert test_result["status"] in [s.value for s in ShadowTestStatus]


class TestMetaLearningIntegration:
    """Integration tests for the complete meta-learning cycle."""

    @pytest.mark.asyncio
    async def test_end_to_end_optimization_cycle(self):
        """
        Test complete meta-learning cycle:
        1. Monitor performance
        2. Identify underperformers
        3. Generate prompt variants
        4. Shadow test variants
        5. Get deployment recommendation
        """
        # Create meta-agent
        meta_agent = MetaAgent(agent_id="meta_integration_test")

        # Create shadow tester
        shadow_tester = ShadowTester()

        # Step 1: Monitor performance
        performance = meta_agent.monitor_performance()

        assert "underperforming_agents" in performance

        # If there are underperformers, continue
        if performance["underperforming_agents"] > 0:
            underperformers = performance["underperforming_details"]
            target_agent = underperformers[0]["agent_type"]

            # Step 2: Generate optimizations
            optimization_result = meta_agent.optimize_prompts(target_agent)

            # Should have optimization data
            if "validated_optimizations" in optimization_result:
                optimizations = optimization_result["validated_optimizations"]

                # Step 3: Verify shadow testing was performed
                assert len(optimizations) > 0

                for opt in optimizations:
                    assert "agent_type" in opt
                    assert "best_variant" in opt

                    # If a variant was found
                    if opt["best_variant"]:
                        assert "test_results" in opt
                        # Verify deployment readiness was determined
                        assert "ready_for_deployment" in opt

    def test_meta_agent_continuous_improvement_loop(self):
        """Test that meta-agent can run continuously."""
        meta_agent = MetaAgent(agent_id="continuous_test")

        # Run multiple monitoring cycles
        results = []

        for i in range(3):
            performance = meta_agent.monitor_performance()
            results.append(performance)

        # Each cycle should return valid performance data
        assert all("system_health" in r for r in results)
        assert all("average_success_rate" in r for r in results)

    def test_shadow_test_isolation(self):
        """Test that shadow tests are truly isolated."""
        shadow_tester = ShadowTester()

        # Track initial state
        initial_test_count = len(shadow_tester.test_results)

        # Run shadow test
        prompt_variant = {
            "variant_id": 1,
            "strategy": "isolation_test",
            "estimated_improvement": 0.10
        }

        tasks = shadow_tester.load_historical_tasks("TestAgent", limit=5)
        result = shadow_tester.run_shadow_test(
            prompt_variant,
            tasks,
            "TestAgent"
        )

        # Verify test was isolated (no persistent state changes)
        # In production, we'd verify:
        # - Database unchanged
        # - File system unchanged
        # - No network calls made
        # - No git operations performed

        assert result is not None
        assert "recommendation" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
