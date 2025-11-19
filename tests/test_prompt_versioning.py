"""
Tests for Prompt Versioning System (Genetic Memory)

This test suite validates:
1. PromptManager version creation and retrieval
2. Rollback functionality
3. Meta-Agent integration with PromptManager
4. Performance tracking and comparison
5. Automatic rollback on performance degradation
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from core.prompt_manager import PromptManager
from infrastructure.database import PromptVersionModel
from agents.managers.meta_agent import MetaAgent


@pytest.fixture
async def mock_db_session():
    """Mock database session for testing."""
    session = AsyncMock()
    return session


@pytest.fixture
async def prompt_manager(mock_db_session):
    """Create PromptManager with mocked session."""
    pm = PromptManager()
    pm.set_session(mock_db_session)
    return pm


@pytest.mark.asyncio
async def test_create_initial_version(prompt_manager, mock_db_session):
    """Test creating the first version of a prompt."""
    # Mock database execute result
    mock_db_session.execute = AsyncMock(return_value=Mock(scalar=Mock(return_value=None)))
    mock_db_session.commit = AsyncMock()
    mock_db_session.refresh = AsyncMock()

    # Create initial version
    version = await prompt_manager.create_initial_version(
        agent_type="coder_agent",
        initial_content="You are a coding assistant.",
        description="Initial prompt for coder agent"
    )

    assert mock_db_session.add.called
    assert mock_db_session.commit.called


@pytest.mark.asyncio
async def test_update_prompt_creates_new_version(prompt_manager, mock_db_session):
    """Test that update_prompt creates a new version and deactivates old."""
    # Mock existing version (version 1)
    old_version = PromptVersionModel(
        agent_type="qa_agent",
        version=1,
        content="Old prompt",
        change_reason="Initial",
        is_active=True
    )

    # Mock database responses
    mock_db_session.execute = AsyncMock(side_effect=[
        Mock(scalar=Mock(return_value=1)),  # max version
        Mock(scalar_one_or_none=Mock(return_value=old_version))  # current active
    ])
    mock_db_session.commit = AsyncMock()
    mock_db_session.refresh = AsyncMock()

    # Update prompt
    new_version = await prompt_manager.update_prompt(
        agent_type="qa_agent",
        new_content="Improved prompt",
        change_reason="Meta-Agent optimization: +15% success rate",
        performance_metrics={
            "success_rate": 85.0,
            "avg_cost": 0.02
        }
    )

    # Verify old version was deactivated
    assert old_version.is_active == False
    assert old_version.deactivated_at is not None

    # Verify new version was added
    assert mock_db_session.add.called


@pytest.mark.asyncio
async def test_rollback_prompt(prompt_manager, mock_db_session):
    """Test rolling back to a previous version."""
    # Current version (v3)
    current = PromptVersionModel(
        agent_type="asset_agent",
        version=3,
        content="Latest prompt",
        change_reason="Recent change",
        is_active=True,
        success_rate=70.0
    )

    # Target version (v2)
    target = PromptVersionModel(
        agent_type="asset_agent",
        version=2,
        content="Previous prompt",
        change_reason="Stable version",
        is_active=False,
        success_rate=85.0
    )

    # Mock database responses
    mock_db_session.execute = AsyncMock(side_effect=[
        Mock(scalar_one_or_none=Mock(return_value=target)),  # find target
        Mock(scalar_one_or_none=Mock(return_value=current))  # find current
    ])
    mock_db_session.commit = AsyncMock()
    mock_db_session.refresh = AsyncMock()

    # Perform rollback
    result = await prompt_manager.rollback_prompt(
        agent_type="asset_agent",
        target_version=2,
        reason="Performance degradation detected"
    )

    # Verify current was deactivated
    assert current.is_active == False

    # Verify target was reactivated
    assert target.is_active == True
    assert target.activated_at is not None

    assert mock_db_session.commit.called


@pytest.mark.asyncio
async def test_get_version_history(prompt_manager, mock_db_session):
    """Test retrieving version history."""
    # Mock versions
    versions = [
        PromptVersionModel(
            agent_type="verifier_agent",
            version=3,
            content="v3",
            change_reason="Latest",
            is_active=True,
            success_rate=92.0
        ),
        PromptVersionModel(
            agent_type="verifier_agent",
            version=2,
            content="v2",
            change_reason="Previous",
            is_active=False,
            success_rate=88.0
        ),
        PromptVersionModel(
            agent_type="verifier_agent",
            version=1,
            content="v1",
            change_reason="Initial",
            is_active=False,
            success_rate=80.0
        )
    ]

    mock_result = Mock()
    mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=versions)))
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    # Get history
    history = await prompt_manager.get_version_history("verifier_agent", limit=10)

    assert len(history) == 3
    assert history[0].version == 3
    assert history[0].is_active == True


@pytest.mark.asyncio
async def test_performance_comparison(prompt_manager, mock_db_session):
    """Test performance comparison across versions."""
    versions = [
        PromptVersionModel(
            agent_type="coder_agent",
            version=3,
            content="v3",
            change_reason="Latest",
            is_active=True,
            performance_score=85.0,
            success_rate=87.0
        ),
        PromptVersionModel(
            agent_type="coder_agent",
            version=2,
            content="v2",
            change_reason="Previous",
            is_active=False,
            performance_score=92.0,  # Best!
            success_rate=90.0
        ),
        PromptVersionModel(
            agent_type="coder_agent",
            version=1,
            content="v1",
            change_reason="Initial",
            is_active=False,
            performance_score=75.0,  # Worst
            success_rate=78.0
        )
    ]

    mock_result = Mock()
    mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=versions)))
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    # Get comparison
    comparison = await prompt_manager.get_performance_comparison("coder_agent")

    assert comparison["agent_type"] == "coder_agent"
    assert comparison["current_version"]["version"] == 3
    assert comparison["best_version"]["version"] == 2
    assert comparison["best_version"]["score"] == 92.0
    assert comparison["worst_version"]["version"] == 1
    assert comparison["current_version"]["is_best"] == False  # v3 is not the best


@pytest.mark.asyncio
async def test_meta_agent_deploys_optimization(mock_db_session):
    """Test that Meta-Agent deploys optimizations to PromptManager."""
    # Create Meta-Agent with PromptManager
    prompt_manager = PromptManager()
    prompt_manager.set_session(mock_db_session)

    meta_agent = MetaAgent(
        agent_id="meta_001",
        db_session=mock_db_session,
        prompt_manager=prompt_manager
    )

    # Mock optimization result with best variant
    optimization = {
        "agent_type": "qa_agent",
        "original_success_rate": 0.75,
        "target_success_rate": 0.85,
        "ready_for_deployment": True,
        "best_variant": {
            "prompt_text": "Optimized QA prompt",
            "success_rate": 0.87,
            "performance_score": 88.0,
            "avg_cost": 0.015,
            "avg_duration": 12.5,
            "strategy": "CLARITY",
            "test_count": 10,
            "test_results": {}
        }
    }

    # Mock database for deployment
    mock_db_session.execute = AsyncMock(side_effect=[
        Mock(scalar=Mock(return_value=1)),  # max version
        Mock(scalar_one_or_none=Mock(return_value=None))  # no current active
    ])
    mock_db_session.commit = AsyncMock()
    mock_db_session.refresh = AsyncMock()

    # Deploy optimization
    deployment = await meta_agent._deploy_optimization(optimization)

    assert deployment["deployed"] == True
    assert deployment["agent_type"] == "qa_agent"
    assert deployment["version"] == 2
    assert deployment["improvement_percent"] == 12.0  # 87% - 75%
    assert mock_db_session.add.called


@pytest.mark.asyncio
async def test_meta_agent_auto_rollback(mock_db_session):
    """Test that Meta-Agent automatically rolls back on performance degradation."""
    prompt_manager = PromptManager()
    prompt_manager.set_session(mock_db_session)

    meta_agent = MetaAgent(
        agent_id="meta_001",
        db_session=mock_db_session,
        prompt_manager=prompt_manager,
        config={
            "auto_rollback_enabled": True,
            "rollback_threshold": 0.10  # 10% degradation triggers rollback
        }
    )

    # Mock version history: current (v2) is worse than previous (v1)
    current = PromptVersionModel(
        agent_type="asset_agent",
        version=2,
        content="New prompt",
        change_reason="Recent optimization",
        is_active=True,
        success_rate=72.0  # Dropped from 85%!
    )

    previous = PromptVersionModel(
        agent_type="asset_agent",
        version=1,
        content="Old prompt",
        change_reason="Original",
        is_active=False,
        success_rate=85.0
    )

    mock_result = Mock()
    mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[current, previous])))

    mock_db_session.execute = AsyncMock(side_effect=[
        mock_result,  # get history
        Mock(scalar_one_or_none=Mock(return_value=previous)),  # find target
        Mock(scalar_one_or_none=Mock(return_value=current))  # find current
    ])
    mock_db_session.commit = AsyncMock()
    mock_db_session.refresh = AsyncMock()

    # Check and rollback
    rollback = await meta_agent.check_and_rollback_if_needed("asset_agent")

    assert rollback is not None
    assert rollback["rolled_back"] == True
    assert rollback["from_version"] == 2
    assert rollback["to_version"] == 1
    assert rollback["degradation_percent"] == 13.0  # 85% - 72%
    assert rollback["reason"] == "automatic_performance_degradation"


@pytest.mark.asyncio
async def test_no_rollback_when_performance_stable(mock_db_session):
    """Test that no rollback occurs when performance is stable."""
    prompt_manager = PromptManager()
    prompt_manager.set_session(mock_db_session)

    meta_agent = MetaAgent(
        agent_id="meta_001",
        db_session=mock_db_session,
        prompt_manager=prompt_manager,
        config={"rollback_threshold": 0.10}
    )

    # Mock versions with stable performance
    current = PromptVersionModel(
        agent_type="coder_agent",
        version=2,
        content="New",
        change_reason="Optimization",
        is_active=True,
        success_rate=88.0
    )

    previous = PromptVersionModel(
        agent_type="coder_agent",
        version=1,
        content="Old",
        change_reason="Initial",
        is_active=False,
        success_rate=85.0
    )

    mock_result = Mock()
    mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[current, previous])))
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    # Check rollback
    rollback = await meta_agent.check_and_rollback_if_needed("coder_agent")

    # No rollback should occur (improvement of 3%, not degradation)
    assert rollback is None


def test_trend_calculation():
    """Test performance trend calculation."""
    pm = PromptManager()

    # Improving trend
    improving_versions = [
        Mock(performance_score=90),
        Mock(performance_score=85),
        Mock(performance_score=80),
        Mock(performance_score=75)
    ]
    assert pm._calculate_trend(improving_versions) == "improving"

    # Degrading trend
    degrading_versions = [
        Mock(performance_score=75),
        Mock(performance_score=80),
        Mock(performance_score=85),
        Mock(performance_score=90)
    ]
    assert pm._calculate_trend(degrading_versions) == "degrading"

    # Stable trend
    stable_versions = [
        Mock(performance_score=85),
        Mock(performance_score=86),
        Mock(performance_score=84),
        Mock(performance_score=85)
    ]
    assert pm._calculate_trend(stable_versions) == "stable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
