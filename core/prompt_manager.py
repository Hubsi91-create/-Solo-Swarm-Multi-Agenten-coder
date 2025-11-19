"""
Prompt Manager - Version Control for Agent Prompts (Genetic Memory)

This module manages the evolution of agent prompts over time, enabling:
- Versioning of all prompt changes
- Rollback when performance degrades
- Performance tracking per version
- Shadow testing before deployment

This is the "genetic memory" of the system - it remembers what worked and what didn't.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database import PromptVersionModel


logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages versioning and rollback of agent system prompts.

    This class provides the "genetic memory" functionality, allowing the system
    to track prompt evolution and revert to previous versions when needed.

    Features:
    - Automatic version numbering per agent type
    - Only one active version per agent at a time
    - Performance tracking and comparison
    - Rollback to any previous version
    - Shadow testing results storage
    """

    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize the Prompt Manager.

        Args:
            db_session: Optional database session (can be set later)
        """
        self.db_session = db_session
        self._in_memory_cache: Dict[str, str] = {}  # agent_type -> current prompt

        logger.info("PromptManager initialized")

    def set_session(self, session: AsyncSession):
        """Set or update the database session."""
        self.db_session = session

    async def get_current_prompt(self, agent_type: str) -> Optional[str]:
        """
        Get the currently active prompt for an agent type.

        Args:
            agent_type: Type of agent (e.g., "coder_agent", "qa_agent")

        Returns:
            Current prompt string, or None if no active version exists
        """
        if not self.db_session:
            raise RuntimeError("Database session not set. Call set_session() first.")

        # Query for active prompt version
        result = await self.db_session.execute(
            select(PromptVersionModel)
            .where(and_(
                PromptVersionModel.agent_type == agent_type,
                PromptVersionModel.is_active == True
            ))
        )
        active_version = result.scalar_one_or_none()

        if active_version:
            logger.debug(
                f"Retrieved active prompt for {agent_type}: "
                f"version {active_version.version}"
            )
            return active_version.content
        else:
            logger.warning(f"No active prompt found for {agent_type}")
            return None

    async def get_version(self, agent_type: str, version: int) -> Optional[PromptVersionModel]:
        """
        Get a specific version of a prompt.

        Args:
            agent_type: Type of agent
            version: Version number

        Returns:
            PromptVersionModel or None if not found
        """
        if not self.db_session:
            raise RuntimeError("Database session not set")

        result = await self.db_session.execute(
            select(PromptVersionModel)
            .where(and_(
                PromptVersionModel.agent_type == agent_type,
                PromptVersionModel.version == version
            ))
        )
        return result.scalar_one_or_none()

    async def update_prompt(
        self,
        agent_type: str,
        new_content: str,
        change_reason: str,
        changed_by: str = "meta_agent",
        performance_metrics: Optional[Dict[str, float]] = None,
        shadow_test_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptVersionModel:
        """
        Create a new prompt version and activate it.

        This method:
        1. Deactivates the current active version
        2. Creates a new version with incremented version number
        3. Activates the new version
        4. Stores performance metrics and shadow test results

        Args:
            agent_type: Type of agent
            new_content: New prompt content
            change_reason: Explanation for the change
            changed_by: Who/what made the change
            performance_metrics: Optional dict with success_rate, avg_cost, avg_duration
            shadow_test_results: Optional shadow testing results
            metadata: Optional additional metadata

        Returns:
            The newly created PromptVersionModel
        """
        if not self.db_session:
            raise RuntimeError("Database session not set")

        logger.info(f"🧬 Updating prompt for {agent_type}: {change_reason}")

        # Step 1: Get current max version number
        result = await self.db_session.execute(
            select(func.max(PromptVersionModel.version))
            .where(PromptVersionModel.agent_type == agent_type)
        )
        max_version = result.scalar()
        new_version = (max_version or 0) + 1

        # Step 2: Deactivate current active version
        result = await self.db_session.execute(
            select(PromptVersionModel)
            .where(and_(
                PromptVersionModel.agent_type == agent_type,
                PromptVersionModel.is_active == True
            ))
        )
        current_active = result.scalar_one_or_none()

        if current_active:
            current_active.is_active = False
            current_active.deactivated_at = datetime.utcnow()
            logger.info(
                f"  Deactivated version {current_active.version} "
                f"(performance: {current_active.performance_score})"
            )

        # Step 3: Create new version
        new_prompt_version = PromptVersionModel(
            agent_type=agent_type,
            version=new_version,
            content=new_content,
            change_reason=change_reason,
            changed_by=changed_by,
            is_active=True,
            activated_at=datetime.utcnow(),
            metadata=metadata or {}
        )

        # Add performance metrics if provided
        if performance_metrics:
            new_prompt_version.performance_score = performance_metrics.get('performance_score')
            new_prompt_version.success_rate = performance_metrics.get('success_rate')
            new_prompt_version.avg_cost = performance_metrics.get('avg_cost')
            new_prompt_version.avg_duration = performance_metrics.get('avg_duration')

        # Add shadow test results if provided
        if shadow_test_results:
            new_prompt_version.shadow_test_count = shadow_test_results.get('test_count')
            new_prompt_version.shadow_test_success_rate = shadow_test_results.get('success_rate')

        # Add to session and commit
        self.db_session.add(new_prompt_version)
        await self.db_session.commit()
        await self.db_session.refresh(new_prompt_version)

        logger.info(
            f"✅ Created and activated version {new_version} for {agent_type} "
            f"(score: {new_prompt_version.performance_score})"
        )

        # Update cache
        self._in_memory_cache[agent_type] = new_content

        return new_prompt_version

    async def rollback_prompt(
        self,
        agent_type: str,
        target_version: int,
        reason: str = "Performance regression detected"
    ) -> PromptVersionModel:
        """
        Rollback to a previous prompt version.

        This deactivates the current version and reactivates a previous one.

        Args:
            agent_type: Type of agent
            target_version: Version number to rollback to
            reason: Reason for rollback

        Returns:
            The reactivated PromptVersionModel

        Raises:
            ValueError: If target version doesn't exist
        """
        if not self.db_session:
            raise RuntimeError("Database session not set")

        logger.warning(f"⚠️  Rolling back {agent_type} to version {target_version}: {reason}")

        # Step 1: Find target version
        target = await self.get_version(agent_type, target_version)

        if not target:
            raise ValueError(
                f"Version {target_version} not found for {agent_type}"
            )

        # Step 2: Deactivate current active version
        result = await self.db_session.execute(
            select(PromptVersionModel)
            .where(and_(
                PromptVersionModel.agent_type == agent_type,
                PromptVersionModel.is_active == True
            ))
        )
        current_active = result.scalar_one_or_none()

        if current_active:
            current_active.is_active = False
            current_active.deactivated_at = datetime.utcnow()
            logger.info(
                f"  Deactivated version {current_active.version} "
                f"(reason: {reason})"
            )

        # Step 3: Reactivate target version
        target.is_active = True
        target.activated_at = datetime.utcnow()

        # Update metadata to track rollback
        if not target.metadata:
            target.metadata = {}
        target.metadata['last_rollback'] = {
            'from_version': current_active.version if current_active else None,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.db_session.commit()
        await self.db_session.refresh(target)

        logger.info(
            f"✅ Rolled back {agent_type} to version {target_version} "
            f"(score: {target.performance_score})"
        )

        # Update cache
        self._in_memory_cache[agent_type] = target.content

        return target

    async def get_version_history(
        self,
        agent_type: str,
        limit: int = 50
    ) -> List[PromptVersionModel]:
        """
        Get version history for an agent type.

        Args:
            agent_type: Type of agent
            limit: Maximum number of versions to return

        Returns:
            List of PromptVersionModel, ordered by version descending
        """
        if not self.db_session:
            raise RuntimeError("Database session not set")

        result = await self.db_session.execute(
            select(PromptVersionModel)
            .where(PromptVersionModel.agent_type == agent_type)
            .order_by(desc(PromptVersionModel.version))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_performance_comparison(
        self,
        agent_type: str
    ) -> Dict[str, Any]:
        """
        Compare performance metrics across versions.

        Args:
            agent_type: Type of agent

        Returns:
            Dictionary with performance comparison data
        """
        if not self.db_session:
            raise RuntimeError("Database session not set")

        history = await self.get_version_history(agent_type, limit=10)

        if not history:
            return {"error": "No version history found"}

        # Get current (active) version
        current = next((v for v in history if v.is_active), None)

        # Calculate statistics
        versions_with_scores = [v for v in history if v.performance_score is not None]

        if not versions_with_scores:
            return {
                "current_version": current.version if current else None,
                "total_versions": len(history),
                "error": "No performance scores available"
            }

        best_version = max(versions_with_scores, key=lambda v: v.performance_score)
        worst_version = min(versions_with_scores, key=lambda v: v.performance_score)

        avg_score = sum(v.performance_score for v in versions_with_scores) / len(versions_with_scores)

        return {
            "agent_type": agent_type,
            "current_version": {
                "version": current.version if current else None,
                "score": current.performance_score if current else None,
                "success_rate": current.success_rate if current else None,
                "is_best": (current.version == best_version.version) if current else False
            },
            "best_version": {
                "version": best_version.version,
                "score": best_version.performance_score,
                "success_rate": best_version.success_rate,
                "change_reason": best_version.change_reason
            },
            "worst_version": {
                "version": worst_version.version,
                "score": worst_version.performance_score
            },
            "statistics": {
                "total_versions": len(history),
                "avg_score": round(avg_score, 2),
                "score_range": round(best_version.performance_score - worst_version.performance_score, 2)
            },
            "recent_trend": self._calculate_trend(versions_with_scores[:5])
        }

    def _calculate_trend(self, versions: List[PromptVersionModel]) -> str:
        """Calculate performance trend from recent versions."""
        if len(versions) < 2:
            return "insufficient_data"

        scores = [v.performance_score for v in versions if v.performance_score is not None]

        if len(scores) < 2:
            return "insufficient_data"

        # Simple trend: compare first half average to second half average
        mid = len(scores) // 2
        first_half_avg = sum(scores[:mid]) / mid
        second_half_avg = sum(scores[mid:]) / (len(scores) - mid)

        diff = second_half_avg - first_half_avg

        if diff > 2:
            return "improving"
        elif diff < -2:
            return "degrading"
        else:
            return "stable"

    async def create_initial_version(
        self,
        agent_type: str,
        initial_content: str,
        description: str = "Initial prompt version"
    ) -> PromptVersionModel:
        """
        Create the initial version for a new agent type.

        Args:
            agent_type: Type of agent
            initial_content: Initial prompt content
            description: Description of the initial version

        Returns:
            The created PromptVersionModel
        """
        return await self.update_prompt(
            agent_type=agent_type,
            new_content=initial_content,
            change_reason=description,
            changed_by="system",
            metadata={"is_initial": True}
        )

    async def update_performance_metrics(
        self,
        agent_type: str,
        version: int,
        performance_metrics: Dict[str, float]
    ) -> bool:
        """
        Update performance metrics for a specific version.

        This is typically called after running shadow tests or collecting
        real-world performance data.

        Args:
            agent_type: Type of agent
            version: Version number
            performance_metrics: Dict with metrics to update

        Returns:
            True if successful, False if version not found
        """
        if not self.db_session:
            raise RuntimeError("Database session not set")

        prompt_version = await self.get_version(agent_type, version)

        if not prompt_version:
            logger.warning(
                f"Cannot update metrics: version {version} not found for {agent_type}"
            )
            return False

        # Update metrics
        if 'performance_score' in performance_metrics:
            prompt_version.performance_score = performance_metrics['performance_score']
        if 'success_rate' in performance_metrics:
            prompt_version.success_rate = performance_metrics['success_rate']
        if 'avg_cost' in performance_metrics:
            prompt_version.avg_cost = performance_metrics['avg_cost']
        if 'avg_duration' in performance_metrics:
            prompt_version.avg_duration = performance_metrics['avg_duration']

        await self.db_session.commit()

        logger.info(
            f"Updated performance metrics for {agent_type} v{version}: "
            f"score={prompt_version.performance_score}"
        )

        return True
