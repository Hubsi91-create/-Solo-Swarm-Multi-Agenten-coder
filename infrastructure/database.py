"""
Database Layer - Async SQLAlchemy Models and Session Management

This module provides async database connectivity and ORM models for the
Solo-Swarm Multi-Agent System dashboard and monitoring.

Models:
- AgentTaskModel: Tracks tasks assigned to agents
- CostLogModel: Records API costs and token usage
- ValidationResultModel: Stores validation results from verifier agents

Database Support:
- PostgreSQL (production)
- SQLite with aiosqlite (testing)
"""

from datetime import datetime
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import String, Integer, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Base class for all models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    pass


class AgentTaskModel(Base):
    """
    Model for tracking agent tasks in the system.

    Tracks task lifecycle from creation through completion,
    including agent assignment, status, and results.
    """
    __tablename__ = "agent_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)

    assigned_agent: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    agent_slot: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)

    # Task data
    context: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    requirements: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Metrics
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Error tracking
    error_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Human-in-the-loop
    requires_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approved_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"AgentTaskModel(id={self.id}, task_id='{self.task_id}', "
            f"type='{self.task_type}', status='{self.status}')"
        )


class CostLogModel(Base):
    """
    Model for tracking API costs and token usage.

    Records every API call with token counts and costs,
    enabling cost monitoring and budget control.
    """
    __tablename__ = "cost_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # API details
    model: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)

    # Token usage
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    thinking_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    # Cost
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)

    # Context
    task_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"CostLogModel(id={self.id}, model='{self.model}', "
            f"tokens={self.total_tokens}, cost=${self.cost_usd:.4f})"
        )


class ValidationResultModel(Base):
    """
    Model for storing asset validation results.

    Tracks validation checks performed by verifier agents,
    including pass/fail status and specific issues found.
    """
    __tablename__ = "validation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Asset details
    asset_path: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Validation status
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)

    # Issues found
    issues: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    issue_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Constraints checked
    constraints: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Metrics (for 3D assets)
    poly_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    has_uv_map: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    dimensions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Task association
    task_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Timestamp
    validated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Retry tracking
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return (
            f"ValidationResultModel(id={self.id}, asset='{self.asset_path}', "
            f"valid={self.is_valid}, issues={self.issue_count})"
        )


class ApprovalRequestModel(Base):
    """
    Model for tracking approval requests from Closed Loop QA.

    Stores approval requests that require human review,
    including cycle details and approval decisions.
    """
    __tablename__ = "approval_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Cycle identification
    cycle_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    request_type: Mapped[str] = mapped_column(String(50), nullable=False, default="closed_loop_qa")

    # Request details
    report_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Test results summary
    tests_passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    test_coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    iterations: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    # Possible values: pending, approved, rejected

    # Approval tracking
    approved_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"ApprovalRequestModel(id={self.id}, cycle_id='{self.cycle_id}', "
            f"status='{self.status}')"
        )


class PromptVersionModel(Base):
    """
    Model for versioning system prompts - the "genetic memory" of agents.

    Tracks prompt evolution over time, enabling rollback when performance degrades.
    This is critical for the Meta-Agent's self-optimization capabilities.
    """
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Agent identification
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    # Examples: "coder_agent", "qa_agent", "architect_agent", "meta_agent"

    # Version tracking
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Version number (1, 2, 3, ...) - auto-incremented per agent_type

    # Prompt content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # The actual prompt text (can be large)

    # Change tracking
    change_reason: Mapped[str] = mapped_column(Text, nullable=False)
    # Why this version was created (e.g., "Meta-Agent optimization: +15% success rate")

    changed_by: Mapped[str] = mapped_column(String(100), nullable=False, default="meta_agent")
    # Who/what made the change (meta_agent, manual, initial)

    # Performance metrics
    performance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Overall performance score (0-100) based on success rate, cost, duration

    success_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Percentage of successful tasks (0-100)

    avg_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Average cost per task in USD

    avg_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Average task duration in seconds

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    # Only one version per agent_type should be active at a time

    # Shadow testing results (before deployment)
    shadow_test_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shadow_test_success_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deactivated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Additional context (optimizer settings, test results, etc.)

    def __repr__(self) -> str:
        return (
            f"PromptVersionModel(id={self.id}, agent='{self.agent_type}', "
            f"version={self.version}, active={self.is_active})"
        )


class DatabaseManager:
    """
    Async Database Manager for Solo-Swarm System.

    Provides async session management and connection handling
    for both PostgreSQL (production) and SQLite (testing).

    Usage:
        async with DatabaseManager("sqlite+aiosqlite:///test.db") as db_manager:
            async with db_manager.get_session() as session:
                result = await session.execute(select(AgentTaskModel))
    """

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
                Examples:
                - "postgresql+asyncpg://user:pass@localhost/dbname" (production)
                - "sqlite+aiosqlite:///./test.db" (testing)
                - "sqlite+aiosqlite:///:memory:" (in-memory testing)
            echo: Whether to echo SQL statements (for debugging)
        """
        self.database_url = database_url
        self.echo = echo
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def initialize(self) -> None:
        """
        Initialize database engine and create tables.

        Creates the async engine and session factory, then creates
        all tables if they don't exist.
        """
        # Create async engine
        # Note: SQLite doesn't support pool_size/max_overflow arguments
        engine_kwargs = {
            "echo": self.echo,
        }

        # Only add pool arguments for non-SQLite databases
        if not self.database_url.startswith("sqlite"):
            engine_kwargs.update({
                "pool_pre_ping": True,  # Verify connections before using
                "pool_size": 5,
                "max_overflow": 10
            })

        self.engine = create_async_engine(self.database_url, **engine_kwargs)

        # Create session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database engine and cleanup resources."""
        if self.engine:
            await self.engine.dispose()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.

        Usage:
            async with db_manager.get_session() as session:
                result = await session.execute(select(AgentTaskModel))

        Yields:
            AsyncSession: Database session
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Dependency injection for FastAPI
async def get_db_session(
    db_manager: DatabaseManager
) -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage in FastAPI:
        @app.get("/api/tasks")
        async def get_tasks(session: AsyncSession = Depends(get_db_session)):
            result = await session.execute(select(AgentTaskModel))
            return result.scalars().all()

    Args:
        db_manager: DatabaseManager instance

    Yields:
        AsyncSession: Database session
    """
    async with db_manager.get_session() as session:
        yield session
