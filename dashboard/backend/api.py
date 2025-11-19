"""
Dashboard API - FastAPI Backend for HOTL (Human-On-The-Loop) Interface

This module provides the REST API and WebSocket endpoints for the
Solo-Swarm dashboard, enabling real-time monitoring and human oversight.

Endpoints:
- WebSocket /ws/dashboard: Real-time updates stream
- GET /api/status/agents: Agent pool status
- GET /api/costs/today: Cost analytics
- POST /api/approval/{task_id}: Task approval/rejection
- GET /api/tasks: Task list with filtering
- GET /api/validations: Validation results

Features:
- Async request handling
- WebSocket broadcast for real-time updates
- Database integration with SQLAlchemy
- CORS support for frontend
- Health check endpoint
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from dashboard.backend.dashboard_manager import DashboardManager, UpdateType, DashboardUpdate
from infrastructure.database import (
    DatabaseManager,
    AgentTaskModel,
    CostLogModel,
    ValidationResultModel,
    get_db_session
)


logger = logging.getLogger(__name__)


# Pydantic Models for API Requests/Responses
class ApprovalRequest(BaseModel):
    """Request model for task approval"""
    approved: bool = Field(..., description="Whether to approve (True) or reject (False)")
    comment: Optional[str] = Field(None, description="Optional approval comment")
    approved_by: str = Field(..., description="Username of approver")


class AgentStatus(BaseModel):
    """Response model for agent status"""
    slot_id: int
    status: str  # "idle", "busy", "error"
    current_task: Optional[str] = None
    agent_type: Optional[str] = None


class CostSummary(BaseModel):
    """Response model for cost summary"""
    total_cost_usd: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    by_model: Dict[str, Dict[str, Any]]
    by_operation: Dict[str, Dict[str, Any]]
    period: str  # "today", "week", "month"


class TaskResponse(BaseModel):
    """Response model for task details"""
    id: int
    task_id: str
    task_type: str
    priority: int
    assigned_agent: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    requires_approval: bool
    approved_by: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for validation results"""
    id: int
    asset_path: str
    asset_type: str
    is_valid: bool
    issues: List[str]
    issue_count: int
    validated_at: str
    poly_count: Optional[int] = None
    has_uv_map: Optional[bool] = None


# Initialize FastAPI app
app = FastAPI(
    title="Solo-Swarm Dashboard API",
    description="HOTL (Human-On-The-Loop) Dashboard Backend for Multi-Agent System",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized on startup)
dashboard_manager: Optional[DashboardManager] = None
db_manager: Optional[DatabaseManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize dashboard manager and database on startup."""
    global dashboard_manager, db_manager

    dashboard_manager = DashboardManager()
    logger.info("Dashboard manager initialized")

    # Initialize database (using SQLite for now, can switch to PostgreSQL)
    database_url = "sqlite+aiosqlite:///./solo_swarm.db"
    # For PostgreSQL: "postgresql+asyncpg://user:pass@localhost/solo_swarm"

    db_manager = DatabaseManager(database_url, echo=False)
    await db_manager.initialize()
    logger.info(f"Database initialized: {database_url}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global db_manager

    if db_manager:
        await db_manager.close()
        logger.info("Database closed")


# Dependency for getting database session
async def get_session() -> AsyncSession:
    """Dependency for getting database session."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")

    async with db_manager.get_session() as session:
        yield session


# WebSocket Endpoint
@app.websocket("/ws/dashboard")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time dashboard updates.

    Clients connect to receive live updates about:
    - Agent status changes
    - Task progress
    - Cost updates
    - Validation results
    - Approval requests

    Args:
        websocket: WebSocket connection
        client_id: Optional client identifier for reconnection

    Usage:
        const ws = new WebSocket('ws://localhost:8000/ws/dashboard?client_id=user123');
        ws.onmessage = (event) => {
            const update = JSON.parse(event.data);
            console.log(update);
        };
    """
    if not dashboard_manager:
        await websocket.close(code=1011, reason="Dashboard not initialized")
        return

    await dashboard_manager.connect(websocket, client_id)

    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages from client (e.g., ping/pong, subscriptions)
            data = await websocket.receive_text()

            # Handle client messages (optional)
            # For now, just echo back for heartbeat
            logger.debug(f"Received from client {client_id}: {data}")

    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        dashboard_manager.disconnect(websocket)


# REST Endpoints

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Solo-Swarm Dashboard API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "websocket": "/ws/dashboard",
            "health": "/health",
            "agents": "/api/status/agents",
            "costs": "/api/costs/today",
            "tasks": "/api/tasks",
            "validations": "/api/validations"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if db_manager else "not initialized",
        "websocket_connections": dashboard_manager.get_stats()["active_connections"]
        if dashboard_manager else 0
    }


@app.get("/api/status/agents", response_model=List[AgentStatus])
async def get_agent_status():
    """
    Get current status of all agent slots.

    Returns list of agent slots with their current status.
    For now, returns mocked data with 100 slots.

    Returns:
        List[AgentStatus]: Agent slot statuses
    """
    # Mock data for 100 agent slots
    # In production, this would query actual agent pool status
    agent_slots = []

    for slot_id in range(100):
        # Simulate some busy slots
        if slot_id < 10:
            agent_slots.append(AgentStatus(
                slot_id=slot_id,
                status="busy",
                current_task=f"task_{slot_id:03d}",
                agent_type="coder_agent" if slot_id % 2 == 0 else "asset_agent"
            ))
        elif slot_id < 15:
            agent_slots.append(AgentStatus(
                slot_id=slot_id,
                status="error",
                current_task=f"task_{slot_id:03d}",
                agent_type="verifier_agent"
            ))
        else:
            agent_slots.append(AgentStatus(
                slot_id=slot_id,
                status="idle",
                current_task=None,
                agent_type=None
            ))

    return agent_slots


@app.get("/api/costs/today", response_model=CostSummary)
async def get_costs_today(
    session: AsyncSession = Depends(get_session)
):
    """
    Get cost summary for today.

    Aggregates costs from the cost_logs table for the current day,
    providing breakdowns by model and operation.

    Args:
        session: Database session

    Returns:
        CostSummary: Cost statistics for today
    """
    # Get start of today (UTC)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Query costs for today
    result = await session.execute(
        select(CostLogModel)
        .where(CostLogModel.timestamp >= today_start)
    )
    costs = result.scalars().all()

    # Aggregate totals
    total_cost = sum(cost.cost_usd for cost in costs)
    total_tokens = sum(cost.total_tokens for cost in costs)
    input_tokens = sum(cost.input_tokens for cost in costs)
    output_tokens = sum(cost.output_tokens for cost in costs)
    thinking_tokens = sum(cost.thinking_tokens for cost in costs)

    # Group by model
    by_model: Dict[str, Dict[str, Any]] = {}
    for cost in costs:
        if cost.model not in by_model:
            by_model[cost.model] = {
                "cost_usd": 0.0,
                "tokens": 0,
                "calls": 0
            }
        by_model[cost.model]["cost_usd"] += cost.cost_usd
        by_model[cost.model]["tokens"] += cost.total_tokens
        by_model[cost.model]["calls"] += 1

    # Group by operation
    by_operation: Dict[str, Dict[str, Any]] = {}
    for cost in costs:
        if cost.operation not in by_operation:
            by_operation[cost.operation] = {
                "cost_usd": 0.0,
                "tokens": 0,
                "calls": 0
            }
        by_operation[cost.operation]["cost_usd"] += cost.cost_usd
        by_operation[cost.operation]["tokens"] += cost.total_tokens
        by_operation[cost.operation]["calls"] += 1

    return CostSummary(
        total_cost_usd=round(total_cost, 4),
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        by_model=by_model,
        by_operation=by_operation,
        period="today"
    )


@app.post("/api/approval/{task_id}")
async def approve_task(
    task_id: str,
    approval: ApprovalRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Approve or reject a task that requires human approval.

    Updates the task's approval status and broadcasts the decision
    to all connected dashboard clients.

    Args:
        task_id: Task identifier
        approval: Approval decision and metadata
        session: Database session

    Returns:
        Dict with approval result

    Raises:
        HTTPException: If task not found or doesn't require approval
    """
    # Find task
    result = await session.execute(
        select(AgentTaskModel).where(AgentTaskModel.task_id == task_id)
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if not task.requires_approval:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} does not require approval"
        )

    # Update approval status
    task.approved_by = approval.approved_by
    task.approved_at = datetime.utcnow()

    if approval.approved:
        task.status = "approved"
    else:
        task.status = "rejected"

    await session.commit()

    # Broadcast update
    if dashboard_manager:
        await dashboard_manager.update_task(
            task_id=task_id,
            status=task.status,
            details={
                "approved": approval.approved,
                "approved_by": approval.approved_by,
                "comment": approval.comment
            }
        )

    logger.info(
        f"Task {task_id} {'approved' if approval.approved else 'rejected'} "
        f"by {approval.approved_by}"
    )

    return {
        "task_id": task_id,
        "approved": approval.approved,
        "approved_by": approval.approved_by,
        "approved_at": task.approved_at.isoformat()
    }


@app.get("/api/tasks", response_model=List[TaskResponse])
async def get_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum tasks to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    session: AsyncSession = Depends(get_session)
):
    """
    Get list of tasks with optional filtering.

    Args:
        status: Optional status filter
        limit: Maximum number of tasks to return
        offset: Pagination offset
        session: Database session

    Returns:
        List[TaskResponse]: List of tasks
    """
    # Build query
    query = select(AgentTaskModel)

    if status:
        query = query.where(AgentTaskModel.status == status)

    query = query.order_by(AgentTaskModel.created_at.desc())
    query = query.limit(limit).offset(offset)

    # Execute query
    result = await session.execute(query)
    tasks = result.scalars().all()

    # Convert to response models
    return [
        TaskResponse(
            id=task.id,
            task_id=task.task_id,
            task_type=task.task_type,
            priority=task.priority,
            assigned_agent=task.assigned_agent,
            status=task.status,
            created_at=task.created_at.isoformat(),
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            duration_seconds=task.duration_seconds,
            requires_approval=task.requires_approval,
            approved_by=task.approved_by
        )
        for task in tasks
    ]


@app.get("/api/validations", response_model=List[ValidationResponse])
async def get_validations(
    is_valid: Optional[bool] = Query(None, description="Filter by validation status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    session: AsyncSession = Depends(get_session)
):
    """
    Get list of validation results with optional filtering.

    Args:
        is_valid: Optional validity filter
        limit: Maximum number of results to return
        offset: Pagination offset
        session: Database session

    Returns:
        List[ValidationResponse]: List of validation results
    """
    # Build query
    query = select(ValidationResultModel)

    if is_valid is not None:
        query = query.where(ValidationResultModel.is_valid == is_valid)

    query = query.order_by(ValidationResultModel.validated_at.desc())
    query = query.limit(limit).offset(offset)

    # Execute query
    result = await session.execute(query)
    validations = result.scalars().all()

    # Convert to response models
    return [
        ValidationResponse(
            id=validation.id,
            asset_path=validation.asset_path,
            asset_type=validation.asset_type,
            is_valid=validation.is_valid,
            issues=validation.issues,
            issue_count=validation.issue_count,
            validated_at=validation.validated_at.isoformat(),
            poly_count=validation.poly_count,
            has_uv_map=validation.has_uv_map
        )
        for validation in validations
    ]


@app.get("/api/stats")
async def get_stats(
    session: AsyncSession = Depends(get_session)
):
    """
    Get overall system statistics.

    Returns aggregated stats about tasks, costs, validations, and connections.

    Args:
        session: Database session

    Returns:
        Dict with system statistics
    """
    # Task stats
    task_result = await session.execute(
        select(
            func.count(AgentTaskModel.id).label("total"),
            AgentTaskModel.status
        ).group_by(AgentTaskModel.status)
    )
    task_stats = {row.status: row.total for row in task_result}

    # Cost stats (today)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    cost_result = await session.execute(
        select(
            func.sum(CostLogModel.cost_usd).label("total_cost"),
            func.sum(CostLogModel.total_tokens).label("total_tokens")
        ).where(CostLogModel.timestamp >= today_start)
    )
    cost_row = cost_result.one()

    # Validation stats
    validation_result = await session.execute(
        select(
            func.count(ValidationResultModel.id).label("total"),
            ValidationResultModel.is_valid
        ).group_by(ValidationResultModel.is_valid)
    )
    validation_stats = {
        "passed": 0,
        "failed": 0
    }
    for row in validation_result:
        if row.is_valid:
            validation_stats["passed"] = row.total
        else:
            validation_stats["failed"] = row.total

    # WebSocket stats
    ws_stats = dashboard_manager.get_stats() if dashboard_manager else {}

    return {
        "tasks": task_stats,
        "costs_today": {
            "total_cost_usd": float(cost_row.total_cost or 0.0),
            "total_tokens": int(cost_row.total_tokens or 0)
        },
        "validations": validation_stats,
        "websocket": ws_stats
    }


# For development/testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
