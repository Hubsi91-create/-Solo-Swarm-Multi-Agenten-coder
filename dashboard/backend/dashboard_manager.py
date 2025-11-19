"""
Dashboard Manager - WebSocket Broadcast and Real-Time Updates

This module manages WebSocket connections for the HOTL (Human-On-The-Loop)
dashboard, providing real-time updates for:
- Agent status changes
- Task progress
- Cost updates
- Validation results

Features:
- Connection management with automatic cleanup
- Broadcast to all connected clients
- Selective updates (by channel/topic)
- Connection drop handling
- Message queuing for disconnected clients
"""

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect


logger = logging.getLogger(__name__)


class UpdateType(str, Enum):
    """Types of dashboard updates"""
    AGENT_STATUS = "agent_status"
    TASK_UPDATE = "task_update"
    COST_UPDATE = "cost_update"
    VALIDATION_RESULT = "validation_result"
    SYSTEM_STATUS = "system_status"
    APPROVAL_REQUEST = "approval_request"


@dataclass
class DashboardUpdate:
    """
    Structure for dashboard update messages.

    Attributes:
        update_type: Type of update
        data: Update payload
        timestamp: When the update occurred
        priority: Update priority (0=low, 1=normal, 2=high)
    """
    update_type: UpdateType
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "update_type": self.update_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class ConnectionManager:
    """
    Manages WebSocket connections for the dashboard.

    Handles connection lifecycle, message broadcasting, and
    graceful handling of connection drops.

    Usage:
        manager = ConnectionManager()

        @app.websocket("/ws/dashboard")
        async def websocket_endpoint(websocket: WebSocket):
            await manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming messages
            except WebSocketDisconnect:
                manager.disconnect(websocket)
    """

    def __init__(self):
        """Initialize the connection manager."""
        # Active WebSocket connections
        self.active_connections: Set[WebSocket] = set()

        # Connection metadata (for tracking clients)
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

        # Message queue for reconnecting clients (optional)
        self.message_queue: Dict[str, List[DashboardUpdate]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info("ConnectionManager initialized")

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None
    ) -> None:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
            client_id: Optional client identifier for reconnection support
        """
        await websocket.accept()

        async with self._lock:
            self.active_connections.add(websocket)

            # Store metadata
            self.connection_metadata[websocket] = {
                "client_id": client_id,
                "connected_at": datetime.utcnow(),
                "message_count": 0
            }

        logger.info(
            f"WebSocket connected: {client_id or 'anonymous'} "
            f"(total: {len(self.active_connections)})"
        )

        # Send welcome message
        welcome = DashboardUpdate(
            update_type=UpdateType.SYSTEM_STATUS,
            data={
                "status": "connected",
                "message": "Connected to Solo-Swarm Dashboard",
                "client_id": client_id,
                "active_connections": len(self.active_connections)
            },
            timestamp=datetime.utcnow()
        )
        await self._send_to_connection(websocket, welcome)

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

            # Get client info before removing metadata
            metadata = self.connection_metadata.get(websocket, {})
            client_id = metadata.get("client_id", "anonymous")

            # Remove metadata
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]

            logger.info(
                f"WebSocket disconnected: {client_id} "
                f"(remaining: {len(self.active_connections)})"
            )

    async def broadcast(
        self,
        update: DashboardUpdate,
        exclude: Optional[Set[WebSocket]] = None
    ) -> int:
        """
        Broadcast an update to all connected clients.

        Args:
            update: Dashboard update to broadcast
            exclude: Optional set of connections to exclude

        Returns:
            Number of clients that received the message
        """
        exclude = exclude or set()
        successful_sends = 0
        failed_connections = []

        # Get snapshot of connections
        async with self._lock:
            connections = self.active_connections.copy()

        # Send to all connections
        for connection in connections:
            if connection not in exclude:
                try:
                    await self._send_to_connection(connection, update)
                    successful_sends += 1

                    # Update message count
                    if connection in self.connection_metadata:
                        self.connection_metadata[connection]["message_count"] += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to send to connection: {e}. "
                        "Connection will be removed."
                    )
                    failed_connections.append(connection)

        # Cleanup failed connections
        for connection in failed_connections:
            self.disconnect(connection)

        logger.debug(
            f"Broadcast: {update.update_type.value} sent to "
            f"{successful_sends}/{len(connections)} clients"
        )

        return successful_sends

    async def send_to_client(
        self,
        client_id: str,
        update: DashboardUpdate
    ) -> bool:
        """
        Send an update to a specific client.

        Args:
            client_id: Client identifier
            update: Dashboard update to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        # Find connection by client_id
        target_connection = None

        async with self._lock:
            for connection, metadata in self.connection_metadata.items():
                if metadata.get("client_id") == client_id:
                    target_connection = connection
                    break

        if not target_connection:
            logger.warning(f"Client {client_id} not found")
            return False

        try:
            await self._send_to_connection(target_connection, update)
            return True
        except Exception as e:
            logger.error(f"Failed to send to client {client_id}: {e}")
            self.disconnect(target_connection)
            return False

    async def _send_to_connection(
        self,
        websocket: WebSocket,
        update: DashboardUpdate
    ) -> None:
        """
        Send an update to a specific WebSocket connection.

        Args:
            websocket: WebSocket connection
            update: Dashboard update to send

        Raises:
            Exception: If send fails (connection will be cleaned up)
        """
        message = update.to_json()
        await websocket.send_text(message)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Dictionary with connection stats
        """
        total_messages = sum(
            metadata.get("message_count", 0)
            for metadata in self.connection_metadata.values()
        )

        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": total_messages,
            "clients": [
                {
                    "client_id": metadata.get("client_id", "anonymous"),
                    "connected_at": metadata.get("connected_at").isoformat()
                    if metadata.get("connected_at") else None,
                    "message_count": metadata.get("message_count", 0)
                }
                for metadata in self.connection_metadata.values()
            ]
        }


class DashboardManager:
    """
    High-level Dashboard Manager.

    Provides convenience methods for sending specific types of updates
    to the dashboard, abstracting away the WebSocket details.

    Usage:
        dashboard = DashboardManager()

        # Update agent status
        await dashboard.update_agent_status("agent_001", "busy", {
            "current_task": "impl_001",
            "progress": 50
        })

        # Send cost update
        await dashboard.update_costs(5.23, 125000)
    """

    def __init__(self):
        """Initialize the dashboard manager."""
        self.connection_manager = ConnectionManager()
        logger.info("DashboardManager initialized")

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None
    ) -> None:
        """
        Connect a new WebSocket client.

        Args:
            websocket: WebSocket connection
            client_id: Optional client identifier
        """
        await self.connection_manager.connect(websocket, client_id)

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect a WebSocket client.

        Args:
            websocket: WebSocket connection to disconnect
        """
        self.connection_manager.disconnect(websocket)

    async def update_agent_status(
        self,
        agent_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast an agent status update.

        Args:
            agent_id: Agent identifier
            status: New status (e.g., "idle", "busy", "error")
            details: Optional additional details

        Returns:
            Number of clients that received the update
        """
        update = DashboardUpdate(
            update_type=UpdateType.AGENT_STATUS,
            data={
                "agent_id": agent_id,
                "status": status,
                "details": details or {},
                "updated_at": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            priority=1
        )

        return await self.connection_manager.broadcast(update)

    async def update_task(
        self,
        task_id: str,
        status: str,
        progress: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast a task update.

        Args:
            task_id: Task identifier
            status: Task status
            progress: Optional progress percentage (0-100)
            details: Optional additional details

        Returns:
            Number of clients that received the update
        """
        update = DashboardUpdate(
            update_type=UpdateType.TASK_UPDATE,
            data={
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "details": details or {},
                "updated_at": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            priority=1
        )

        return await self.connection_manager.broadcast(update)

    async def update_costs(
        self,
        total_cost_usd: float,
        total_tokens: int,
        breakdown: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast a cost update.

        Args:
            total_cost_usd: Total cost in USD
            total_tokens: Total token count
            breakdown: Optional cost breakdown by model/operation

        Returns:
            Number of clients that received the update
        """
        update = DashboardUpdate(
            update_type=UpdateType.COST_UPDATE,
            data={
                "total_cost_usd": total_cost_usd,
                "total_tokens": total_tokens,
                "breakdown": breakdown or {},
                "updated_at": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            priority=1
        )

        return await self.connection_manager.broadcast(update)

    async def send_validation_result(
        self,
        asset_path: str,
        is_valid: bool,
        issues: List[str],
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast a validation result.

        Args:
            asset_path: Path to validated asset
            is_valid: Whether validation passed
            issues: List of issues found
            details: Optional additional details

        Returns:
            Number of clients that received the update
        """
        update = DashboardUpdate(
            update_type=UpdateType.VALIDATION_RESULT,
            data={
                "asset_path": asset_path,
                "is_valid": is_valid,
                "issues": issues,
                "issue_count": len(issues),
                "details": details or {},
                "validated_at": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            priority=2 if not is_valid else 1  # Higher priority for failures
        )

        return await self.connection_manager.broadcast(update)

    async def request_approval(
        self,
        task_id: str,
        task_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast an approval request to dashboard users.

        Args:
            task_id: Task identifier requiring approval
            task_type: Type of task
            description: Human-readable description
            metadata: Optional additional metadata

        Returns:
            Number of clients that received the request
        """
        update = DashboardUpdate(
            update_type=UpdateType.APPROVAL_REQUEST,
            data={
                "task_id": task_id,
                "task_type": task_type,
                "description": description,
                "metadata": metadata or {},
                "requested_at": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            priority=2  # High priority for approval requests
        )

        return await self.connection_manager.broadcast(update)

    async def broadcast(self, update: DashboardUpdate) -> int:
        """
        Broadcast a custom update.

        Args:
            update: Dashboard update to broadcast

        Returns:
            Number of clients that received the update
        """
        return await self.connection_manager.broadcast(update)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dashboard statistics.

        Returns:
            Dictionary with dashboard stats
        """
        return self.connection_manager.get_stats()
