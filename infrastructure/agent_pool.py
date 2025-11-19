"""
Agent Pool Manager - Resource Allocation for Agent Execution
Manages slot allocation to respect API rate limits (Google AI Ultra: 100 concurrent requests)
"""

import threading
from typing import Dict, Optional, Set
from datetime import datetime
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class SlotAllocation:
    """Information about a slot allocation"""
    slot_id: str
    agent_type: str
    allocated_at: datetime
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SlotAllocationError(Exception):
    """Raised when slot allocation fails"""
    pass


class AgentPoolManager:
    """
    Manages agent execution slots to respect API rate limits.

    The Google AI Ultra plan has a limit of 100 concurrent requests.
    This class ensures we never exceed that limit by managing slot allocations.
    """

    # Google AI Ultra concurrent request limit
    DEFAULT_MAX_SLOTS = 100

    def __init__(self, max_slots: int = DEFAULT_MAX_SLOTS):
        """
        Initialize the Agent Pool Manager.

        Args:
            max_slots: Maximum number of concurrent slots (default: 100)
        """
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")

        self.max_slots = max_slots
        self.active_slots: Dict[str, SlotAllocation] = {}
        self.allocation_history: list = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.total_allocations = 0
        self.total_releases = 0
        self.allocation_failures = 0

        logger.info(f"AgentPoolManager initialized with {max_slots} max slots")

    @property
    def available_slots(self) -> int:
        """Get number of available slots"""
        with self._lock:
            return self.max_slots - len(self.active_slots)

    @property
    def used_slots(self) -> int:
        """Get number of currently used slots"""
        with self._lock:
            return len(self.active_slots)

    @property
    def utilization_percentage(self) -> float:
        """Get slot utilization as percentage"""
        with self._lock:
            return (len(self.active_slots) / self.max_slots) * 100

    def allocate_slot(
        self,
        agent_type: str,
        slot_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Allocate a slot for an agent.

        Args:
            agent_type: Type of agent requesting the slot
            slot_id: Optional specific slot ID (auto-generated if None)
            metadata: Optional metadata to attach to the allocation

        Returns:
            Allocated slot ID

        Raises:
            SlotAllocationError: If no slots are available
        """
        with self._lock:
            # Check if slots are available
            if len(self.active_slots) >= self.max_slots:
                self.allocation_failures += 1
                raise SlotAllocationError(
                    f"No slots available. All {self.max_slots} slots are in use. "
                    f"Current utilization: {self.utilization_percentage:.1f}%"
                )

            # Generate slot ID if not provided
            if slot_id is None:
                slot_id = self._generate_slot_id(agent_type)

            # Check if slot_id already exists
            if slot_id in self.active_slots:
                raise SlotAllocationError(f"Slot {slot_id} is already allocated")

            # Create allocation
            allocation = SlotAllocation(
                slot_id=slot_id,
                agent_type=agent_type,
                allocated_at=datetime.utcnow(),
                metadata=metadata or {}
            )

            # Store allocation
            self.active_slots[slot_id] = allocation
            self.allocation_history.append({
                "action": "allocate",
                "slot_id": slot_id,
                "agent_type": agent_type,
                "timestamp": allocation.allocated_at.isoformat()
            })
            self.total_allocations += 1

            logger.info(
                f"Allocated slot {slot_id} for {agent_type} "
                f"({self.used_slots}/{self.max_slots} slots in use)"
            )

            return slot_id

    def release_slot(self, slot_id: str) -> bool:
        """
        Release a previously allocated slot.

        Args:
            slot_id: Slot ID to release

        Returns:
            True if slot was released, False if slot was not found

        Raises:
            ValueError: If slot_id is None or empty
        """
        if not slot_id:
            raise ValueError("slot_id cannot be None or empty")

        with self._lock:
            if slot_id not in self.active_slots:
                logger.warning(f"Attempted to release non-existent slot: {slot_id}")
                return False

            # Get allocation info before removing
            allocation = self.active_slots[slot_id]

            # Remove from active slots
            del self.active_slots[slot_id]

            # Record in history
            self.allocation_history.append({
                "action": "release",
                "slot_id": slot_id,
                "agent_type": allocation.agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "duration_seconds": (
                    datetime.utcnow() - allocation.allocated_at
                ).total_seconds()
            })
            self.total_releases += 1

            logger.info(
                f"Released slot {slot_id} (agent: {allocation.agent_type}) "
                f"({self.used_slots}/{self.max_slots} slots in use)"
            )

            return True

    def is_slot_available(self) -> bool:
        """
        Check if at least one slot is available.

        Returns:
            True if a slot can be allocated, False otherwise
        """
        with self._lock:
            return len(self.active_slots) < self.max_slots

    def get_slot_info(self, slot_id: str) -> Optional[SlotAllocation]:
        """
        Get information about a specific slot.

        Args:
            slot_id: Slot ID to query

        Returns:
            SlotAllocation object or None if not found
        """
        with self._lock:
            return self.active_slots.get(slot_id)

    def get_active_slots(self) -> Dict[str, SlotAllocation]:
        """
        Get all currently active slot allocations.

        Returns:
            Dictionary of slot_id -> SlotAllocation
        """
        with self._lock:
            return self.active_slots.copy()

    def get_slots_by_agent_type(self, agent_type: str) -> Dict[str, SlotAllocation]:
        """
        Get all active slots for a specific agent type.

        Args:
            agent_type: Agent type to filter by

        Returns:
            Dictionary of slot_id -> SlotAllocation for matching agent type
        """
        with self._lock:
            return {
                slot_id: allocation
                for slot_id, allocation in self.active_slots.items()
                if allocation.agent_type == agent_type
            }

    def clear_all_slots(self) -> int:
        """
        Clear all allocated slots (use with caution).

        Returns:
            Number of slots that were cleared
        """
        with self._lock:
            count = len(self.active_slots)
            self.active_slots.clear()

            self.allocation_history.append({
                "action": "clear_all",
                "timestamp": datetime.utcnow().isoformat(),
                "cleared_count": count
            })

            logger.warning(f"Cleared all {count} active slots")
            return count

    def get_statistics(self) -> Dict:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            return {
                "max_slots": self.max_slots,
                "used_slots": self.used_slots,
                "available_slots": self.available_slots,
                "utilization_percentage": round(self.utilization_percentage, 2),
                "total_allocations": self.total_allocations,
                "total_releases": self.total_releases,
                "allocation_failures": self.allocation_failures,
                "active_agent_types": self._get_active_agent_types()
            }

    def _get_active_agent_types(self) -> Dict[str, int]:
        """Get count of active slots per agent type"""
        agent_type_counts = {}
        for allocation in self.active_slots.values():
            agent_type_counts[allocation.agent_type] = (
                agent_type_counts.get(allocation.agent_type, 0) + 1
            )
        return agent_type_counts

    def _generate_slot_id(self, agent_type: str) -> str:
        """Generate a unique slot ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{agent_type}_{timestamp}_{self.total_allocations}"

    def wait_for_slot(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for a slot to become available.

        Note: This is a simple polling implementation.
        For production, consider using a proper event-based approach.

        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            True if a slot became available, False if timeout occurred
        """
        import time

        start_time = time.time()

        while not self.is_slot_available():
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False
            time.sleep(0.1)  # Poll every 100ms

        return True

    def __repr__(self) -> str:
        return (
            f"AgentPoolManager("
            f"max_slots={self.max_slots}, "
            f"used={self.used_slots}, "
            f"available={self.available_slots}, "
            f"utilization={self.utilization_percentage:.1f}%)"
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear all slots"""
        self.clear_all_slots()
