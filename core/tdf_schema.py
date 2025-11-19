"""
Task Definition Schema - Pydantic Models for Task Management
Implements strict validation using Pydantic 2.5+
"""

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue


class TaskType(str, Enum):
    """Enumeration of available task types in the Solo-Swarm system"""

    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"
    REFACTORING = "refactoring"


class TaskDefinition(BaseModel):
    """
    Core Task Definition Model with strict validation.

    This model ensures all tasks in the Solo-Swarm system follow a consistent
    structure and are properly validated before execution.
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        use_enum_values=False,
        json_schema_extra={
            "examples": [
                {
                    "task_id": "task_001",
                    "task_type": "implementation",
                    "priority": 1,
                    "assigned_agent": "coder_agent",
                    "context": {"language": "python", "framework": "fastapi"},
                    "requirements": {"timeout": 300, "max_retries": 3}
                }
            ]
        }
    )

    task_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the task",
        pattern=r"^[a-zA-Z0-9_-]+$"
    )

    task_type: TaskType = Field(
        ...,
        description="Type of task to be executed"
    )

    priority: int = Field(
        ...,
        ge=1,
        le=10,
        description="Task priority (1=highest, 10=lowest)"
    )

    assigned_agent: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name/ID of the agent assigned to this task",
        pattern=r"^[a-zA-Z0-9_-]+$"
    )

    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual information needed for task execution"
    )

    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific requirements and constraints for the task"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when task was created"
    )

    status: str = Field(
        default="pending",
        description="Current status of the task",
        pattern=r"^(pending|in_progress|completed|failed|cancelled)$"
    )

    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task execution result (populated after completion)"
    )

    @field_validator('context')
    @classmethod
    def validate_context(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure context is a valid dictionary"""
        if not isinstance(v, dict):
            raise ValueError("context must be a dictionary")
        return v

    @field_validator('requirements')
    @classmethod
    def validate_requirements(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure requirements is a valid dictionary"""
        if not isinstance(v, dict):
            raise ValueError("requirements must be a dictionary")
        return v

    @model_validator(mode='after')
    def validate_task_consistency(self) -> 'TaskDefinition':
        """
        Perform cross-field validation to ensure task consistency.
        High-priority tasks should have proper requirements set.
        """
        if self.priority <= 3 and not self.requirements:
            # High priority tasks should have requirements defined
            pass  # Warning only, not enforcing

        return self

    def to_strict_json(self) -> Dict[str, Any]:
        """
        Export to strict JSON format with all fields validated.
        Uses mode='json' to ensure datetime serialization.
        """
        return self.model_dump(mode='json', exclude_none=False)

    def update_status(self, new_status: str) -> None:
        """
        Update task status with validation.

        Args:
            new_status: New status value (must match pattern)

        Raises:
            ValueError: If status is invalid
        """
        valid_statuses = {"pending", "in_progress", "completed", "failed", "cancelled"}
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status: {new_status}. Must be one of {valid_statuses}")
        self.status = new_status

    def set_result(self, result: Dict[str, Any]) -> None:
        """
        Set task result and mark as completed.

        Args:
            result: Dictionary containing task execution results
        """
        self.result = result
        self.update_status("completed")

    def __repr__(self) -> str:
        return (
            f"TaskDefinition(task_id='{self.task_id}', "
            f"type={self.task_type.value}, "
            f"priority={self.priority}, "
            f"agent='{self.assigned_agent}', "
            f"status='{self.status}')"
        )
