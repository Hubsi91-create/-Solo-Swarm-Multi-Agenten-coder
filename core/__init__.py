"""
Solo-Swarm Multi-Agent System - Core Package
"""

from .tdf_schema import TaskDefinition, TaskType
from .agent_framework import BaseAgent
from .token_tracker import TokenTracker
from .context_manager import ContextManager
from .claude_md_manager import CLAUDEMDManager, AgentRole

__all__ = [
    "TaskDefinition",
    "TaskType",
    "BaseAgent",
    "TokenTracker",
    "ContextManager",
    "CLAUDEMDManager",
    "AgentRole",
]
