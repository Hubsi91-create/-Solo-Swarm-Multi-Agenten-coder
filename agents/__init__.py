"""
Solo-Swarm Multi-Agent System - Agents Package
"""

from .workers.coder_agent import CoderAgent
from .orchestrator.architect_agent import ArchitectAgent

__all__ = [
    "CoderAgent",
    "ArchitectAgent",
]
