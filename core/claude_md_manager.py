"""
CLAUDE.md Manager - Rules and Guidelines Parser
Parses CLAUDE.md files and injects context-specific rules into prompts
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent role types for rule filtering"""
    WORKER = "worker"
    ORCHESTRATOR = "orchestrator"
    GLOBAL = "global"


@dataclass
class RuleSection:
    """Represents a section of rules from CLAUDE.md"""
    title: str
    content: str
    role: AgentRole


class CLAUDEMDManager:
    """
    Manages CLAUDE.md rules and injects them into agent prompts.

    CLAUDE.md structure:
    # Global Rules
    (applies to all agents)

    # Worker Rules
    (applies only to worker agents: CoderAgent, ReviewerAgent, etc.)

    # Orchestrator Rules
    (applies only to orchestrator agents: ArchitectAgent)
    """

    # Section markers
    GLOBAL_MARKER = "# Global Rules"
    WORKER_MARKER = "# Worker Rules"
    ORCHESTRATOR_MARKER = "# Orchestrator Rules"

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the CLAUDE.md Manager.

        Args:
            config_dir: Directory containing CLAUDE.md file
        """
        self.config_dir = Path(config_dir)
        self.claude_md_path = self.config_dir / "CLAUDE.md"
        self.sections: Dict[AgentRole, List[RuleSection]] = {
            AgentRole.GLOBAL: [],
            AgentRole.WORKER: [],
            AgentRole.ORCHESTRATOR: []
        }

        # Parse the file if it exists
        if self.claude_md_path.exists():
            self.parse(str(self.claude_md_path))
        else:
            logger.warning(f"CLAUDE.md not found at {self.claude_md_path}")

    def parse(self, file_path: Optional[str] = None) -> None:
        """
        Parse CLAUDE.md file and extract rule sections.

        Args:
            file_path: Path to CLAUDE.md file (uses default if None)
        """
        if file_path:
            path = Path(file_path)
        else:
            path = self.claude_md_path

        if not path.exists():
            logger.error(f"CLAUDE.md not found at {path}")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Clear existing sections
            self.sections = {
                AgentRole.GLOBAL: [],
                AgentRole.WORKER: [],
                AgentRole.ORCHESTRATOR: []
            }

            # Split into sections
            current_role = AgentRole.GLOBAL
            current_section = []
            current_title = "Global Rules"

            for line in content.split('\n'):
                # Check for section markers
                if line.strip() == self.GLOBAL_MARKER:
                    # Save previous section
                    if current_section:
                        self._save_section(current_role, current_title, current_section)
                    current_role = AgentRole.GLOBAL
                    current_title = "Global Rules"
                    current_section = []

                elif line.strip() == self.WORKER_MARKER:
                    # Save previous section
                    if current_section:
                        self._save_section(current_role, current_title, current_section)
                    current_role = AgentRole.WORKER
                    current_title = "Worker Rules"
                    current_section = []

                elif line.strip() == self.ORCHESTRATOR_MARKER:
                    # Save previous section
                    if current_section:
                        self._save_section(current_role, current_title, current_section)
                    current_role = AgentRole.ORCHESTRATOR
                    current_title = "Orchestrator Rules"
                    current_section = []

                else:
                    current_section.append(line)

            # Save last section
            if current_section:
                self._save_section(current_role, current_title, current_section)

            logger.info(
                f"Parsed CLAUDE.md: "
                f"{len(self.sections[AgentRole.GLOBAL])} global, "
                f"{len(self.sections[AgentRole.WORKER])} worker, "
                f"{len(self.sections[AgentRole.ORCHESTRATOR])} orchestrator sections"
            )

        except Exception as e:
            logger.error(f"Error parsing CLAUDE.md: {e}")

    def _save_section(
        self,
        role: AgentRole,
        title: str,
        lines: List[str]
    ) -> None:
        """Save a parsed section"""
        content = '\n'.join(lines).strip()
        if content:  # Only save non-empty sections
            section = RuleSection(
                title=title,
                content=content,
                role=role
            )
            self.sections[role].append(section)

    def inject_into_prompt(
        self,
        system_prompt: str,
        agent_role: AgentRole
    ) -> str:
        """
        Inject relevant CLAUDE.md rules into a system prompt.

        Args:
            system_prompt: Base system prompt
            agent_role: Role of the agent (WORKER or ORCHESTRATOR)

        Returns:
            Enhanced system prompt with injected rules
        """
        # Collect applicable sections
        applicable_sections = []

        # Always include global rules
        applicable_sections.extend(self.sections[AgentRole.GLOBAL])

        # Include role-specific rules
        applicable_sections.extend(self.sections[agent_role])

        if not applicable_sections:
            logger.warning(f"No rules found for role {agent_role}")
            return system_prompt

        # Build injected prompt
        parts = [system_prompt, "\n"]

        # Add separator
        parts.append("=" * 80)
        parts.append("\nCLAUDE.md RULES AND GUIDELINES\n")
        parts.append("=" * 80)
        parts.append("\n")

        # Add each section
        for section in applicable_sections:
            parts.append(f"\n## {section.title}\n")
            parts.append(section.content)
            parts.append("\n")

        # Add closing separator
        parts.append("=" * 80)
        parts.append("\n")

        injected_prompt = "".join(parts)

        logger.info(
            f"Injected {len(applicable_sections)} rule sections for {agent_role} "
            f"({len(injected_prompt) - len(system_prompt)} chars added)"
        )

        return injected_prompt

    def get_rules_for_role(self, role: AgentRole) -> List[RuleSection]:
        """
        Get all applicable rules for a specific role.

        Args:
            role: Agent role

        Returns:
            List of applicable rule sections
        """
        rules = []
        rules.extend(self.sections[AgentRole.GLOBAL])
        rules.extend(self.sections[role])
        return rules

    def get_section_count(self) -> Dict[str, int]:
        """Get count of sections per role"""
        return {
            "global": len(self.sections[AgentRole.GLOBAL]),
            "worker": len(self.sections[AgentRole.WORKER]),
            "orchestrator": len(self.sections[AgentRole.ORCHESTRATOR])
        }

    def reload(self) -> None:
        """Reload CLAUDE.md from disk"""
        self.parse()
        logger.info("CLAUDE.md reloaded")

    def __repr__(self) -> str:
        counts = self.get_section_count()
        return (
            f"CLAUDEMDManager("
            f"global={counts['global']}, "
            f"worker={counts['worker']}, "
            f"orchestrator={counts['orchestrator']})"
        )
