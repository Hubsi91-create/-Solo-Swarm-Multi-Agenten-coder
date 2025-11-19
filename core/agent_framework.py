"""
Base Agent Framework - Abstract Agent Implementation
Defines the core lifecycle and interfaces for all agents in the Solo-Swarm system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from .tdf_schema import TaskDefinition


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentError:
    """Structured error information for agent operations"""

    error_type: str
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    recoverable: bool = False

    def __str__(self) -> str:
        return f"[{self.error_type}] {self.message} (recoverable={self.recoverable})"


@dataclass
class AgentResponse:
    """Standard response format for agent operations"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[AgentError] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelCallException(Exception):
    """Exception raised when LLM model calls fail"""

    def __init__(self, model: str, message: str, original_error: Optional[Exception] = None):
        self.model = model
        self.message = message
        self.original_error = original_error
        super().__init__(f"Model call failed for {model}: {message}")


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Solo-Swarm system.

    Implements the core agent lifecycle:
    1. gather_context: Collect necessary information
    2. take_action: Execute the task
    3. verify_work: Validate the results

    Subclasses must implement these lifecycle methods.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_name: Human-readable name for the agent
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config = config or {}
        self.execution_history: List[Dict[str, Any]] = []
        self.error_log: List[AgentError] = []

        logger.info(f"Initialized agent: {self.agent_name} (ID: {self.agent_id})")

    # ==================== Agent Lifecycle Methods ====================

    @abstractmethod
    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather all necessary context for task execution.

        This method should:
        - Analyze the task requirements
        - Collect relevant information from available sources
        - Prepare the execution environment
        - Validate prerequisites

        Args:
            task: The TaskDefinition to gather context for

        Returns:
            AgentResponse with gathered context data or error
        """
        pass

    @abstractmethod
    def take_action(self, task: TaskDefinition, context: Dict[str, Any]) -> AgentResponse:
        """
        Phase 2: Execute the main task action.

        This method should:
        - Use the gathered context to perform the task
        - Make necessary LLM calls (via call_sonnet or call_haiku)
        - Execute required operations
        - Handle errors gracefully

        Args:
            task: The TaskDefinition to execute
            context: Context data gathered in phase 1

        Returns:
            AgentResponse with execution results or error
        """
        pass

    @abstractmethod
    def verify_work(self, task: TaskDefinition, action_result: Dict[str, Any]) -> AgentResponse:
        """
        Phase 3: Verify that the work was completed correctly.

        This method should:
        - Validate the action results
        - Check for quality and correctness
        - Run any necessary tests
        - Determine if retry is needed

        Args:
            task: The original TaskDefinition
            action_result: Results from the take_action phase

        Returns:
            AgentResponse indicating verification success or failure
        """
        pass

    # ==================== Orchestration Method ====================

    def execute_task(self, task: TaskDefinition) -> AgentResponse:
        """
        Orchestrates the full task execution lifecycle.

        Executes: gather_context -> take_action -> verify_work

        Args:
            task: The TaskDefinition to execute

        Returns:
            Final AgentResponse after all phases
        """
        logger.info(f"Agent {self.agent_name} starting task execution: {task.task_id}")

        try:
            # Phase 1: Gather Context
            context_response = self.gather_context(task)
            if not context_response.success:
                return self._handle_lifecycle_error("gather_context", context_response.error, task)

            context_data = context_response.data or {}

            # Phase 2: Take Action
            action_response = self.take_action(task, context_data)
            if not action_response.success:
                return self._handle_lifecycle_error("take_action", action_response.error, task)

            action_data = action_response.data or {}

            # Phase 3: Verify Work
            verify_response = self.verify_work(task, action_data)
            if not verify_response.success:
                return self._handle_lifecycle_error("verify_work", verify_response.error, task)

            # Success - log and return
            self._log_execution(task, "completed", verify_response.data)
            logger.info(f"Agent {self.agent_name} completed task: {task.task_id}")

            return AgentResponse(
                success=True,
                data=verify_response.data,
                metadata={
                    "agent_id": self.agent_id,
                    "task_id": task.task_id,
                    "lifecycle_completed": True
                }
            )

        except Exception as e:
            error = AgentError(
                error_type="unexpected_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            self.error_log.append(error)
            logger.error(f"Unexpected error in task execution: {error}")

            return AgentResponse(success=False, error=error)

    # ==================== LLM Integration Stubs ====================

    def call_sonnet(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, int, int]:
        """
        Call Claude Sonnet 3.5 model (stub for Anthropic SDK integration).

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments for the API

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            ModelCallException: If the API call fails
        """
        # TODO: Integrate Anthropic SDK
        # This is a stub that will be implemented when SDK is integrated

        logger.warning("call_sonnet is a stub - Anthropic SDK not yet integrated")

        raise NotImplementedError(
            "Anthropic SDK integration pending. "
            "This method will call Claude Sonnet 3.5 when implemented."
        )

    def call_haiku(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> Tuple[str, int, int]:
        """
        Call Claude Haiku 3.5 model (stub for Anthropic SDK integration).

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments for the API

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            ModelCallException: If the API call fails
        """
        # TODO: Integrate Anthropic SDK
        # This is a stub that will be implemented when SDK is integrated

        logger.warning("call_haiku is a stub - Anthropic SDK not yet integrated")

        raise NotImplementedError(
            "Anthropic SDK integration pending. "
            "This method will call Claude Haiku 3.5 when implemented."
        )

    # ==================== Error Handling Methods ====================

    def _handle_lifecycle_error(
        self,
        phase: str,
        error: Optional[AgentError],
        task: TaskDefinition
    ) -> AgentResponse:
        """Handle errors during lifecycle execution"""

        if error is None:
            error = AgentError(
                error_type="unknown_error",
                message=f"Error in {phase} with no error details",
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id, "phase": phase},
                recoverable=False
            )

        self.error_log.append(error)
        self._log_execution(task, "failed", {"phase": phase, "error": str(error)})

        logger.error(f"Lifecycle error in {phase}: {error}")

        return AgentResponse(success=False, error=error)

    def _log_execution(
        self,
        task: TaskDefinition,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log task execution to history"""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": task.task_id,
            "status": status,
            "details": details or {}
        }
        self.execution_history.append(log_entry)

    # ==================== Utility Methods ====================

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Return the agent's execution history"""
        return self.execution_history.copy()

    def get_error_log(self) -> List[AgentError]:
        """Return the agent's error log"""
        return self.error_log.copy()

    def clear_history(self) -> None:
        """Clear execution history and error log"""
        self.execution_history.clear()
        self.error_log.clear()
        logger.info(f"Cleared history for agent {self.agent_name}")

    def __repr__(self) -> str:
        return (
            f"BaseAgent(id='{self.agent_id}', "
            f"name='{self.agent_name}', "
            f"executions={len(self.execution_history)})"
        )
