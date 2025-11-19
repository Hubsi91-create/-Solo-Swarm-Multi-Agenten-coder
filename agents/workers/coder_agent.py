"""
Coder Agent - Specialized agent for code generation tasks
Inherits from BaseAgent and implements code generation workflow
"""

from typing import Dict, Any, Optional, List
import logging

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType
from datetime import datetime


logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """
    Specialized agent for code generation and implementation tasks.

    This agent focuses on:
    - Analyzing code requirements
    - Generating code implementations
    - Verifying code quality
    - Using Haiku model for cost-effective code generation
    """

    SUPPORTED_LANGUAGES = {
        "python", "javascript", "typescript", "java", "go", "rust",
        "c++", "c#", "ruby", "php", "swift", "kotlin"
    }

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "CoderAgent",
        config: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[TokenTracker] = None
    ):
        """
        Initialize the Coder Agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_name: Human-readable name for the agent
            config: Optional configuration dictionary
            token_tracker: Optional TokenTracker instance for cost tracking
        """
        super().__init__(agent_id, agent_name, config)

        # Initialize token tracker
        self.token_tracker = token_tracker or TokenTracker()

        # Agent-specific configuration
        self.default_language = self.config.get("default_language", "python")
        self.max_code_lines = self.config.get("max_code_lines", 500)
        self.temperature = self.config.get("temperature", 0.3)  # Lower for code generation

        logger.info(
            f"CoderAgent initialized: {agent_name} (ID: {agent_id}), "
            f"default language: {self.default_language}"
        )

    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather context for code generation.

        Analyzes the task to extract:
        - Programming language
        - Required libraries/frameworks
        - Code specifications
        - Constraints

        Args:
            task: The TaskDefinition to gather context for

        Returns:
            AgentResponse with gathered context data or error
        """
        logger.info(f"CoderAgent gathering context for task: {task.task_id}")

        try:
            # Validate task type
            if task.task_type not in [TaskType.IMPLEMENTATION, TaskType.REFACTORING]:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="invalid_task_type",
                        message=f"CoderAgent only handles IMPLEMENTATION and REFACTORING tasks, got {task.task_type}",
                        timestamp=datetime.utcnow(),
                        context={"task_id": task.task_id},
                        recoverable=False
                    )
                )

            # Extract language from context
            language = task.context.get("language", self.default_language)
            if language.lower() not in self.SUPPORTED_LANGUAGES:
                logger.warning(
                    f"Unsupported language '{language}', defaulting to {self.default_language}"
                )
                language = self.default_language

            # Extract requirements
            requirements = task.requirements or {}
            framework = task.context.get("framework", None)
            specifications = task.context.get("specifications", "")

            # Prepare context data
            context_data = {
                "language": language,
                "framework": framework,
                "specifications": specifications,
                "requirements": requirements,
                "max_lines": requirements.get("max_lines", self.max_code_lines),
                "style_guide": requirements.get("style_guide", "PEP8" if language == "python" else "standard"),
                "include_tests": requirements.get("include_tests", False),
                "include_docs": requirements.get("include_docs", True)
            }

            logger.info(f"Context gathered for {task.task_id}: language={language}")

            return AgentResponse(
                success=True,
                data=context_data,
                metadata={"agent_id": self.agent_id, "phase": "gather_context"}
            )

        except Exception as e:
            error = AgentError(
                error_type="context_gathering_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            logger.error(f"Error gathering context: {error}")
            return AgentResponse(success=False, error=error)

    def take_action(self, task: TaskDefinition, context: Dict[str, Any]) -> AgentResponse:
        """
        Phase 2: Generate code based on gathered context.

        This method will eventually call the Haiku model via Anthropic SDK.
        For now, it simulates code generation with structured output.

        Args:
            task: The TaskDefinition to execute
            context: Context data gathered in phase 1

        Returns:
            AgentResponse with generated code or error
        """
        logger.info(f"CoderAgent taking action for task: {task.task_id}")

        try:
            language = context.get("language", "python")
            framework = context.get("framework")
            specifications = context.get("specifications", "")

            # Simulate code generation
            # In production, this would call self.call_haiku() with the prompt
            generated_code = self._simulate_code_generation(
                language=language,
                framework=framework,
                specifications=specifications,
                task=task
            )

            # Simulate token usage (Haiku model)
            # In production, these would come from the actual API response
            simulated_input_tokens = 500  # Prompt tokens
            simulated_output_tokens = 1000  # Generated code tokens

            # Track usage (simulated for now)
            cost = self.token_tracker.track_usage(
                model=ModelType.HAIKU_3_5,
                input_tokens=simulated_input_tokens,
                output_tokens=simulated_output_tokens,
                metadata={
                    "task_id": task.task_id,
                    "agent_id": self.agent_id,
                    "language": language
                }
            )

            # Prepare action result
            action_result = {
                "generated_code": generated_code,
                "language": language,
                "framework": framework,
                "token_usage": {
                    "input_tokens": simulated_input_tokens,
                    "output_tokens": simulated_output_tokens,
                    "cost_usd": cost
                },
                "metadata": {
                    "model": ModelType.HAIKU_3_5,
                    "temperature": self.temperature,
                    "agent_id": self.agent_id
                }
            }

            logger.info(
                f"Code generation completed for {task.task_id}: "
                f"{simulated_output_tokens} tokens, ${cost:.6f}"
            )

            return AgentResponse(
                success=True,
                data=action_result,
                metadata={"agent_id": self.agent_id, "phase": "take_action"}
            )

        except Exception as e:
            error = AgentError(
                error_type="code_generation_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=True
            )
            logger.error(f"Error generating code: {error}")
            return AgentResponse(success=False, error=error)

    def verify_work(self, task: TaskDefinition, action_result: Dict[str, Any]) -> AgentResponse:
        """
        Phase 3: Verify generated code quality.

        Performs basic validation checks on the generated code:
        - Syntax validation (simulated)
        - Line count check
        - Basic quality metrics

        Args:
            task: The original TaskDefinition
            action_result: Results from the take_action phase

        Returns:
            AgentResponse indicating verification success or failure
        """
        logger.info(f"CoderAgent verifying work for task: {task.task_id}")

        try:
            generated_code = action_result.get("generated_code", "")
            language = action_result.get("language", "python")

            # Basic validation checks
            validation_results = {
                "syntax_valid": self._validate_syntax(generated_code, language),
                "line_count": len(generated_code.split("\n")),
                "meets_requirements": True,  # Simulated
                "quality_score": self._calculate_quality_score(generated_code)
            }

            # Check if code meets line limit
            max_lines = task.requirements.get("max_lines", self.max_code_lines)
            if validation_results["line_count"] > max_lines:
                logger.warning(
                    f"Generated code exceeds max lines: "
                    f"{validation_results['line_count']} > {max_lines}"
                )
                validation_results["meets_requirements"] = False

            # Determine if verification passed
            verification_passed = (
                validation_results["syntax_valid"] and
                validation_results["meets_requirements"] and
                validation_results["quality_score"] >= 70
            )

            verification_data = {
                "verification_passed": verification_passed,
                "validation_results": validation_results,
                "generated_code": generated_code,
                "language": language,
                "token_usage": action_result.get("token_usage")
            }

            if not verification_passed:
                error = AgentError(
                    error_type="verification_failed",
                    message="Generated code did not pass quality checks",
                    timestamp=datetime.utcnow(),
                    context={
                        "task_id": task.task_id,
                        "validation_results": validation_results
                    },
                    recoverable=True
                )
                logger.warning(f"Verification failed for {task.task_id}: {error}")
                return AgentResponse(success=False, error=error, data=verification_data)

            logger.info(
                f"Verification passed for {task.task_id}: "
                f"quality_score={validation_results['quality_score']}"
            )

            return AgentResponse(
                success=True,
                data=verification_data,
                metadata={"agent_id": self.agent_id, "phase": "verify_work"}
            )

        except Exception as e:
            error = AgentError(
                error_type="verification_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            logger.error(f"Error during verification: {error}")
            return AgentResponse(success=False, error=error)

    def _simulate_code_generation(
        self,
        language: str,
        framework: Optional[str],
        specifications: str,
        task: TaskDefinition
    ) -> str:
        """
        Simulate code generation (placeholder for real Anthropic API call).

        This will be replaced with actual call_haiku() when SDK is integrated.

        Args:
            language: Programming language
            framework: Optional framework
            specifications: Code specifications
            task: Task definition

        Returns:
            Generated code as string
        """
        # This is a simulation - in production, this would call self.call_haiku()
        framework_line = f"\n# Framework: {framework}" if framework else ""
        specs_line = f"\n# Specifications: {specifications}" if specifications else ""

        template = f'''"""
Generated code for task: {task.task_id}
Language: {language}{framework_line}{specs_line}
"""

def main():
    """
    Main function for {task.task_id}

    This is simulated code generation.
    In production, this would be generated by Claude Haiku via Anthropic SDK.
    """
    print("Hello from {language}!")

    # TODO: Implement actual functionality based on specifications
    result = process_task()

    return result


def process_task():
    """Process the task according to specifications"""
    # Simulated implementation
    return {{"status": "success", "message": "Task completed"}}


if __name__ == "__main__":
    main()
'''
        return template

    def _validate_syntax(self, code: str, language: str) -> bool:
        """
        Validate code syntax (simulated).

        In production, this could use language-specific parsers.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            True if syntax is valid, False otherwise
        """
        # Simulated validation - always returns True for now
        # In production, use ast.parse() for Python, etc.
        return len(code.strip()) > 0

    def _calculate_quality_score(self, code: str) -> int:
        """
        Calculate a simple quality score for generated code.

        Args:
            code: Code to analyze

        Returns:
            Quality score (0-100)
        """
        score = 80  # Base score

        # Check for documentation
        if '"""' in code or "'''" in code or "/**" in code:
            score += 10

        # Check for proper function definitions
        if "def " in code or "function " in code:
            score += 5

        # Check for comments
        if "#" in code or "//" in code:
            score += 5

        return min(score, 100)

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get token usage summary for this agent.

        Returns:
            Dictionary with usage statistics
        """
        summary = self.token_tracker.get_summary()
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "usage_summary": summary.to_dict(),
            "executions": len(self.execution_history)
        }

    def __repr__(self) -> str:
        return (
            f"CoderAgent(id='{self.agent_id}', "
            f"name='{self.agent_name}', "
            f"language='{self.default_language}', "
            f"executions={len(self.execution_history)})"
        )
