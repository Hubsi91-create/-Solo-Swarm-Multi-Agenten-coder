"""
Verifier Agent - Validates 3D assets using Blender
Uses Haiku 3.5 for fast decision making and validation logic
"""

from typing import Dict, Any, Optional
import json
import logging
import subprocess
import os
from datetime import datetime
from pathlib import Path

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType


logger = logging.getLogger(__name__)


class ValidationResult:
    """
    Represents the result of asset validation.
    """

    def __init__(
        self,
        success: bool,
        validation_passed: bool,
        metrics: Dict[str, Any],
        issues: list,
        error: Optional[str] = None
    ):
        self.success = success
        self.validation_passed = validation_passed
        self.metrics = metrics
        self.issues = issues
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "validation_passed": self.validation_passed,
            "metrics": self.metrics,
            "issues": self.issues,
            "error": self.error
        }

    def __repr__(self) -> str:
        status = "PASSED" if self.validation_passed else "FAILED"
        return f"ValidationResult({status}, issues={len(self.issues)})"


class VerifierAgent(BaseAgent):
    """
    Verifier Agent - Quality gatekeeper for 3D assets.

    Responsibilities:
    - Validate assets using Blender in headless mode
    - Check poly counts, UV maps, dimensions, materials
    - Enforce quality constraints
    - Use Haiku 3.5 for fast validation decisions
    - Generate detailed validation reports
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "VerifierAgent",
        config: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[TokenTracker] = None
    ):
        """
        Initialize the Verifier Agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_name: Human-readable name for the agent
            config: Optional configuration dictionary
            token_tracker: Optional TokenTracker instance
        """
        super().__init__(agent_id, agent_name, config)

        # Initialize token tracker
        self.token_tracker = token_tracker or TokenTracker()

        # Agent-specific configuration
        self.blender_path = self.config.get("blender_path", "blender")
        self.validator_script = self.config.get(
            "validator_script",
            os.path.join(os.path.dirname(__file__), "../../scripts/blender_validator.py")
        )
        self.timeout = self.config.get("timeout", 60)  # 60 seconds timeout

        # Default constraints
        self.default_constraints = {
            "max_triangles": 50000,
            "min_dimensions": 0.1,
            "max_dimensions": 100.0,
            "require_uv_map": True,
            "max_materials": 10
        }

        logger.info(
            f"VerifierAgent initialized: {agent_name} (ID: {agent_id}), "
            f"blender={self.blender_path}"
        )

    def verify_asset(
        self,
        file_path: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Verify a 3D asset using Blender.

        Args:
            file_path: Path to the asset file
            constraints: Optional validation constraints

        Returns:
            ValidationResult object with metrics and pass/fail status
        """
        logger.info(f"Verifying asset: {file_path}")

        # Check file exists
        if not os.path.exists(file_path):
            logger.error(f"Asset file not found: {file_path}")
            return ValidationResult(
                success=False,
                validation_passed=False,
                metrics={},
                issues=[],
                error=f"File not found: {file_path}"
            )

        # Merge constraints with defaults
        final_constraints = {**self.default_constraints}
        if constraints:
            final_constraints.update(constraints)

        try:
            # Build Blender command
            cmd = [
                self.blender_path,
                "--background",
                "--python", self.validator_script,
                "--",
                file_path,
                json.dumps(final_constraints)
            ]

            logger.debug(f"Running Blender validation: {' '.join(cmd)}")

            # Execute Blender validation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Parse JSON output
            try:
                output = result.stdout.strip()
                validation_data = json.loads(output)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Blender output: {e}")
                logger.debug(f"STDOUT: {result.stdout}")
                logger.debug(f"STDERR: {result.stderr}")
                return ValidationResult(
                    success=False,
                    validation_passed=False,
                    metrics={},
                    issues=[],
                    error=f"Invalid JSON output from Blender: {str(e)}"
                )

            # Convert to ValidationResult
            return ValidationResult(
                success=validation_data.get("success", False),
                validation_passed=validation_data.get("validation_passed", False),
                metrics=validation_data.get("metrics", {}),
                issues=validation_data.get("issues", []),
                error=validation_data.get("error")
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Blender validation timed out after {self.timeout}s")
            return ValidationResult(
                success=False,
                validation_passed=False,
                metrics={},
                issues=[],
                error=f"Validation timed out after {self.timeout} seconds"
            )
        except FileNotFoundError:
            logger.error(f"Blender executable not found: {self.blender_path}")
            return ValidationResult(
                success=False,
                validation_passed=False,
                metrics={},
                issues=[],
                error=f"Blender not found at: {self.blender_path}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return ValidationResult(
                success=False,
                validation_passed=False,
                metrics={},
                issues=[],
                error=f"Validation error: {str(e)}"
            )

    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather context for asset verification.

        Extracts:
        - Asset file path
        - Validation constraints
        - Quality requirements

        Args:
            task: The TaskDefinition for verification

        Returns:
            AgentResponse with gathered context or error
        """
        logger.info(f"VerifierAgent gathering context for: {task.task_id}")

        try:
            # Extract file path
            file_path = task.context.get("file_path", "")
            if not file_path:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="missing_file_path",
                        message="No file_path found in task context",
                        timestamp=datetime.utcnow(),
                        context={"task_id": task.task_id},
                        recoverable=False
                    )
                )

            # Extract constraints
            constraints = task.context.get("constraints", {})

            # Merge with requirements
            if task.requirements:
                for key in ["max_triangles", "require_uv_map", "max_dimensions"]:
                    if key in task.requirements:
                        constraints[key] = task.requirements[key]

            context_data = {
                "file_path": file_path,
                "constraints": constraints,
                "quality": task.context.get("quality", "medium")
            }

            logger.info(f"Context gathered: file={file_path}")

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
        Phase 2: Validate the asset.

        Uses Haiku 3.5 for decision making about validation rules.

        Args:
            task: The verification TaskDefinition
            context: Context data from gather_context

        Returns:
            AgentResponse with validation results or error
        """
        logger.info(f"VerifierAgent validating asset for: {task.task_id}")

        try:
            file_path = context["file_path"]
            constraints = context["constraints"]

            # Simulate Haiku usage for constraint adjustment
            # In production: Haiku might adjust constraints based on asset type
            haiku_input_tokens = 100
            haiku_output_tokens = 20

            cost = self.token_tracker.track_usage(
                model=ModelType.HAIKU_3_5,
                input_tokens=haiku_input_tokens,
                output_tokens=haiku_output_tokens,
                metadata={
                    "task_id": task.task_id,
                    "agent_id": self.agent_id,
                    "operation": "validation_planning"
                }
            )

            # Verify asset
            validation_result = self.verify_asset(file_path, constraints)

            # Prepare result
            result = {
                "validation_result": validation_result.to_dict(),
                "file_path": file_path,
                "constraints": constraints,
                "token_usage": {
                    "input_tokens": haiku_input_tokens,
                    "output_tokens": haiku_output_tokens,
                    "cost_usd": cost
                },
                "metadata": {
                    "model": ModelType.HAIKU_3_5,
                    "agent_id": self.agent_id
                }
            }

            if validation_result.success:
                logger.info(
                    f"Validation completed: {validation_result.validation_passed}, "
                    f"issues={len(validation_result.issues)}"
                )
                return AgentResponse(
                    success=True,
                    data=result,
                    metadata={"agent_id": self.agent_id, "phase": "take_action"}
                )
            else:
                error = AgentError(
                    error_type="validation_error",
                    message=validation_result.error or "Validation failed",
                    timestamp=datetime.utcnow(),
                    context={
                        "task_id": task.task_id,
                        "file_path": file_path
                    },
                    recoverable=True
                )
                logger.error(f"Validation error: {error}")
                return AgentResponse(success=False, error=error, data=result)

        except Exception as e:
            error = AgentError(
                error_type="action_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            logger.error(f"Error during validation: {error}")
            return AgentResponse(success=False, error=error)

    def verify_work(self, task: TaskDefinition, action_result: Dict[str, Any]) -> AgentResponse:
        """
        Phase 3: Verify validation results.

        Checks that validation was performed correctly.

        Args:
            task: The original TaskDefinition
            action_result: Results from take_action phase

        Returns:
            AgentResponse indicating verification success or failure
        """
        logger.info(f"VerifierAgent verifying validation for: {task.task_id}")

        try:
            validation_result = action_result["validation_result"]

            verification_results = {
                "validation_completed": validation_result.get("success", False),
                "has_metrics": len(validation_result.get("metrics", {})) > 0,
                "validation_passed": validation_result.get("validation_passed", False)
            }

            verification_passed = verification_results["validation_completed"]

            verification_data = {
                "verification_passed": verification_passed,
                "verification_results": verification_results,
                "validation_result": validation_result,
                "token_usage": action_result.get("token_usage")
            }

            if verification_passed:
                logger.info(f"Validation verification passed for {task.task_id}")
                return AgentResponse(
                    success=True,
                    data=verification_data,
                    metadata={"agent_id": self.agent_id, "phase": "verify_work"}
                )
            else:
                error = AgentError(
                    error_type="verification_failed",
                    message="Validation verification failed",
                    timestamp=datetime.utcnow(),
                    context={
                        "task_id": task.task_id,
                        "verification_results": verification_results
                    },
                    recoverable=True
                )
                logger.warning(f"Verification failed for {task.task_id}: {error}")
                return AgentResponse(success=False, error=error, data=verification_data)

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

    def __repr__(self) -> str:
        return (
            f"VerifierAgent(id='{self.agent_id}', "
            f"name='{self.agent_name}', "
            f"blender='{self.blender_path}')"
        )
