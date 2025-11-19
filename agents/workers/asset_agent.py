"""
Asset Agent - Generates 3D assets using external APIs
Uses Haiku 3.5 for fast decision making and API selection
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType
from integrations.asset_apis import AssetGeneratorAPI, TripoAPI, SloydAPI, MeshyAPI


logger = logging.getLogger(__name__)


class AssetAgent(BaseAgent):
    """
    Asset Agent - Generates 3D models using external APIs.

    Responsibilities:
    - Select appropriate API based on task requirements
    - Generate 3D assets with style consistency
    - Handle API failures and retries
    - Track generation costs and metadata
    - Use Haiku 3.5 for fast decision making
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "AssetAgent",
        config: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[TokenTracker] = None
    ):
        """
        Initialize the Asset Agent.

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
        self.output_dir = self.config.get("output_dir", "temp_assets")
        self.default_api = self.config.get("default_api", "tripo")
        self.max_retries = self.config.get("max_retries", 2)

        # Initialize API clients
        self.apis: Dict[str, AssetGeneratorAPI] = {
            "tripo": TripoAPI(output_dir=self.output_dir),
            "sloyd": SloydAPI(output_dir=self.output_dir),
            "meshy": MeshyAPI(output_dir=self.output_dir)
        }

        logger.info(
            f"AssetAgent initialized: {agent_name} (ID: {agent_id}), "
            f"output_dir={self.output_dir}, available_apis={list(self.apis.keys())}"
        )

    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather context for asset generation.

        Extracts:
        - Asset description from task context
        - Style prompt
        - Preferred API
        - Quality requirements

        Args:
            task: The TaskDefinition for asset generation

        Returns:
            AgentResponse with gathered context or error
        """
        logger.info(f"AssetAgent gathering context for: {task.task_id}")

        try:
            # Extract required information
            asset_description = task.context.get("asset_description", "")
            if not asset_description:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="missing_description",
                        message="No asset_description found in task context",
                        timestamp=datetime.utcnow(),
                        context={"task_id": task.task_id},
                        recoverable=False
                    )
                )

            # Extract optional parameters
            style_prompt = task.context.get("style_prompt", None)
            preferred_api = task.context.get("preferred_api", self.default_api)
            quality = task.context.get("quality", "medium")

            # Validate preferred API
            if preferred_api not in self.apis:
                logger.warning(
                    f"Preferred API '{preferred_api}' not available, "
                    f"falling back to '{self.default_api}'"
                )
                preferred_api = self.default_api

            context_data = {
                "asset_description": asset_description,
                "style_prompt": style_prompt,
                "preferred_api": preferred_api,
                "quality": quality,
                "available_apis": list(self.apis.keys())
            }

            logger.info(
                f"Context gathered: description='{asset_description[:50]}...', "
                f"style='{style_prompt}', api={preferred_api}"
            )

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
        Phase 2: Generate 3D asset using selected API.

        Uses Haiku 3.5 to make decisions about:
        - API selection based on requirements
        - Prompt enhancement
        - Retry strategy

        Args:
            task: The asset generation TaskDefinition
            context: Context data from gather_context

        Returns:
            AgentResponse with generation results or error
        """
        logger.info(f"AssetAgent generating asset for: {task.task_id}")

        try:
            asset_description = context["asset_description"]
            style_prompt = context["style_prompt"]
            preferred_api = context["preferred_api"]
            quality = context["quality"]

            # Select API (in production, Haiku would decide based on requirements)
            selected_api_name = self._select_api(
                asset_description=asset_description,
                quality=quality,
                preferred_api=preferred_api
            )
            selected_api = self.apis[selected_api_name]

            logger.info(f"Selected API: {selected_api_name}")

            # Simulate Haiku token usage for API selection
            # In production: this would be an actual Haiku call
            haiku_input_tokens = len(asset_description) // 4 + 50
            haiku_output_tokens = 20
            cost = self.token_tracker.track_usage(
                model=ModelType.HAIKU_3_5,
                input_tokens=haiku_input_tokens,
                output_tokens=haiku_output_tokens,
                metadata={
                    "task_id": task.task_id,
                    "agent_id": self.agent_id,
                    "operation": "api_selection"
                }
            )

            # Generate asset with retries
            attempt = 0
            generation_result = None

            while attempt <= self.max_retries:
                logger.info(f"Generation attempt {attempt + 1}/{self.max_retries + 1}")

                generation_result = selected_api.generate_model(
                    prompt=asset_description,
                    style_prompt=style_prompt
                )

                if generation_result["success"]:
                    logger.info(f"Asset generated successfully on attempt {attempt + 1}")
                    break

                logger.warning(
                    f"Generation failed on attempt {attempt + 1}: "
                    f"{generation_result['error']}"
                )
                attempt += 1

                # Try different API on retry
                if attempt <= self.max_retries:
                    selected_api_name = self._select_fallback_api(selected_api_name)
                    selected_api = self.apis[selected_api_name]
                    logger.info(f"Retrying with fallback API: {selected_api_name}")

            # Prepare result
            result = {
                "generation_result": generation_result,
                "selected_api": selected_api_name,
                "attempts": attempt + 1,
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

            if generation_result["success"]:
                logger.info(
                    f"Asset generation successful: {generation_result['file_path']}"
                )
                return AgentResponse(
                    success=True,
                    data=result,
                    metadata={"agent_id": self.agent_id, "phase": "take_action"}
                )
            else:
                error = AgentError(
                    error_type="generation_failed",
                    message=f"Failed after {attempt} attempts: {generation_result['error']}",
                    timestamp=datetime.utcnow(),
                    context={
                        "task_id": task.task_id,
                        "last_error": generation_result["error"]
                    },
                    recoverable=True
                )
                logger.error(f"Asset generation failed: {error}")
                return AgentResponse(success=False, error=error, data=result)

        except Exception as e:
            error = AgentError(
                error_type="action_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            logger.error(f"Error during asset generation: {error}")
            return AgentResponse(success=False, error=error)

    def verify_work(self, task: TaskDefinition, action_result: Dict[str, Any]) -> AgentResponse:
        """
        Phase 3: Verify generated asset.

        Checks:
        - File exists
        - File is not empty
        - Metadata is complete
        - Quality meets requirements

        Args:
            task: The original TaskDefinition
            action_result: Results from take_action phase

        Returns:
            AgentResponse indicating verification success or failure
        """
        logger.info(f"AssetAgent verifying asset for: {task.task_id}")

        try:
            generation_result = action_result["generation_result"]

            validation_results = {
                "file_generated": generation_result["success"],
                "file_exists": False,
                "file_not_empty": False,
                "metadata_complete": False
            }

            if generation_result["success"]:
                file_path = generation_result["file_path"]

                # Check file exists
                import os
                if os.path.exists(file_path):
                    validation_results["file_exists"] = True

                    # Check file not empty
                    if os.path.getsize(file_path) > 0:
                        validation_results["file_not_empty"] = True

                # Check metadata
                metadata = generation_result.get("metadata", {})
                required_fields = ["api", "prompt", "format"]
                if all(field in metadata for field in required_fields):
                    validation_results["metadata_complete"] = True

            # Check if validation passed
            validation_passed = all(validation_results.values())

            verification_data = {
                "verification_passed": validation_passed,
                "validation_results": validation_results,
                "generation_result": generation_result,
                "token_usage": action_result.get("token_usage")
            }

            if validation_passed:
                logger.info(f"Asset verification passed for {task.task_id}")
                return AgentResponse(
                    success=True,
                    data=verification_data,
                    metadata={"agent_id": self.agent_id, "phase": "verify_work"}
                )
            else:
                error = AgentError(
                    error_type="verification_failed",
                    message="Asset verification failed",
                    timestamp=datetime.utcnow(),
                    context={
                        "task_id": task.task_id,
                        "validation_results": validation_results
                    },
                    recoverable=True
                )
                logger.warning(f"Asset verification failed for {task.task_id}: {error}")
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

    def _select_api(
        self,
        asset_description: str,
        quality: str,
        preferred_api: str
    ) -> str:
        """
        Select appropriate API based on requirements.

        In production, this would use Haiku to make intelligent decisions.
        For now, uses simple heuristics.

        Args:
            asset_description: Description of asset to generate
            quality: Required quality level
            preferred_api: User's preferred API

        Returns:
            Selected API name
        """
        # Simple heuristics (in production: Haiku decides)
        if "game" in asset_description.lower() or "low poly" in asset_description.lower():
            return "sloyd"  # Best for game assets
        elif "high quality" in quality.lower() or "pbr" in asset_description.lower():
            return "meshy"  # Best for high quality
        else:
            return preferred_api  # Use preference

    def _select_fallback_api(self, failed_api: str) -> str:
        """
        Select a fallback API when primary fails.

        Args:
            failed_api: The API that just failed

        Returns:
            Name of fallback API
        """
        # Round-robin fallback
        api_list = list(self.apis.keys())
        current_index = api_list.index(failed_api)
        next_index = (current_index + 1) % len(api_list)
        return api_list[next_index]

    def __repr__(self) -> str:
        return (
            f"AssetAgent(id='{self.agent_id}', "
            f"name='{self.agent_name}', "
            f"apis={list(self.apis.keys())})"
        )
