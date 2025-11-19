"""
Art Director - Manages style consistency and asset decomposition
Uses Sonnet 3.5 for creative direction and high-level planning
"""

from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime

from core.agent_framework import BaseAgent, AgentResponse, AgentError
from core.tdf_schema import TaskDefinition, TaskType
from core.token_tracker import TokenTracker, ModelType


logger = logging.getLogger(__name__)


class ArtDirector(BaseAgent):
    """
    Art Director - Master of style consistency and asset planning.

    Responsibilities:
    - Create master style prompts for consistent visual themes
    - Decompose high-level asset requests into atomic tasks
    - Ensure visual coherence across all generated assets
    - Use Sonnet 3.5 for creative reasoning
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "ArtDirector",
        config: Optional[Dict[str, Any]] = None,
        token_tracker: Optional[TokenTracker] = None
    ):
        """
        Initialize the Art Director.

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
        self.temperature = self.config.get("temperature", 0.8)  # Higher for creativity
        self.style_templates = self._load_style_templates()

        logger.info(
            f"ArtDirector initialized: {agent_name} (ID: {agent_id}), "
            f"temperature={self.temperature}"
        )

    def create_master_style_prompt(self, theme: str) -> str:
        """
        Create a master style prompt for consistent asset generation.

        Uses Sonnet 3.5 to generate a comprehensive style guide that ensures
        all assets in a set maintain visual consistency.

        Args:
            theme: High-level theme (e.g., "Medieval Dungeon", "Sci-Fi Space Station")

        Returns:
            Comprehensive style prompt string
        """
        logger.info(f"ArtDirector creating master style prompt for theme: '{theme}'")

        # Simulate Sonnet call for style prompt creation
        # In production: would call actual Sonnet API
        style_prompt = self._generate_style_prompt(theme)

        # Track token usage (simulated)
        input_tokens = len(theme) // 4 + 100  # Base prompt + theme
        output_tokens = len(style_prompt) // 4

        cost = self.token_tracker.track_usage(
            model=ModelType.SONNET_3_5,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "agent_id": self.agent_id,
                "operation": "create_style_prompt",
                "theme": theme
            }
        )

        logger.info(
            f"Master style prompt created: {len(style_prompt)} chars, "
            f"cost: ${cost:.4f}"
        )

        return style_prompt

    def decompose_asset_request(
        self,
        request: str,
        theme: Optional[str] = None,
        style_prompt: Optional[str] = None
    ) -> List[TaskDefinition]:
        """
        Decompose a high-level asset request into atomic tasks.

        Uses Sonnet 3.5 to intelligently break down complex requests like
        "Create a dungeon set" into individual asset tasks.

        Args:
            request: High-level asset request
            theme: Optional theme context
            style_prompt: Optional master style prompt for consistency

        Returns:
            List of TaskDefinition objects for individual assets
        """
        logger.info(f"ArtDirector decomposing request: '{request}'")

        # Create master style prompt if not provided
        if not style_prompt and theme:
            style_prompt = self.create_master_style_prompt(theme)
        elif not style_prompt:
            # Generate default style prompt
            style_prompt = self.create_master_style_prompt("Generic Game Assets")

        # Simulate Sonnet call for decomposition
        # In production: would call actual Sonnet API
        asset_list = self._decompose_request(request, theme)

        # Track token usage
        input_tokens = len(request) // 4 + len(style_prompt or "") // 4 + 200
        output_tokens = len(asset_list) * 50  # Estimate

        cost = self.token_tracker.track_usage(
            model=ModelType.SONNET_3_5,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "agent_id": self.agent_id,
                "operation": "decompose_request",
                "request": request
            }
        )

        # Create TaskDefinition objects
        tasks = []
        for idx, asset_info in enumerate(asset_list):
            task = self._create_asset_task(
                asset_info=asset_info,
                style_prompt=style_prompt,
                index=idx,
                request_id=f"asset_req_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            )
            tasks.append(task)

        logger.info(
            f"Request decomposed into {len(tasks)} tasks, "
            f"cost: ${cost:.4f}"
        )

        return tasks

    def gather_context(self, task: TaskDefinition) -> AgentResponse:
        """
        Phase 1: Gather context for art direction.

        Args:
            task: The TaskDefinition for art direction

        Returns:
            AgentResponse with gathered context or error
        """
        logger.info(f"ArtDirector gathering context for: {task.task_id}")

        try:
            # Extract request and theme
            request = task.context.get("asset_request", "")
            theme = task.context.get("theme", None)

            if not request:
                return AgentResponse(
                    success=False,
                    error=AgentError(
                        error_type="missing_request",
                        message="No asset_request found in task context",
                        timestamp=datetime.utcnow(),
                        context={"task_id": task.task_id},
                        recoverable=False
                    )
                )

            context_data = {
                "asset_request": request,
                "theme": theme,
                "request_length": len(request)
            }

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
        Phase 2: Decompose request and create tasks.

        Args:
            task: The art direction TaskDefinition
            context: Context data from gather_context

        Returns:
            AgentResponse with decomposed tasks or error
        """
        logger.info(f"ArtDirector decomposing for: {task.task_id}")

        try:
            request = context["asset_request"]
            theme = context.get("theme")

            # Decompose request into tasks
            asset_tasks = self.decompose_asset_request(
                request=request,
                theme=theme
            )

            result = {
                "asset_tasks": [t.to_strict_json() for t in asset_tasks],
                "task_count": len(asset_tasks),
                "master_style_prompt": asset_tasks[0].context.get("style_prompt") if asset_tasks else None
            }

            logger.info(f"Decomposed into {len(asset_tasks)} tasks")

            return AgentResponse(
                success=True,
                data=result,
                metadata={"agent_id": self.agent_id, "phase": "take_action"}
            )

        except Exception as e:
            error = AgentError(
                error_type="decomposition_error",
                message=str(e),
                timestamp=datetime.utcnow(),
                context={"task_id": task.task_id},
                recoverable=False
            )
            logger.error(f"Error during decomposition: {error}")
            return AgentResponse(success=False, error=error)

    def verify_work(self, task: TaskDefinition, action_result: Dict[str, Any]) -> AgentResponse:
        """
        Phase 3: Verify decomposition quality.

        Args:
            task: The original TaskDefinition
            action_result: Results from take_action phase

        Returns:
            AgentResponse indicating verification success or failure
        """
        logger.info(f"ArtDirector verifying decomposition for: {task.task_id}")

        try:
            asset_tasks = action_result["asset_tasks"]
            master_style = action_result["master_style_prompt"]

            validation_results = {
                "has_tasks": len(asset_tasks) > 0,
                "has_master_style": master_style is not None,
                "all_tasks_valid": all(
                    "asset_description" in t["context"] for t in asset_tasks
                ),
                "style_consistent": all(
                    t["context"].get("style_prompt") == master_style
                    for t in asset_tasks
                )
            }

            validation_passed = all(validation_results.values())

            verification_data = {
                "verification_passed": validation_passed,
                "validation_results": validation_results,
                "task_count": len(asset_tasks)
            }

            if validation_passed:
                logger.info(f"Decomposition verification passed for {task.task_id}")
                return AgentResponse(
                    success=True,
                    data=verification_data,
                    metadata={"agent_id": self.agent_id, "phase": "verify_work"}
                )
            else:
                error = AgentError(
                    error_type="verification_failed",
                    message="Decomposition verification failed",
                    timestamp=datetime.utcnow(),
                    context={
                        "task_id": task.task_id,
                        "validation_results": validation_results
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

    def _load_style_templates(self) -> Dict[str, str]:
        """
        Load predefined style templates.

        Returns:
            Dictionary of theme -> style template
        """
        return {
            "medieval": "Low Poly, PBR materials, hand-painted textures, stylized medieval aesthetic",
            "scifi": "Hard surface modeling, metallic PBR, clean geometric shapes, futuristic design",
            "fantasy": "Vibrant colors, magical elements, soft lighting, whimsical proportions",
            "horror": "Dark atmosphere, gritty textures, unsettling proportions, dramatic lighting",
            "cartoon": "Exaggerated features, bright colors, simple shapes, cel-shaded appearance",
            "realistic": "High poly, photorealistic PBR, accurate proportions, detailed textures",
            "lowpoly": "Low poly count, flat colors or simple gradients, sharp edges, minimalist",
            "voxel": "Cubic voxel style, blocky appearance, pixelated textures, Minecraft-like"
        }

    def _generate_style_prompt(self, theme: str) -> str:
        """
        Generate a style prompt based on theme.

        Simulates Sonnet generation. In production, would call actual API.

        Args:
            theme: Theme string

        Returns:
            Generated style prompt
        """
        # Normalize theme
        theme_lower = theme.lower()

        # Check for matching template
        for key, template in self.style_templates.items():
            if key in theme_lower:
                base_style = template
                break
        else:
            # Default style
            base_style = "Low Poly, PBR materials, game-ready, optimized topology"

        # Enhance with theme-specific details
        style_prompt = f"{base_style}, {theme} theme, consistent art style, production-ready"

        return style_prompt

    def _decompose_request(self, request: str, theme: Optional[str]) -> List[Dict[str, Any]]:
        """
        Decompose request into individual assets.

        Simulates Sonnet decomposition. In production, would call actual API.

        Args:
            request: Asset request string
            theme: Optional theme

        Returns:
            List of asset info dictionaries
        """
        request_lower = request.lower()

        # Pattern matching for common requests
        if "dungeon" in request_lower:
            assets = [
                {"name": "Stone Wall", "description": "Dungeon stone wall section", "priority": 1},
                {"name": "Stone Floor", "description": "Dungeon stone floor tile", "priority": 1},
                {"name": "Wooden Door", "description": "Dungeon wooden door", "priority": 2},
                {"name": "Treasure Chest", "description": "Dungeon treasure chest", "priority": 2},
                {"name": "Torch", "description": "Wall-mounted dungeon torch", "priority": 3}
            ]
        elif "spaceship" in request_lower or "space station" in request_lower:
            assets = [
                {"name": "Hull Panel", "description": "Sci-fi hull panel section", "priority": 1},
                {"name": "Control Console", "description": "Spaceship control console", "priority": 2},
                {"name": "Airlock Door", "description": "Sci-fi airlock door", "priority": 2},
                {"name": "Cargo Container", "description": "Futuristic cargo container", "priority": 3}
            ]
        elif "furniture" in request_lower:
            assets = [
                {"name": "Chair", "description": "Simple chair", "priority": 1},
                {"name": "Table", "description": "Simple table", "priority": 1},
                {"name": "Shelf", "description": "Wall shelf", "priority": 2},
                {"name": "Lamp", "description": "Desk or floor lamp", "priority": 2}
            ]
        elif "weapon" in request_lower:
            assets = [
                {"name": "Sword", "description": "Medieval sword", "priority": 1},
                {"name": "Shield", "description": "Medieval shield", "priority": 1},
                {"name": "Bow", "description": "Medieval bow", "priority": 2},
                {"name": "Axe", "description": "Medieval battle axe", "priority": 2}
            ]
        else:
            # Generic decomposition
            assets = [
                {"name": "Asset 1", "description": f"Primary asset for {request}", "priority": 1},
                {"name": "Asset 2", "description": f"Secondary asset for {request}", "priority": 2}
            ]

        return assets

    def _create_asset_task(
        self,
        asset_info: Dict[str, Any],
        style_prompt: str,
        index: int,
        request_id: str
    ) -> TaskDefinition:
        """
        Create a TaskDefinition for an individual asset.

        Args:
            asset_info: Asset information dictionary
            style_prompt: Master style prompt
            index: Index in the asset list
            request_id: Parent request ID

        Returns:
            TaskDefinition for asset generation
        """
        task_id = f"{request_id}_asset_{index}"

        task = TaskDefinition(
            task_id=task_id,
            task_type=TaskType.IMPLEMENTATION,  # Asset generation is implementation
            priority=asset_info.get("priority", 5),
            assigned_agent="asset_agent",
            context={
                "asset_description": asset_info["description"],
                "asset_name": asset_info["name"],
                "style_prompt": style_prompt,
                "request_id": request_id,
                "index": index
            },
            requirements={
                "quality": "medium",
                "format": "glb",
                "style_consistency": True
            }
        )

        return task

    def handle_verification_failure(
        self,
        failure_report: Dict[str, Any],
        original_task: TaskDefinition
    ) -> TaskDefinition:
        """
        Handle asset verification failure by creating a corrected task.

        Analyzes validation issues and generates a new task with adjusted
        prompts and constraints to fix the problems.

        Args:
            failure_report: Validation failure report with metrics and issues
            original_task: The original asset generation task

        Returns:
            New TaskDefinition with corrected requirements
        """
        logger.info("ArtDirector handling verification failure")

        # Extract information from original task
        original_description = original_task.context.get("asset_description", "")
        original_style = original_task.context.get("style_prompt", "")
        asset_name = original_task.context.get("asset_name", "Asset")

        # Extract validation issues
        issues = failure_report.get("issues", [])
        metrics = failure_report.get("metrics", {})

        logger.info(f"Validation issues: {issues}")

        # Analyze issues and build corrections
        corrections = []
        adjusted_requirements = {}

        for issue in issues:
            issue_lower = issue.lower()

            # Triangle count too high
            if "triangle count" in issue_lower or "poly" in issue_lower:
                triangle_count = metrics.get("triangle_count", 0)
                # Extract max from constraints or use default
                constraints = failure_report.get("constraints", {})
                max_allowed = constraints.get("max_triangles", 50000)
                # Target is 70% of maximum allowed (to leave safety margin)
                target = int(max_allowed * 0.7)
                corrections.append(f"Reduce polygon count to approximately {target} triangles")
                adjusted_requirements["max_triangles"] = target

            # Missing UV map
            elif "uv map" in issue_lower:
                corrections.append("Ensure proper UV mapping is applied")
                adjusted_requirements["require_uv_map"] = True

            # Dimensions too large
            elif "max dimension" in issue_lower:
                max_dim = metrics.get("dimensions", {}).get("max", 100)
                target = max_dim * 0.8  # Reduce by 20%
                corrections.append(f"Scale down to maximum {target:.1f} units")
                adjusted_requirements["max_dimensions"] = target

            # Dimensions too small
            elif "min dimension" in issue_lower:
                min_dim = metrics.get("dimensions", {}).get("min", 0)
                target = min_dim * 1.5  # Increase by 50%
                corrections.append(f"Scale up to minimum {target:.1f} units")
                adjusted_requirements["min_dimensions"] = target

            # Too many materials
            elif "material count" in issue_lower:
                mat_count = metrics.get("material_count", 0)
                target = max(mat_count - 2, 1)
                corrections.append(f"Reduce to {target} materials or fewer")
                adjusted_requirements["max_materials"] = target

        # Build corrected prompt
        if corrections:
            correction_text = ". ".join(corrections)
            corrected_description = f"{original_description}. CORRECTIONS: {correction_text}"
        else:
            corrected_description = f"{original_description}. Regenerate with improved quality"

        # Track token usage for analysis (Sonnet)
        input_tokens = len(str(failure_report)) // 4 + 100
        output_tokens = len(corrected_description) // 4

        cost = self.token_tracker.track_usage(
            model=ModelType.SONNET_3_5,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "agent_id": self.agent_id,
                "operation": "handle_verification_failure",
                "original_task_id": original_task.task_id
            }
        )

        # Create new task
        retry_task_id = f"{original_task.task_id}_retry_{datetime.utcnow().strftime('%H%M%S')}"

        retry_task = TaskDefinition(
            task_id=retry_task_id,
            task_type=TaskType.IMPLEMENTATION,
            priority=1,  # High priority for retries
            assigned_agent="asset_agent",
            context={
                "asset_description": corrected_description,
                "asset_name": f"{asset_name} (Retry)",
                "style_prompt": original_style,
                "original_task_id": original_task.task_id,
                "retry": True,
                "correction_reason": corrections
            },
            requirements={
                **original_task.requirements,
                **adjusted_requirements,
                "is_retry": True
            }
        )

        logger.info(
            f"Created retry task {retry_task_id} with {len(corrections)} corrections, "
            f"cost: ${cost:.4f}"
        )

        return retry_task

    def __repr__(self) -> str:
        return (
            f"ArtDirector(id='{self.agent_id}', "
            f"name='{self.agent_name}', "
            f"temperature={self.temperature})"
        )
