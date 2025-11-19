"""
Asset Generation Tests - Testing asset pipeline components
Tests for API integrations, AssetAgent, and ArtDirector
"""

import pytest
import os
import shutil
import tempfile
from pathlib import Path

from integrations.asset_apis import AssetGeneratorAPI, TripoAPI, SloydAPI, MeshyAPI
from agents.workers.asset_agent import AssetAgent
from agents.managers.art_director import ArtDirector
from core.tdf_schema import TaskDefinition, TaskType


class TestAssetAPIs:
    """Test suite for Asset Generation APIs"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for tests"""
        # Setup: Create temporary output directory
        self.temp_dir = tempfile.mkdtemp(prefix="test_assets_")
        yield
        # Teardown: Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_tripo_api_initialization(self):
        """Test TripoAPI initialization"""
        api = TripoAPI(output_dir=self.temp_dir)

        assert api.output_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
        assert api.success_rate == 0.9
        assert "glb" in api.get_supported_formats()

    def test_tripo_api_generate_model(self):
        """Test TripoAPI model generation"""
        # Force success for deterministic testing
        api = TripoAPI(
            output_dir=self.temp_dir,
            config={"success_rate": 1.0, "latency_range": (0.01, 0.02)}
        )

        result = api.generate_model(
            prompt="Simple cube",
            style_prompt="Low Poly, PBR"
        )

        assert result["success"] is True
        assert result["file_path"] is not None
        assert os.path.exists(result["file_path"])
        assert result["metadata"]["api"] == "tripo"
        assert result["metadata"]["prompt"] == "Simple cube"
        assert result["metadata"]["style_prompt"] == "Low Poly, PBR"
        assert result["error"] is None

    def test_tripo_api_failure(self):
        """Test TripoAPI failure simulation"""
        # Force failure
        api = TripoAPI(
            output_dir=self.temp_dir,
            config={"success_rate": 0.0, "latency_range": (0.01, 0.02)}
        )

        result = api.generate_model(prompt="Test asset")

        assert result["success"] is False
        assert result["file_path"] is None
        assert result["error"] is not None

    def test_sloyd_api_initialization(self):
        """Test SloydAPI initialization"""
        api = SloydAPI(output_dir=self.temp_dir)

        assert api.output_dir == self.temp_dir
        assert api.success_rate == 0.85
        assert "fbx" in api.get_supported_formats()

    def test_sloyd_api_generate_model(self):
        """Test SloydAPI model generation"""
        api = SloydAPI(
            output_dir=self.temp_dir,
            config={"success_rate": 1.0, "latency_range": (0.01, 0.02)}
        )

        result = api.generate_model(
            prompt="Game-ready tree",
            style_prompt="Low Poly"
        )

        assert result["success"] is True
        assert result["file_path"] is not None
        assert os.path.exists(result["file_path"])
        assert result["metadata"]["api"] == "sloyd"
        assert result["metadata"]["game_ready"] is True
        assert result["metadata"]["pbr_materials"] is True

    def test_meshy_api_initialization(self):
        """Test MeshyAPI initialization"""
        api = MeshyAPI(output_dir=self.temp_dir)

        assert api.output_dir == self.temp_dir
        assert api.success_rate == 0.8
        assert "glb" in api.get_supported_formats()
        assert "obj" in api.get_supported_formats()

    def test_meshy_api_generate_model(self):
        """Test MeshyAPI model generation"""
        api = MeshyAPI(
            output_dir=self.temp_dir,
            config={"success_rate": 1.0, "latency_range": (0.01, 0.02)}
        )

        result = api.generate_model(
            prompt="High quality character",
            style_prompt="Realistic, PBR"
        )

        assert result["success"] is True
        assert result["file_path"] is not None
        assert os.path.exists(result["file_path"])
        assert result["metadata"]["api"] == "meshy"
        assert result["metadata"]["quality"] == "high"
        assert result["metadata"]["texture_resolution"] == "2048x2048"

    def test_api_output_path_creation(self):
        """Test output path creation"""
        api = TripoAPI(output_dir=self.temp_dir)

        path = api._create_output_path("Test Asset", "glb")

        assert self.temp_dir in path
        assert path.endswith(".glb")
        assert "Test_Asset" in path or "test_asset" in path.lower()

    def test_all_apis_support_formats(self):
        """Test that all APIs support their claimed formats"""
        apis = [
            TripoAPI(output_dir=self.temp_dir),
            SloydAPI(output_dir=self.temp_dir),
            MeshyAPI(output_dir=self.temp_dir)
        ]

        for api in apis:
            formats = api.get_supported_formats()
            assert isinstance(formats, list)
            assert len(formats) > 0
            assert all(isinstance(fmt, str) for fmt in formats)


class TestAssetAgent:
    """Test suite for AssetAgent"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for tests"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_agent_")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_asset_agent_initialization(self):
        """Test AssetAgent initialization"""
        agent = AssetAgent(
            agent_id="test_asset_agent",
            config={"output_dir": self.temp_dir}
        )

        assert agent.agent_id == "test_asset_agent"
        assert agent.output_dir == self.temp_dir
        assert len(agent.apis) == 3  # tripo, sloyd, meshy
        assert "tripo" in agent.apis
        assert "sloyd" in agent.apis
        assert "meshy" in agent.apis

    def test_asset_agent_gather_context(self):
        """Test AssetAgent context gathering"""
        agent = AssetAgent(
            agent_id="test_agent",
            config={"output_dir": self.temp_dir}
        )

        task = TaskDefinition(
            task_id="test_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={
                "asset_description": "Medieval sword",
                "style_prompt": "Low Poly, PBR",
                "preferred_api": "tripo"
            }
        )

        response = agent.gather_context(task)

        assert response.success is True
        assert response.data["asset_description"] == "Medieval sword"
        assert response.data["style_prompt"] == "Low Poly, PBR"
        assert response.data["preferred_api"] == "tripo"

    def test_asset_agent_missing_description(self):
        """Test AssetAgent with missing asset description"""
        agent = AssetAgent(
            agent_id="test_agent",
            config={"output_dir": self.temp_dir}
        )

        task = TaskDefinition(
            task_id="test_002",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={}  # Missing asset_description
        )

        response = agent.gather_context(task)

        assert response.success is False
        assert response.error is not None
        assert response.error.error_type == "missing_description"

    def test_asset_agent_api_selection(self):
        """Test AssetAgent API selection logic"""
        agent = AssetAgent(
            agent_id="test_agent",
            config={"output_dir": self.temp_dir}
        )

        # Test game asset -> Sloyd
        api = agent._select_api(
            asset_description="Game-ready low poly tree",
            quality="medium",
            preferred_api="tripo"
        )
        assert api == "sloyd"

        # Test high quality -> Meshy
        api = agent._select_api(
            asset_description="Detailed character model",
            quality="high quality",
            preferred_api="tripo"
        )
        assert api == "meshy"

        # Test PBR mention -> Meshy
        api = agent._select_api(
            asset_description="PBR textured building",
            quality="medium",
            preferred_api="tripo"
        )
        assert api == "meshy"

        # Test default -> preferred
        api = agent._select_api(
            asset_description="Simple object",
            quality="medium",
            preferred_api="sloyd"
        )
        assert api == "sloyd"

    def test_asset_agent_fallback_api(self):
        """Test AssetAgent fallback API selection"""
        agent = AssetAgent(
            agent_id="test_agent",
            config={"output_dir": self.temp_dir}
        )

        # Test round-robin fallback
        fallback1 = agent._select_fallback_api("tripo")
        assert fallback1 in ["sloyd", "meshy"]

        fallback2 = agent._select_fallback_api(fallback1)
        assert fallback2 != fallback1

    def test_asset_agent_full_execution(self):
        """Test AssetAgent full execution lifecycle"""
        # Configure all APIs for guaranteed success and fast execution
        config = {
            "output_dir": self.temp_dir,
            "max_retries": 0  # No retries for faster test
        }

        agent = AssetAgent(agent_id="test_agent", config=config)

        # Reconfigure APIs for success
        for api_name in agent.apis:
            agent.apis[api_name] = type(agent.apis[api_name])(
                output_dir=self.temp_dir,
                config={"success_rate": 1.0, "latency_range": (0.01, 0.02)}
            )

        task = TaskDefinition(
            task_id="test_003",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={
                "asset_description": "Simple cube",
                "style_prompt": "Low Poly"
            }
        )

        response = agent.execute_task(task)

        assert response.success is True
        assert response.data["verification_passed"] is True


class TestArtDirector:
    """Test suite for ArtDirector"""

    def test_art_director_initialization(self):
        """Test ArtDirector initialization"""
        director = ArtDirector(
            agent_id="test_director",
            agent_name="TestDirector"
        )

        assert director.agent_id == "test_director"
        assert director.agent_name == "TestDirector"
        assert director.temperature == 0.8
        assert len(director.style_templates) > 0

    def test_art_director_create_master_style_prompt(self):
        """Test master style prompt creation"""
        director = ArtDirector(agent_id="test_director")

        # Test medieval theme
        style_prompt = director.create_master_style_prompt("Medieval Dungeon")

        assert isinstance(style_prompt, str)
        assert len(style_prompt) > 0
        assert "medieval" in style_prompt.lower() or "dungeon" in style_prompt.lower()
        assert "Low Poly" in style_prompt or "PBR" in style_prompt

    def test_art_director_style_consistency(self):
        """Test that style prompts are consistent for same theme"""
        director = ArtDirector(agent_id="test_director")

        style1 = director.create_master_style_prompt("Sci-Fi")
        style2 = director.create_master_style_prompt("Sci-Fi Space Station")

        # Both should contain sci-fi related styling
        assert "scifi" in style1.lower() or "sci-fi" in style1.lower() or "futuristic" in style1.lower()
        assert "scifi" in style2.lower() or "sci-fi" in style2.lower() or "futuristic" in style2.lower()

    def test_art_director_decompose_dungeon_request(self):
        """Test decomposing a dungeon set request"""
        director = ArtDirector(agent_id="test_director")

        tasks = director.decompose_asset_request(
            request="Create a medieval dungeon set",
            theme="Medieval"
        )

        assert len(tasks) > 0
        assert all(isinstance(task, TaskDefinition) for task in tasks)
        assert all(task.task_type == TaskType.IMPLEMENTATION for task in tasks)
        assert all(task.assigned_agent == "asset_agent" for task in tasks)

        # Check that all tasks have the same style prompt (consistency!)
        style_prompts = [task.context.get("style_prompt") for task in tasks]
        assert len(set(style_prompts)) == 1  # All should be the same

        # Check that tasks have asset descriptions
        assert all("asset_description" in task.context for task in tasks)

    def test_art_director_decompose_spaceship_request(self):
        """Test decomposing a spaceship request"""
        director = ArtDirector(agent_id="test_director")

        tasks = director.decompose_asset_request(
            request="Create a spaceship interior",
            theme="Sci-Fi"
        )

        assert len(tasks) > 0

        # Verify all tasks share the same style
        style_prompts = [task.context.get("style_prompt") for task in tasks]
        assert len(set(style_prompts)) == 1

    def test_art_director_decompose_furniture_request(self):
        """Test decomposing a furniture request"""
        director = ArtDirector(agent_id="test_director")

        tasks = director.decompose_asset_request(
            request="Create basic furniture set"
        )

        assert len(tasks) > 0
        assert all(isinstance(task, TaskDefinition) for task in tasks)

    def test_art_director_full_execution(self):
        """Test ArtDirector full execution lifecycle"""
        director = ArtDirector(agent_id="test_director")

        task = TaskDefinition(
            task_id="art_dir_001",
            task_type=TaskType.ANALYSIS,
            priority=1,
            assigned_agent="art_director",
            context={
                "asset_request": "Create a dungeon set",
                "theme": "Medieval"
            }
        )

        response = director.execute_task(task)

        assert response.success is True
        assert response.data["verification_passed"] is True
        assert response.data["task_count"] > 0

    def test_art_director_style_templates(self):
        """Test that style templates exist for common themes"""
        director = ArtDirector(agent_id="test_director")

        templates = director.style_templates

        # Check for common themes
        assert "medieval" in templates
        assert "scifi" in templates
        assert "fantasy" in templates
        assert "lowpoly" in templates

        # All templates should be non-empty strings
        assert all(isinstance(template, str) and len(template) > 0
                   for template in templates.values())


class TestAssetPipelineIntegration:
    """Integration tests for the complete asset generation pipeline"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for integration tests"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_pipeline_")
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_asset_generation(self):
        """Test complete pipeline: ArtDirector -> AssetAgent -> File"""
        # Step 1: ArtDirector decomposes request
        director = ArtDirector(agent_id="director_001")

        asset_tasks = director.decompose_asset_request(
            request="Create a simple dungeon",
            theme="Medieval"
        )

        assert len(asset_tasks) > 0

        # Step 2: AssetAgent generates first asset
        agent = AssetAgent(
            agent_id="agent_001",
            config={"output_dir": self.temp_dir, "max_retries": 0}
        )

        # Configure for success
        for api_name in agent.apis:
            agent.apis[api_name] = type(agent.apis[api_name])(
                output_dir=self.temp_dir,
                config={"success_rate": 1.0, "latency_range": (0.01, 0.02)}
            )

        # Execute first task
        first_task = asset_tasks[0]
        response = agent.execute_task(first_task)

        # Step 3: Verify results
        assert response.success is True
        assert response.data["verification_passed"] is True

        # Check file was created
        gen_result = response.data["generation_result"]
        assert gen_result["success"] is True
        assert os.path.exists(gen_result["file_path"])

    def test_style_consistency_across_assets(self):
        """Test that all assets in a set share the same style"""
        director = ArtDirector(agent_id="director_002")

        tasks = director.decompose_asset_request(
            request="Create weapon set",
            theme="Medieval"
        )

        # Extract all style prompts
        style_prompts = [task.context.get("style_prompt") for task in tasks]

        # All should be identical (style consistency!)
        assert all(sp == style_prompts[0] for sp in style_prompts)
        assert "medieval" in style_prompts[0].lower() or "Medieval" in style_prompts[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
