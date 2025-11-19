"""
Asset Verification Tests - Testing verification pipeline components
Tests for VerifierAgent, Blender integration, and ArtDirector retry logic
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from agents.workers.verifier_agent import VerifierAgent, ValidationResult
from agents.managers.art_director import ArtDirector
from core.tdf_schema import TaskDefinition, TaskType


class TestValidationResult:
    """Test suite for ValidationResult class"""

    def test_validation_result_creation(self):
        """Test creating ValidationResult"""
        result = ValidationResult(
            success=True,
            validation_passed=True,
            metrics={"triangle_count": 1000},
            issues=[]
        )

        assert result.success is True
        assert result.validation_passed is True
        assert result.metrics["triangle_count"] == 1000
        assert len(result.issues) == 0
        assert result.error is None

    def test_validation_result_with_issues(self):
        """Test ValidationResult with validation issues"""
        result = ValidationResult(
            success=True,
            validation_passed=False,
            metrics={"triangle_count": 60000},
            issues=["Triangle count (60000) exceeds maximum (50000)"]
        )

        assert result.success is True
        assert result.validation_passed is False
        assert len(result.issues) == 1

    def test_validation_result_to_dict(self):
        """Test ValidationResult to_dict conversion"""
        result = ValidationResult(
            success=True,
            validation_passed=True,
            metrics={"triangle_count": 1000},
            issues=[]
        )

        dict_result = result.to_dict()

        assert isinstance(dict_result, dict)
        assert dict_result["success"] is True
        assert dict_result["validation_passed"] is True
        assert "metrics" in dict_result
        assert "issues" in dict_result

    def test_validation_result_repr(self):
        """Test ValidationResult repr"""
        result = ValidationResult(
            success=True,
            validation_passed=True,
            metrics={},
            issues=[]
        )

        repr_str = repr(result)
        assert "PASSED" in repr_str


class TestVerifierAgent:
    """Test suite for VerifierAgent"""

    @pytest.fixture
    def temp_asset_file(self):
        """Create a temporary asset file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.glb', delete=False) as f:
            f.write("# Dummy GLB file\n")
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_verifier_agent_initialization(self):
        """Test VerifierAgent initialization"""
        agent = VerifierAgent(
            agent_id="test_verifier",
            config={"blender_path": "/usr/bin/blender"}
        )

        assert agent.agent_id == "test_verifier"
        assert agent.blender_path == "/usr/bin/blender"
        assert agent.timeout == 60
        assert "max_triangles" in agent.default_constraints

    def test_verifier_agent_verify_asset_file_not_found(self):
        """Test verification with non-existent file"""
        agent = VerifierAgent(agent_id="test_verifier")

        result = agent.verify_asset("/nonexistent/file.glb")

        assert result.success is False
        assert result.validation_passed is False
        assert "not found" in result.error.lower()

    @patch('subprocess.run')
    def test_verifier_agent_verify_asset_success(self, mock_run, temp_asset_file):
        """Test successful asset verification with mocked Blender"""
        agent = VerifierAgent(agent_id="test_verifier")

        # Mock Blender output
        blender_output = {
            "success": True,
            "validation_passed": True,
            "metrics": {
                "triangle_count": 1000,
                "vertex_count": 500,
                "has_uv_map": True,
                "material_count": 1,
                "dimensions": {
                    "width": 5.0,
                    "height": 3.0,
                    "depth": 2.0,
                    "max": 5.0,
                    "min": 2.0
                }
            },
            "issues": [],
            "error": None
        }

        mock_run.return_value = Mock(
            stdout=json.dumps(blender_output),
            stderr="",
            returncode=0
        )

        result = agent.verify_asset(temp_asset_file)

        assert result.success is True
        assert result.validation_passed is True
        assert result.metrics["triangle_count"] == 1000
        assert len(result.issues) == 0

    @patch('subprocess.run')
    def test_verifier_agent_verify_asset_failure(self, mock_run, temp_asset_file):
        """Test asset verification failure with too many triangles"""
        agent = VerifierAgent(agent_id="test_verifier")

        # Mock Blender output with validation failure
        blender_output = {
            "success": True,
            "validation_passed": False,
            "metrics": {
                "triangle_count": 75000,
                "vertex_count": 40000,
                "has_uv_map": True,
                "material_count": 1
            },
            "issues": ["Triangle count (75000) exceeds maximum (50000)"],
            "error": None
        }

        mock_run.return_value = Mock(
            stdout=json.dumps(blender_output),
            stderr="",
            returncode=1
        )

        result = agent.verify_asset(temp_asset_file)

        assert result.success is True
        assert result.validation_passed is False
        assert len(result.issues) == 1
        assert "Triangle count" in result.issues[0]

    @patch('subprocess.run')
    def test_verifier_agent_verify_asset_missing_uv(self, mock_run, temp_asset_file):
        """Test asset verification failure with missing UV map"""
        agent = VerifierAgent(agent_id="test_verifier")

        blender_output = {
            "success": True,
            "validation_passed": False,
            "metrics": {
                "triangle_count": 1000,
                "has_uv_map": False
            },
            "issues": ["No UV map found"],
            "error": None
        }

        mock_run.return_value = Mock(
            stdout=json.dumps(blender_output),
            stderr="",
            returncode=1
        )

        result = agent.verify_asset(temp_asset_file)

        assert result.success is True
        assert result.validation_passed is False
        assert "UV map" in result.issues[0]

    @patch('subprocess.run')
    def test_verifier_agent_blender_timeout(self, mock_run, temp_asset_file):
        """Test Blender timeout handling"""
        from subprocess import TimeoutExpired

        agent = VerifierAgent(
            agent_id="test_verifier",
            config={"timeout": 1}
        )

        mock_run.side_effect = TimeoutExpired("blender", 1)

        result = agent.verify_asset(temp_asset_file)

        assert result.success is False
        assert "timed out" in result.error.lower()

    @patch('subprocess.run')
    def test_verifier_agent_blender_not_found(self, mock_run, temp_asset_file):
        """Test Blender executable not found"""
        agent = VerifierAgent(
            agent_id="test_verifier",
            config={"blender_path": "/nonexistent/blender"}
        )

        mock_run.side_effect = FileNotFoundError("blender not found")

        result = agent.verify_asset(temp_asset_file)

        assert result.success is False
        assert "not found" in result.error.lower()

    @patch('subprocess.run')
    def test_verifier_agent_custom_constraints(self, mock_run, temp_asset_file):
        """Test verification with custom constraints"""
        agent = VerifierAgent(agent_id="test_verifier")

        blender_output = {
            "success": True,
            "validation_passed": True,
            "metrics": {"triangle_count": 15000},
            "issues": [],
            "error": None
        }

        mock_run.return_value = Mock(
            stdout=json.dumps(blender_output),
            stderr="",
            returncode=0
        )

        custom_constraints = {
            "max_triangles": 20000,
            "require_uv_map": False
        }

        result = agent.verify_asset(temp_asset_file, custom_constraints)

        # Verify that custom constraints were passed
        call_args = mock_run.call_args
        assert custom_constraints["max_triangles"] == 20000

    def test_verifier_agent_gather_context(self):
        """Test VerifierAgent context gathering"""
        agent = VerifierAgent(agent_id="test_verifier")

        task = TaskDefinition(
            task_id="verify_001",
            task_type=TaskType.REVIEW,
            priority=1,
            assigned_agent="verifier_agent",
            context={
                "file_path": "/path/to/asset.glb",
                "constraints": {"max_triangles": 30000}
            }
        )

        response = agent.gather_context(task)

        assert response.success is True
        assert response.data["file_path"] == "/path/to/asset.glb"
        assert response.data["constraints"]["max_triangles"] == 30000

    def test_verifier_agent_missing_file_path(self):
        """Test VerifierAgent with missing file path"""
        agent = VerifierAgent(agent_id="test_verifier")

        task = TaskDefinition(
            task_id="verify_002",
            task_type=TaskType.REVIEW,
            priority=1,
            assigned_agent="verifier_agent",
            context={}  # Missing file_path
        )

        response = agent.gather_context(task)

        assert response.success is False
        assert response.error is not None
        assert response.error.error_type == "missing_file_path"


class TestArtDirectorVerification:
    """Test suite for ArtDirector verification failure handling"""

    def test_art_director_handle_verification_failure_poly_count(self):
        """Test handling verification failure due to high poly count"""
        director = ArtDirector(agent_id="test_director")

        # Original task
        original_task = TaskDefinition(
            task_id="asset_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={
                "asset_description": "Medieval sword",
                "asset_name": "Sword",
                "style_prompt": "Low Poly, PBR"
            },
            requirements={}
        )

        # Failure report
        failure_report = {
            "success": True,
            "validation_passed": False,
            "metrics": {
                "triangle_count": 75000,
                "vertex_count": 40000
            },
            "issues": ["Triangle count (75000) exceeds maximum (50000)"]
        }

        # Handle failure
        retry_task = director.handle_verification_failure(failure_report, original_task)

        assert retry_task.task_id != original_task.task_id
        assert "_retry_" in retry_task.task_id
        assert retry_task.priority == 1  # High priority
        assert retry_task.context["retry"] is True
        assert "CORRECTIONS" in retry_task.context["asset_description"]
        assert "Reduce polygon count" in retry_task.context["asset_description"]
        assert "max_triangles" in retry_task.requirements

    def test_art_director_handle_verification_failure_uv_map(self):
        """Test handling verification failure due to missing UV map"""
        director = ArtDirector(agent_id="test_director")

        original_task = TaskDefinition(
            task_id="asset_002",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={
                "asset_description": "Stone wall",
                "asset_name": "Wall",
                "style_prompt": "Medieval"
            },
            requirements={}
        )

        failure_report = {
            "success": True,
            "validation_passed": False,
            "metrics": {
                "triangle_count": 5000,
                "has_uv_map": False
            },
            "issues": ["No UV map found"]
        }

        retry_task = director.handle_verification_failure(failure_report, original_task)

        assert "UV mapping" in retry_task.context["asset_description"]
        assert retry_task.requirements["require_uv_map"] is True

    def test_art_director_handle_verification_failure_dimensions(self):
        """Test handling verification failure due to dimensions"""
        director = ArtDirector(agent_id="test_director")

        original_task = TaskDefinition(
            task_id="asset_003",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={
                "asset_description": "Character model",
                "asset_name": "Character",
                "style_prompt": "Realistic"
            },
            requirements={}
        )

        failure_report = {
            "success": True,
            "validation_passed": False,
            "metrics": {
                "triangle_count": 10000,
                "dimensions": {
                    "max": 150.0,
                    "min": 120.0
                }
            },
            "issues": ["Max dimension (150.0) exceeds limit (100.0)"]
        }

        retry_task = director.handle_verification_failure(failure_report, original_task)

        assert "Scale down" in retry_task.context["asset_description"]
        assert "max_dimensions" in retry_task.requirements

    def test_art_director_handle_verification_failure_multiple_issues(self):
        """Test handling multiple validation issues"""
        director = ArtDirector(agent_id="test_director")

        original_task = TaskDefinition(
            task_id="asset_004",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={
                "asset_description": "Complex prop",
                "asset_name": "Prop",
                "style_prompt": "Game Ready"
            },
            requirements={}
        )

        failure_report = {
            "success": True,
            "validation_passed": False,
            "metrics": {
                "triangle_count": 80000,
                "has_uv_map": False,
                "material_count": 15
            },
            "issues": [
                "Triangle count (80000) exceeds maximum (50000)",
                "No UV map found",
                "Material count (15) exceeds maximum (10)"
            ]
        }

        retry_task = director.handle_verification_failure(failure_report, original_task)

        # Should have multiple corrections
        corrections = retry_task.context["correction_reason"]
        assert len(corrections) == 3
        assert any("polygon" in c.lower() for c in corrections)
        assert any("uv" in c.lower() for c in corrections)
        assert any("material" in c.lower() for c in corrections)


class TestVerificationPipelineIntegration:
    """Integration tests for the verification pipeline"""

    @pytest.fixture
    def temp_asset_file(self):
        """Create a temporary asset file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.glb', delete=False) as f:
            f.write("# Dummy GLB file\n")
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @patch('subprocess.run')
    def test_end_to_end_verification_pass(self, mock_run, temp_asset_file):
        """Test complete verification pipeline with passing asset"""
        # Step 1: Create verification task
        verify_task = TaskDefinition(
            task_id="verify_001",
            task_type=TaskType.REVIEW,
            priority=1,
            assigned_agent="verifier_agent",
            context={
                "file_path": temp_asset_file,
                "constraints": {"max_triangles": 50000}
            }
        )

        # Step 2: VerifierAgent validates
        agent = VerifierAgent(agent_id="verifier_001")

        # Mock passing validation
        blender_output = {
            "success": True,
            "validation_passed": True,
            "metrics": {"triangle_count": 10000, "has_uv_map": True},
            "issues": [],
            "error": None
        }

        mock_run.return_value = Mock(
            stdout=json.dumps(blender_output),
            stderr="",
            returncode=0
        )

        response = agent.execute_task(verify_task)

        # Step 3: Verify results
        assert response.success is True
        assert response.data["verification_passed"] is True

    @patch('subprocess.run')
    def test_end_to_end_verification_fail_and_retry(self, mock_run, temp_asset_file):
        """Test verification failure and ArtDirector retry generation"""
        # Step 1: Original asset task
        original_task = TaskDefinition(
            task_id="asset_001",
            task_type=TaskType.IMPLEMENTATION,
            priority=1,
            assigned_agent="asset_agent",
            context={
                "asset_description": "Detailed character",
                "asset_name": "Character",
                "style_prompt": "Realistic, PBR"
            },
            requirements={}
        )

        # Step 2: Verification fails
        agent = VerifierAgent(agent_id="verifier_001")

        blender_output = {
            "success": True,
            "validation_passed": False,
            "metrics": {"triangle_count": 100000},
            "issues": ["Triangle count (100000) exceeds maximum (50000)"],
            "error": None
        }

        mock_run.return_value = Mock(
            stdout=json.dumps(blender_output),
            stderr="",
            returncode=1
        )

        result = agent.verify_asset(temp_asset_file)

        # Step 3: ArtDirector handles failure
        director = ArtDirector(agent_id="director_001")

        retry_task = director.handle_verification_failure(
            failure_report=result.to_dict(),
            original_task=original_task
        )

        # Step 4: Verify retry task was created with corrections
        assert retry_task.task_id != original_task.task_id
        assert retry_task.context["retry"] is True
        assert "Reduce polygon count" in retry_task.context["asset_description"]
        assert retry_task.requirements["max_triangles"] < 50000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
