#!/usr/bin/env python3
"""
Blender Asset Validator - Standalone Script for Blender
Validates 3D assets using Blender's bpy module

This script is designed to be executed by Blender in headless mode:
    blender --background --python blender_validator.py -- <asset_file> [options]

Outputs validation results as JSON to STDOUT.
"""

import sys
import json
import os
from typing import Dict, Any, Optional


def validate_asset_with_blender(file_path: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate a 3D asset using Blender's bpy module.

    Args:
        file_path: Path to the asset file (.glb, .fbx, .obj)
        constraints: Optional validation constraints

    Returns:
        Validation result dictionary with metrics and pass/fail status
    """
    try:
        import bpy
    except ImportError:
        return {
            "success": False,
            "error": "bpy module not available - must run inside Blender",
            "metrics": {}
        }

    # Set default constraints
    if constraints is None:
        constraints = {}

    max_triangles = constraints.get("max_triangles", 50000)
    min_dimensions = constraints.get("min_dimensions", 0.1)
    max_dimensions = constraints.get("max_dimensions", 100.0)
    require_uv_map = constraints.get("require_uv_map", True)
    max_materials = constraints.get("max_materials", 10)

    try:
        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Determine file format and import
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.glb' or file_ext == '.gltf':
            bpy.ops.import_scene.gltf(filepath=file_path)
        elif file_ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=file_path)
        elif file_ext == '.obj':
            bpy.ops.import_scene.obj(filepath=file_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file format: {file_ext}",
                "metrics": {}
            }

        # Get all mesh objects
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

        if not mesh_objects:
            return {
                "success": False,
                "error": "No mesh objects found in file",
                "metrics": {}
            }

        # Calculate metrics
        total_triangles = 0
        total_vertices = 0
        has_uv_map = False
        material_count = 0
        materials_list = []

        # Bounding box calculation
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for obj in mesh_objects:
            mesh = obj.data

            # Triangle count
            mesh.calc_loop_triangles()
            total_triangles += len(mesh.loop_triangles)

            # Vertex count
            total_vertices += len(mesh.vertices)

            # UV map check
            if mesh.uv_layers and len(mesh.uv_layers) > 0:
                has_uv_map = True

            # Materials
            for mat in mesh.materials:
                if mat and mat.name not in materials_list:
                    materials_list.append(mat.name)

            # Bounding box
            for vertex in mesh.vertices:
                world_coord = obj.matrix_world @ vertex.co
                min_x = min(min_x, world_coord.x)
                min_y = min(min_y, world_coord.y)
                min_z = min(min_z, world_coord.z)
                max_x = max(max_x, world_coord.x)
                max_y = max(max_y, world_coord.y)
                max_z = max(max_z, world_coord.z)

        material_count = len(materials_list)

        # Calculate dimensions
        width = max_x - min_x
        height = max_y - min_y
        depth = max_z - min_z
        max_dimension = max(width, height, depth)
        min_dimension = min(width, height, depth)

        # Prepare metrics
        metrics = {
            "triangle_count": total_triangles,
            "vertex_count": total_vertices,
            "has_uv_map": has_uv_map,
            "material_count": material_count,
            "materials": materials_list,
            "dimensions": {
                "width": round(width, 3),
                "height": round(height, 3),
                "depth": round(depth, 3),
                "max": round(max_dimension, 3),
                "min": round(min_dimension, 3)
            },
            "bounding_box": {
                "min": [round(min_x, 3), round(min_y, 3), round(min_z, 3)],
                "max": [round(max_x, 3), round(max_y, 3), round(max_z, 3)]
            },
            "mesh_count": len(mesh_objects)
        }

        # Validation checks
        validation_issues = []

        if total_triangles > max_triangles:
            validation_issues.append(
                f"Triangle count ({total_triangles}) exceeds maximum ({max_triangles})"
            )

        if require_uv_map and not has_uv_map:
            validation_issues.append("No UV map found")

        if max_dimension > max_dimensions:
            validation_issues.append(
                f"Max dimension ({max_dimension:.2f}) exceeds limit ({max_dimensions})"
            )

        if min_dimension < min_dimensions:
            validation_issues.append(
                f"Min dimension ({min_dimension:.2f}) below minimum ({min_dimensions})"
            )

        if material_count > max_materials:
            validation_issues.append(
                f"Material count ({material_count}) exceeds maximum ({max_materials})"
            )

        # Determine pass/fail
        validation_passed = len(validation_issues) == 0

        return {
            "success": True,
            "validation_passed": validation_passed,
            "metrics": metrics,
            "constraints": {
                "max_triangles": max_triangles,
                "max_dimensions": max_dimensions,
                "min_dimensions": min_dimensions,
                "require_uv_map": require_uv_map,
                "max_materials": max_materials
            },
            "issues": validation_issues,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Validation error: {str(e)}",
            "metrics": {}
        }


def main():
    """
    Main entry point when run as a Blender script.

    Expected arguments:
        blender --background --python blender_validator.py -- <file_path> [constraints_json]
    """
    # Parse arguments (after --)
    try:
        # Find the -- separator
        if "--" in sys.argv:
            args_start = sys.argv.index("--") + 1
            script_args = sys.argv[args_start:]
        else:
            script_args = []

        if len(script_args) < 1:
            result = {
                "success": False,
                "error": "No file path provided. Usage: blender --background --python blender_validator.py -- <file_path> [constraints_json]",
                "metrics": {}
            }
            print(json.dumps(result, indent=2))
            sys.exit(1)

        file_path = script_args[0]

        # Check if file exists
        if not os.path.exists(file_path):
            result = {
                "success": False,
                "error": f"File not found: {file_path}",
                "metrics": {}
            }
            print(json.dumps(result, indent=2))
            sys.exit(1)

        # Parse constraints if provided
        constraints = None
        if len(script_args) > 1:
            try:
                constraints = json.loads(script_args[1])
            except json.JSONDecodeError as e:
                result = {
                    "success": False,
                    "error": f"Invalid constraints JSON: {str(e)}",
                    "metrics": {}
                }
                print(json.dumps(result, indent=2))
                sys.exit(1)

        # Validate asset
        result = validate_asset_with_blender(file_path, constraints)

        # Output JSON to STDOUT
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        if result["success"] and result.get("validation_passed", False):
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        result = {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "metrics": {}
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    # This script is designed to run inside Blender
    # Check if we're in Blender context
    try:
        import bpy
        # Running inside Blender
        main()
    except ImportError:
        # Not in Blender, provide helpful message
        print(json.dumps({
            "success": False,
            "error": "This script must be run inside Blender using: blender --background --python blender_validator.py -- <file_path>",
            "metrics": {}
        }, indent=2))
        sys.exit(1)
