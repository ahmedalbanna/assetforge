"""
Blender script: Clean up mesh — remove floaters, close holes, smooth.

Usage:
    blender --background --python blender_cleanup.py -- <input> <output>
"""

import bpy
import os
import sys


def cleanup_mesh(input_path, output_path):
    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Import mesh
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=input_path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=input_path)
    else:
        print(f"Unsupported format: {ext}")
        return False

    # Get imported mesh
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        print("No mesh objects found")
        return False

    obj = objs[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Enter edit mode for cleanup operations
    bpy.ops.object.mode_set(mode="EDIT")

    # Select all and remove doubles (merge nearby vertices)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.001)

    # Fill holes
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.fill_holes(sides=4)

    # Return to object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Apply smooth shading
    bpy.ops.object.shade_smooth()

    # Export
    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext == ".obj":
        bpy.ops.wm.obj_export(filepath=output_path)
    else:
        bpy.ops.export_scene.gltf(filepath=output_path)

    print(f"Cleaned mesh saved to: {output_path}")
    return True


if __name__ == "__main__":
    argv = sys.argv
    try:
        idx = argv.index("--") + 1
        args = argv[idx:]
        input_file = args[0]
        output_file = args[1]
        cleanup_mesh(input_file, output_file)
    except (ValueError, IndexError):
        print("Usage: blender -b -P blender_cleanup.py -- <input> <output>")
