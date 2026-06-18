"""
Blender script: Quad remeshing for game-optimized topology.

Usage:
    blender --background --python blender_remesh.py -- <input> <output> <target_faces>
"""

import bpy
import os
import sys


def apply_remesh(input_path, output_path, target_faces=15000):
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

    # Apply Remesh modifier (quadriflow)
    mod = obj.modifiers.new(name="Auto_Remesh", type="REMESH")
    mod.mode = "QUAD"
    mod.octree_depth = 8  # Adjust for detail level
    mod.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="Auto_Remesh")

    # Export
    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext == ".obj":
        bpy.ops.wm.obj_export(filepath=output_path)
    else:
        bpy.ops.export_scene.gltf(filepath=output_path)

    print(f"Remeshed mesh saved to: {output_path}")
    return True


if __name__ == "__main__":
    argv = sys.argv
    try:
        idx = argv.index("--") + 1
        args = argv[idx:]
        input_file = args[0]
        output_file = args[1]
        apply_remesh(input_file, output_file)
    except (ValueError, IndexError):
        print("Usage: blender -b -P blender_remesh.py -- <input> <output>")
