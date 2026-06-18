"""
Blender script: Decimate mesh to reduce polygon count.

Usage:
    blender --background --python blender_decimate.py -- <input> <output> <ratio>
"""

import bpy
import os
import sys


def apply_decimate(input_path, output_path, ratio=0.5):
    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Import mesh
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=input_path)
    elif ext in (".fbx",):
        bpy.ops.import_scene.fbx(filepath=input_path)
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

    # Apply decimate modifier
    mod = obj.modifiers.new(name="Auto_Decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="Auto_Decimate")

    # Export
    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext == ".obj":
        bpy.ops.wm.obj_export(filepath=output_path)
    elif out_ext == ".fbx":
        bpy.ops.export_scene.fbx(filepath=output_path)
    else:
        bpy.ops.export_scene.gltf(filepath=output_path)

    print(f"Decimated mesh saved to: {output_path}")
    return True


if __name__ == "__main__":
    argv = sys.argv
    try:
        idx = argv.index("--") + 1
        args = argv[idx:]
        input_file = args[0]
        output_file = args[1]
        ratio = float(args[2]) if len(args) > 2 else 0.5
        apply_decimate(input_file, output_file, ratio)
    except (ValueError, IndexError):
        print("Usage: blender -b -P blender_decimate.py -- <input> <output> <ratio>")
