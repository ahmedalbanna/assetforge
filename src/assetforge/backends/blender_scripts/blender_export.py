"""
Blender script: Export mesh in specified format (auto-detected from extension).

Usage:
    blender --background --python blender_export.py -- <input> <output>
"""

import bpy
import os
import sys


def export_mesh(input_path, output_path):
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
        print(f"Unsupported import format: {ext}")
        return False

    # Export in requested format
    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext == ".obj":
        bpy.ops.wm.obj_export(filepath=output_path)
    elif out_ext == ".fbx":
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=True,
            add_leaf_bones=False,
        )
    elif out_ext in (".glb", ".gltf"):
        # GLTF export in Blender 4.0 uses minimal required params
        # export_format: 'GLB' or 'GLTF_SEPARATE'
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format="GLB" if out_ext == ".glb" else "GLTF_SEPARATE",
            export_materials="EXPORT",
            export_image_format="AUTO",
            use_selection=False,
            export_texcoords=True,
            export_normals=True,
        )
    elif out_ext == ".stl":
        bpy.ops.export_mesh.stl(filepath=output_path)
    else:
        # Default: GLB
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format="GLB",
            export_materials="EXPORT",
            export_image_format="AUTO",
            use_selection=False,
            export_texcoords=True,
            export_normals=True,
        )

    print(f"Mesh exported to: {output_path}")
    return True


if __name__ == "__main__":
    argv = sys.argv
    try:
        idx = argv.index("--") + 1
        args = argv[idx:]
        input_file = args[0]
        output_file = args[1]
        export_mesh(input_file, output_file)
    except (ValueError, IndexError):
        print("Usage: blender -b -P blender_export.py -- <input> <output>")
