"""
Blender script: Bake high-poly details to low-poly PBR textures.

Usage:
    blender --background --python blender_bake.py -- <low_poly> <high_poly> <output_dir> <resolution>
"""

import bpy
import os
import sys


def bake_textures(low_poly_path, high_poly_path, output_dir, resolution=2048):
    # Clear scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Import low-poly mesh
    ext = os.path.splitext(low_poly_path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=low_poly_path)
    else:
        bpy.ops.import_scene.gltf(filepath=low_poly_path)

    low_poly = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
    low_poly.name = "LowPoly"

    # Import high-poly mesh
    bpy.ops.wm.obj_import(filepath=high_poly_path)
    high_poly = [
        o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "LowPoly"
    ][0]
    high_poly.name = "HighPoly"

    # Set up render engine
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.cage_extrusion = 0.01

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create image textures
    image_names = {
        "Albedo": (resolution, resolution),
        "Normal": (resolution, resolution),
        "Roughness": (resolution, resolution),
        "Metallic": (resolution, resolution),
    }

    for name, (w, h) in image_names.items():
        img = bpy.data.images.new(name, width=w, height=h, alpha=False)
        img.filepath = os.path.join(output_dir, f"{name.lower()}.png")
        img.file_format = "PNG"

    # Bake each map
    bake_types = {
        "Albedo": "DIFFUSE",
        "Normal": "NORMAL",
        "Roughness": "ROUGHNESS",
        "Metallic": "METALLIC",
    }

    for img_name, bake_type in bake_types.items():
        img = bpy.data.images[img_name]
        low_poly.select_set(True)
        bpy.context.view_layer.objects.active = low_poly

        for mat in low_poly.data.materials:
            if mat:
                mat.node_tree.nodes.new("ShaderNodeTexImage")
                mat.node_tree.nodes[-1].image = img

        bpy.context.scene.render.bake.bake_type = bake_type
        bpy.ops.object.bake(type=bake_type)
        img.save()

    print(f"Textures baked to: {output_dir}")
    return True


if __name__ == "__main__":
    argv = sys.argv
    try:
        idx = argv.index("--") + 1
        args = argv[idx:]
        low_poly = args[0]
        high_poly = args[1]
        out_dir = args[2]
        res = int(args[3]) if len(args) > 3 else 2048
        bake_textures(low_poly, high_poly, out_dir, res)
    except (ValueError, IndexError):
        print(
            "Usage: blender -b -P blender_bake.py -- <low_poly> <high_poly> <out_dir> <res>"
        )
