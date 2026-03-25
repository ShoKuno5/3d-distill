"""Blender Python script: render a single OBJ mesh to PNG.

Called by render_grid_blender.py via:
  blender --background --python _blender_render_mesh.py -- \
    --input mesh.obj --output render.png --resolution 512

Camera and lighting are set up for unit-sphere-normalized meshes.
Renders with Cycles CPU, transparent background composited to white.
"""

import bpy
import math
import os
import sys

# Parse arguments after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input OBJ file path")
parser.add_argument("--output", required=True, help="Output PNG file path")
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--azimuth", type=float, default=135.0, help="Camera azimuth (degrees)")
parser.add_argument("--elevation", type=float, default=25.0, help="Camera elevation (degrees)")
args = parser.parse_args(argv)

# --- Clear default scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# --- Import OBJ ---
bpy.ops.import_scene.obj(filepath=args.input, axis_forward='-Z', axis_up='Y')

# Select imported objects
imported = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not imported:
    print("ERROR: No mesh imported")
    sys.exit(1)

# Assign material with subtle shading
for obj in imported:
    mat = bpy.data.materials.new(name="MeshMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.65, 0.65, 0.75, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0

    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Smooth shading
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)

# --- Camera ---
cam_data = bpy.data.cameras.new(name="Camera")
cam_data.lens = 50
cam_data.sensor_width = 36
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Position camera for unit sphere objects
az = math.radians(args.azimuth)
el = math.radians(args.elevation)
dist = 3.0
cx = dist * math.cos(el) * math.sin(az)
cy = dist * math.cos(el) * math.cos(az)
cz = dist * math.sin(el)
cam_obj.location = (cx, -cy, cz)

# Point camera at origin
direction = cam_obj.location.copy()
direction.negate()
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

# --- Lighting ---
# Key light
key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
key_data.energy = 100
key_data.size = 3
key_obj = bpy.data.objects.new("KeyLight", key_data)
key_obj.location = (2, -2, 3)
key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
scene.collection.objects.link(key_obj)

# Fill light
fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
fill_data.energy = 40
fill_data.size = 4
fill_obj = bpy.data.objects.new("FillLight", fill_data)
fill_obj.location = (-3, -1, 2)
fill_obj.rotation_euler = (math.radians(60), 0, math.radians(-60))
scene.collection.objects.link(fill_obj)

# Rim light
rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
rim_data.energy = 60
rim_data.size = 2
rim_obj = bpy.data.objects.new("RimLight", rim_data)
rim_obj.location = (-1, 3, 2)
rim_obj.rotation_euler = (math.radians(30), 0, math.radians(180))
scene.collection.objects.link(rim_obj)

# --- World (subtle environment) ---
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs['Strength'].default_value = 0.3
bg.inputs['Color'].default_value = (1, 1, 1, 1)

# --- Render settings ---
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64

# GPU rendering (CUDA)
prefs = bpy.context.preferences.addons.get("cycles")
if prefs:
    prefs.preferences.compute_device_type = "CUDA"
    prefs.preferences.get_devices()
    for device in prefs.preferences.devices:
        device.use = (device.type != "CPU")
    scene.cycles.device = "GPU"
else:
    scene.cycles.device = 'CPU'
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# Output
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
scene.render.filepath = args.output

# --- Render ---
bpy.ops.render.render(write_still=True)
print(f"Rendered: {args.output}")
