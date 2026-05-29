"""Blender Python script: render an OBJ mesh to one or more views.

Single view (legacy interface, still used by render_grid_blender.py and
render_intermediate_steps.py):
  blender --background --python _blender_render_mesh.py -- \
    --input mesh.obj --output render.png --resolution 512 \
    [--azimuth A --elevation E]

Multi view (one process, many azimuths — avoids re-launching Blender and
re-importing the mesh once per view):
  blender --background --python _blender_render_mesh.py -- \
    --input mesh.obj --output-dir out/ --azimuths 0,90,180,270 \
    --resolution 512 [--elevation E]
  → writes out/view_<int(az)>.png for each azimuth.

The scene (mesh import, material, lighting, world, render settings, CUDA
device selection) is built ONCE; in multi-view mode only the camera moves
between renders. Lights are fixed in world space (view-independent), matching
the previous per-view behaviour.

Camera and lighting are set up for unit-sphere-normalized meshes.
Renders with Cycles GPU (CUDA), transparent background.
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
parser.add_argument("--output", help="Output PNG path (single-view mode)")
parser.add_argument("--output-dir", dest="output_dir",
                    help="Output directory (multi-view mode); writes view_<az>.png")
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--azimuth", type=float, default=135.0,
                    help="Camera azimuth in degrees (single-view mode)")
parser.add_argument("--azimuths", help="Comma-separated azimuths in degrees (multi-view mode)")
parser.add_argument("--elevation", type=float, default=25.0, help="Camera elevation (degrees)")
args = parser.parse_args(argv)

# --- Resolve render jobs: list of (azimuth_deg, output_path) ---
jobs = []
if args.azimuths and args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
    for tok in args.azimuths.split(","):
        tok = tok.strip()
        if not tok:
            continue
        az = float(tok)
        jobs.append((az, os.path.join(args.output_dir, f"view_{int(az)}.png")))
elif args.output:
    jobs.append((args.azimuth, args.output))
else:
    print("ERROR: provide either --output (single view) or --output-dir + --azimuths (multi view)")
    sys.exit(1)

if not jobs:
    print("ERROR: no azimuths to render")
    sys.exit(1)

# --- Clear default scene (ONCE) ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# --- Import OBJ (ONCE) ---
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

# --- Camera (created once; repositioned per view) ---
cam_data = bpy.data.cameras.new(name="Camera")
cam_data.lens = 50
cam_data.sensor_width = 36
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# --- Lighting (fixed in world space, view-independent) ---
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

# GPU rendering (CUDA) — device enumeration done once
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

# --- Render each view (camera moves; everything else stays) ---
dist = 3.0
el = math.radians(args.elevation)
for az_deg, out_path in jobs:
    az = math.radians(az_deg)
    cx = dist * math.cos(el) * math.sin(az)
    cy = dist * math.cos(el) * math.cos(az)
    cz = dist * math.sin(el)
    cam_obj.location = (cx, -cy, cz)

    # Point camera at origin
    direction = cam_obj.location.copy()
    direction.negate()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {out_path}")
