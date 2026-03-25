"""Blender Python script: render .blend object to image.png, depth.png, segmentation.png.

Uses the official Toys4k renderer approach (rehg-lab/lowshot-shapebias):
imports objects from .blend via bpy.data.libraries.load, preserving materials.

Produces three output files:
  <output-dir>/image.png        — RGBA render with transparent background
  <output-dir>/depth.png        — Normalized depth (white=near, dark=far), RGBA
  <output-dir>/segmentation.png — Binary silhouette mask, RGBA

Usage (Blender 3.6):
  blender --background --python _blender_render_multipass.py -- \
    --input path/to/object.blend --output-dir renders/512/apple/apple_048 --resolution 512
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
parser.add_argument("--input", required=True, help="Input .blend file path")
parser.add_argument("--output-dir", required=True, help="Output directory")
parser.add_argument("--resolution", type=int, default=512)
args = parser.parse_args(argv)

os.makedirs(args.output_dir, exist_ok=True)

# --- Clean scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# --- Import objects from .blend (preserves materials/textures) ---
with bpy.data.libraries.load(args.input, link=False) as (data_from, data_to):
    data_to.objects = [name for name in data_from.objects]

for obj in data_to.objects:
    if obj is not None:
        scene.collection.objects.link(obj)

mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
if not mesh_objs:
    print(f"ERROR: No mesh objects found in {args.input}")
    sys.exit(1)

obj = mesh_objs[0]
obj.name = "object"

# --- Official Toys4k transform ---
from mathutils import Matrix

# Rotate -90° on X, center at origin, apply transforms
obj.rotation_mode = "XYZ"
obj.rotation_euler = (math.radians(-90), 0, 0)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
obj.location = (0, 0, 0)
bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

# Rescale to fit (official factor: 0.45)
dims = obj.dimensions
max_dim = max(dims)
if max_dim > 0:
    scale_factor = 0.45 / (max_dim / 2)
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(scale=True)

# Apply viewpoint rotation (official default: azimuth=315°, elevation=45°)
rot_az = Matrix.Rotation(math.radians(315), 4, "Y")
rot_el = Matrix.Rotation(math.radians(45), 4, "X")
obj.matrix_world = rot_el @ rot_az @ obj.matrix_world
bpy.ops.object.transform_apply(rotation=True)

obj.select_set(False)

# --- Camera (official parameters) ---
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50
cam_data.sensor_width = 32
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, 0, 2.2)
cam_obj.rotation_euler = (0, 0, 0)

# --- Area light (official parameters) ---
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.light_add(type="AREA", location=(0, 0, 5), rotation=(0, 0, 0))
lamp = bpy.context.active_object
lamp_data = lamp.data
lamp_data.shape = "RECTANGLE"
lamp_data.size = 10
lamp_data.size_y = 10
lamp_data.use_nodes = True
nodes = lamp_data.node_tree.nodes
for node in nodes:
    nodes.remove(node)
node_blackbody = nodes.new(type="ShaderNodeBlackbody")
node_emission = nodes.new(type="ShaderNodeEmission")
node_output = nodes.new(type="ShaderNodeOutputLight")
lamp_data.node_tree.links.new(node_blackbody.outputs[0], node_emission.inputs[0])
lamp_data.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])
node_emission.inputs[1].default_value = 30
node_blackbody.inputs[0].default_value = 6000

# --- World background ---
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (1, 1, 1, 1)
bg.inputs["Strength"].default_value = 0.25

# --- Render settings (official: Cycles, 10 samples, denoising) ---
scene.render.engine = "CYCLES"
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

scene.cycles.samples = 10
scene.cycles.max_bounces = 2
scene.cycles.min_bounces = 2
scene.cycles.glossy_bounces = 2
scene.cycles.transmission_bounces = 2
scene.cycles.transparent_max_bounces = 2
scene.cycles.transparent_min_bounces = 2
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.render.use_persistent_data = True

# Denoising (API varies by Blender version)
vl = scene.view_layers[0]
if hasattr(vl, "cycles"):
    vl.cycles.use_denoising = True
else:
    vl.use_denoising = True

# GPU rendering
prefs = bpy.context.preferences.addons.get("cycles")
if prefs:
    prefs.preferences.compute_device_type = "CUDA"
    prefs.preferences.get_devices()
    for device in prefs.preferences.devices:
        device.use = True
    scene.cycles.device = "GPU"

# Enable depth pass
vl.use_pass_z = True

# --- Compositor: output image, depth, segmentation ---
scene.use_nodes = True
tree = scene.node_tree
for n in tree.nodes:
    tree.nodes.remove(n)

rl_node = tree.nodes.new("CompositorNodeRLayers")

# Image output
img_out = tree.nodes.new("CompositorNodeOutputFile")
img_out.base_path = args.output_dir
img_out.file_slots[0].path = "image"
img_out.format.file_format = "PNG"
img_out.format.color_mode = "RGBA"
tree.links.new(rl_node.outputs["Image"], img_out.inputs[0])

# Depth output: normalize + invert (near=white), combine with alpha
normalize = tree.nodes.new("CompositorNodeNormalize")
tree.links.new(rl_node.outputs["Depth"], normalize.inputs[0])

invert = tree.nodes.new("CompositorNodeInvert")
tree.links.new(normalize.outputs[0], invert.inputs["Color"])

set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
tree.links.new(invert.outputs[0], set_alpha.inputs["Image"])
tree.links.new(rl_node.outputs["Alpha"], set_alpha.inputs["Alpha"])

depth_out = tree.nodes.new("CompositorNodeOutputFile")
depth_out.base_path = args.output_dir
depth_out.file_slots[0].path = "depth"
depth_out.format.file_format = "PNG"
depth_out.format.color_mode = "RGBA"
tree.links.new(set_alpha.outputs[0], depth_out.inputs[0])

# Segmentation output: alpha as white silhouette
seg_alpha = tree.nodes.new("CompositorNodeSetAlpha")
tree.links.new(rl_node.outputs["Alpha"], seg_alpha.inputs["Image"])
tree.links.new(rl_node.outputs["Alpha"], seg_alpha.inputs["Alpha"])

seg_out = tree.nodes.new("CompositorNodeOutputFile")
seg_out.base_path = args.output_dir
seg_out.file_slots[0].path = "segmentation"
seg_out.format.file_format = "PNG"
seg_out.format.color_mode = "RGBA"
tree.links.new(seg_alpha.outputs[0], seg_out.inputs[0])

# Composite output (required by Blender)
composite = tree.nodes.new("CompositorNodeComposite")
tree.links.new(rl_node.outputs["Image"], composite.inputs["Image"])

# --- Render ---
bpy.ops.render.render()

# Blender appends frame number: rename image0001.png -> image.png etc.
for name in ("image", "depth", "segmentation"):
    src = os.path.join(args.output_dir, f"{name}0001.png")
    dst = os.path.join(args.output_dir, f"{name}.png")
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)

print(f"OK {args.output_dir}")
