#!/usr/bin/env python3

import os
import sys
import bpy
import math
import json
import torch
import utils3d
import argparse
import numpy as np
from typing import *
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from queue import Queue
from pathlib import Path
from mathutils import Vector
from types import MethodType
from typing import Dict, Tuple, Optional
import torch.nn.functional as F
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

import trellis.modules.sparse as sp
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from scipy.ndimage import label, generate_binary_structure
from plyfile import PlyData, PlyElement

torch.set_grad_enabled(False)


def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def radical_inverse(base, n):
        val = 0
        inv_base = 1.0 / base
        inv_base_n = inv_base
        while n > 0:
            digit = n % base
            val += digit * inv_base_n
            n //= base
            inv_base_n *= inv_base
        return val

    def halton_sequence(dim, n):
        return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

    def hammersley_sequence(dim, n, num_samples):
        return [n / num_samples] + halton_sequence(dim - 1, n)

    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

class BpyFrontRenderer:
    """渲染器类 - 专门用于正面渲染"""

    def __init__(self, resolution: int = 512, engine: str = "CYCLES", geo_mode: bool = False, split_normal: bool = False):
        self.resolution = resolution
        self.engine = engine
        self.geo_mode = geo_mode
        self.split_normal = split_normal
        self.import_functions = self._setup_import_functions()

    def _setup_import_functions(self):
        """设置文件导入函数映射"""
        import_functions = {
            "obj": bpy.ops.wm.obj_import,
            "glb": bpy.ops.import_scene.gltf,
            "gltf": bpy.ops.import_scene.gltf,
            "usd": bpy.ops.import_scene.usd,
            "fbx": bpy.ops.import_scene.fbx,
            "stl": bpy.ops.import_mesh.stl,
            "usda": bpy.ops.import_scene.usda,
            "dae": bpy.ops.wm.collada_import,
            "ply": bpy.ops.wm.ply_import,
            "abc": bpy.ops.wm.alembic_import,
            "blend": bpy.ops.wm.append,
        }
        return import_functions

    def init_render_settings(self):
        """初始化渲染设置"""
        bpy.context.scene.render.engine = self.engine
        bpy.context.scene.render.resolution_x = self.resolution
        bpy.context.scene.render.resolution_y = self.resolution
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True
        if self.engine == "CYCLES":
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.samples = 128 if not self.geo_mode else 1
            bpy.context.scene.cycles.filter_type = "BOX"
            bpy.context.scene.cycles.filter_width = 1
            bpy.context.scene.cycles.diffuse_bounces = 1
            bpy.context.scene.cycles.glossy_bounces = 1
            bpy.context.scene.cycles.transparent_max_bounces = (3 if not self.geo_mode else 0)
            bpy.context.scene.cycles.transmission_bounces = (3 if not self.geo_mode else 1)
            bpy.context.scene.cycles.use_denoising = True
            try:
                bpy.context.scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            except:
                pass

    def init_scene(self):
        """清空场景"""
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera(self):
        """初始化摄像机"""
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty
        return cam

    def init_lighting(self):
        """初始化灯光 - 简化版本，适合正面渲染"""
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        # 主前置灯光
        front_light = bpy.data.objects.new("Front_Light", bpy.data.lights.new("Front_Light", type="POINT"))
        bpy.context.collection.objects.link(front_light)
        front_light.data.energy = 1500
        front_light.location = (0, 2, 3)

        # 顶部补光
        top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 8000
        top_light.location = (0, 0, 8)
        top_light.scale = (50, 50, 50)

        # 底部补光
        bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 800
        bottom_light.location = (0, 0, -8)

        return {"front_light": front_light, "top_light": top_light, "bottom_light": bottom_light}

    def load_object(self, object_path: str):
        """加载3D模型"""
        file_extension = object_path.split(".")[-1].lower()
        if file_extension not in self.import_functions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        import_function = self.import_functions[file_extension]
        print(f"Loading object from {object_path}")
        if file_extension == "blend":
            import_function(directory=object_path, link=False)
        elif file_extension in {"glb", "gltf"}:
            import_function(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
        else:
            import_function(filepath=object_path)

    def unhide_all_objects(self):
        """显示所有隐藏的对象"""
        for obj in bpy.context.scene.objects:
            obj.hide_set(False)

    def delete_invisible_objects(self):
        """删除隐藏对象"""
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
        bpy.ops.object.delete()
        invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
        for col in invisible_collections:
            bpy.data.collections.remove(col)

    def split_mesh_normal(self):
        """分割网格法线"""
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def override_material(self):
        """覆盖材质 - 用于几何模式"""
        new_mat = bpy.data.materials.new(name="Override0123456789")
        new_mat.use_nodes = True
        new_mat.node_tree.nodes.clear()
        bsdf = new_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        bsdf.inputs[1].default_value = 1
        output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        new_mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bpy.context.scene.view_layers["View Layer"].material_override = new_mat

    def scene_bbox(self) -> Tuple[Vector, Vector]:
        """计算场景边界框"""
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
        for obj in scene_meshes:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(self) -> Tuple[float, Vector]:
        """归一化场景 - 缩放并居中"""
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]

        bbox_min, bbox_max = self.scene_bbox()
        print(f"[INFO] Bounding box: {bbox_min}, {bbox_max}")
        scale = 1 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale
        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return scale, offset

    def get_transform_matrix(self, obj: bpy.types.Object) -> list:
        """获取变换矩阵"""
        pos, rt, _ = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        matrix = []
        for ii in range(3):
            a = []
            for jj in range(3):
                a.append(rt[ii][jj])
            a.append(pos[ii])
            matrix.append(a)
        matrix.append([0, 0, 0, 1])
        return matrix

    def render_front_view(self, file_path: str, output_dir: str, output_name: str = "front.png",
                         scale: float = 1.0, offset: Vector = None) -> Dict:
        """
        正面渲染 - 从正前方渲染单张图像

        Args:
            file_path: 模型文件路径
            output_dir: 输出目录
            output_name: 输出图像名称
            scale: 缩放比例
            offset: 位置偏移

        Returns:
            渲染信息字典
        """
        os.makedirs(output_dir, exist_ok=True)
        self.init_render_settings()

        if file_path.endswith(".blend"):
            self.delete_invisible_objects()
        else:
            self.init_scene()
            self.load_object(file_path)
            if self.split_normal:
                self.split_mesh_normal()

        print("[INFO] Scene initialized.")

        # 归一化场景
        if offset is None:
            scale, offset = self.normalize_scene()
            print(f"[INFO] Scene normalized with auto scale: {scale}, offset: {offset}")
        else:
            scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
            if len(scene_root_objects) > 1:
                scene = bpy.data.objects.new("ParentEmpty", None)
                bpy.context.scene.collection.objects.link(scene)
                for obj in scene_root_objects:
                    obj.parent = scene
            else:
                scene = scene_root_objects[0]
            scene.scale = scene.scale * scale
            bpy.context.view_layer.update()
            scene.matrix_world.translation += offset
            bpy.ops.object.select_all(action="DESELECT")
            print(f"[INFO] Scene scaled with specified scale: {scale}, offset: {offset}")

        # 初始化摄像机和灯光
        cam = self.init_camera()
        self.init_lighting()
        print("[INFO] Camera and lighting initialized.")

        if self.geo_mode:
            self.override_material()

        # 正面视图设置：从正前方看（Y轴正向，Z轴向上）
        # yaw=π/2, pitch=0 意味着从正前方观看
        radius = 2.0
        fov    = 40 / 180 * np.pi
        if output_name == "front.png":
            yaw    = - np.pi / 2  # 旋转90度到正面
        elif output_name == "back.png":
            yaw    = np.pi / 2  # 旋转90度到背面
        pitch  = 0

        # 设置摄像机位置：正前方
        cam.location = (
            radius * np.cos(yaw) * np.cos(pitch),
            radius * np.sin(yaw) * np.cos(pitch),
            radius * np.sin(pitch),
        )
        cam.data.lens = 16 / np.tan(fov / 2)

        # 渲染
        output_path = os.path.join(output_dir, output_name)
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        # 保存元数据
        metadata = {
            "file_path": output_name,
            "camera_angle_x": float(fov),
            "camera_position": [float(cam.location.x), float(cam.location.y), float(cam.location.z)],
            "transform_matrix": self.get_transform_matrix(cam),
            "scale": float(scale),
            "offset": [float(offset.x), float(offset.y), float(offset.z)],
        }

        metadata_path = os.path.join(output_dir, "front_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[INFO] Front view rendered: {output_path}")

        return {
            "rendered": True,
            "output_dir": output_dir,
            "image_file": output_path,
            "metadata_file": metadata_path,
        }


def render_front_view(file_path: str, output_dir: str, output_name: str = "front.png",
                      scale: float = 1.0, offset: Vector = None, resolution: int = 512,
                      engine: str = "CYCLES", geo_mode: bool = False, split_normal: bool = False) -> Dict:
    """
    便捷函数 - 正面渲染

    Args:
        file_path: 模型文件路径
        output_dir: 输出目录
        output_name: 输出图像名称
        scale: 缩放比例
        offset: 位置偏移
        resolution: 渲染分辨率
        engine: 渲染引擎 (CYCLES or EEVEE)
        geo_mode: 几何模式（快速渲染）
        split_normal: 是否分割法线

    Returns:
        渲染信息字典
    """
    renderer = BpyFrontRenderer(resolution=resolution, engine=engine, geo_mode=geo_mode, split_normal=split_normal)
    return renderer.render_front_view(file_path, output_dir, output_name, scale, offset)

class BpyRenderer:
    def __init__(self, resolution: int = 512, engine: str = "CYCLES", geo_mode: bool = False, split_normal: bool = False):
        self.resolution = resolution
        self.engine = engine
        self.geo_mode = geo_mode
        self.split_normal = split_normal
        self.import_functions = self._setup_import_functions()

    def _setup_import_functions(self):
        import_functions = {
            "obj": bpy.ops.wm.obj_import,
            "glb": bpy.ops.import_scene.gltf,
            "gltf": bpy.ops.import_scene.gltf,
            "usd": bpy.ops.import_scene.usd,
            "fbx": bpy.ops.import_scene.fbx,
            "stl": bpy.ops.import_mesh.stl,
            "usda": bpy.ops.import_scene.usda,
            "dae": bpy.ops.wm.collada_import,
            "ply": bpy.ops.wm.ply_import,
            "abc": bpy.ops.wm.alembic_import,
            "blend": bpy.ops.wm.append,
        }
        return import_functions

    def init_render_settings(self):
        bpy.context.scene.render.engine = self.engine
        bpy.context.scene.render.resolution_x = self.resolution
        bpy.context.scene.render.resolution_y = self.resolution
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True
        if self.engine == "CYCLES":
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.samples = 128 if not self.geo_mode else 1
            bpy.context.scene.cycles.filter_type = "BOX"
            bpy.context.scene.cycles.filter_width = 1
            bpy.context.scene.cycles.diffuse_bounces = 1
            bpy.context.scene.cycles.glossy_bounces = 1
            bpy.context.scene.cycles.transparent_max_bounces = (3 if not self.geo_mode else 0)
            bpy.context.scene.cycles.transmission_bounces = (3 if not self.geo_mode else 1)
            bpy.context.scene.cycles.use_denoising = True
            try:
                bpy.context.scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            except:
                pass

    def init_scene(self):
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera(self):
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty
        return cam

    def init_lighting(self):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
        bpy.context.collection.objects.link(default_light)
        default_light.data.energy = 1000
        default_light.location = (4, 1, 6)
        default_light.rotation_euler = (0, 0, 0)

        top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 10000
        top_light.location = (0, 0, 10)
        top_light.scale = (100, 100, 100)

        bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 1000
        bottom_light.location = (0, 0, -10)
        bottom_light.rotation_euler = (0, 0, 0)
        return {"default_light": default_light, "top_light": top_light, "bottom_light": bottom_light}

    def load_object(self, object_path: str):
        file_extension = object_path.split(".")[-1].lower()
        if file_extension not in self.import_functions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        import_function = self.import_functions[file_extension]
        print(f"Loading object from {object_path}")
        if file_extension == "blend":
            import_function(directory=object_path, link=False)
        elif file_extension in {"glb", "gltf"}:
            import_function(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
        else:
            import_function(filepath=object_path)

    def delete_invisible_objects(self):
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
        bpy.ops.object.delete()
        invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
        for col in invisible_collections:
            bpy.data.collections.remove(col)

    def unhide_all_objects(self):
        for obj in bpy.context.scene.objects:
            obj.hide_set(False)

    def convert_to_meshes(self):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
        for obj in bpy.context.scene.objects:
            obj.select_set(True)
        bpy.ops.object.convert(target="MESH")

    def triangulate_meshes(self):
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def split_mesh_normal(self):
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def delete_custom_normals(self):
        for this_obj in bpy.data.objects:
            if this_obj.type == "MESH":
                bpy.context.view_layer.objects.active = this_obj
                bpy.ops.mesh.customdata_custom_splitnormals_clear()

    def override_material(self):
        new_mat = bpy.data.materials.new(name="Override0123456789")
        new_mat.use_nodes = True
        new_mat.node_tree.nodes.clear()
        bsdf = new_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        bsdf.inputs[1].default_value = 1
        output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        new_mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bpy.context.scene.view_layers["View Layer"].material_override = new_mat

    def scene_bbox(self) -> Tuple[Vector, Vector]:
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
        for obj in scene_meshes:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(self) -> Tuple[float, Vector]:
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]

        bbox_min, bbox_max = self.scene_bbox()
        print(f"[INFO] Bounding box: {bbox_min}, {bbox_max}")
        scale = 1 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale
        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return scale, offset

    def get_transform_matrix(self, obj: bpy.types.Object) -> list:
        pos, rt, _ = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        matrix = []
        for ii in range(3):
            a = []
            for jj in range(3):
                a.append(rt[ii][jj])
            a.append(pos[ii])
            matrix.append(a)
        matrix.append([0, 0, 0, 1])
        return matrix

    def render_object(self, file_path: str, output_dir: str, num_views: int = 150, scale: float = 1.0, offset: Vector = None, save_mesh: bool = True) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        self.init_render_settings()
        if file_path.endswith(".blend"):
            self.delete_invisible_objects()
        else:
            self.init_scene()
            self.load_object(file_path)
            if self.split_normal:
                self.split_mesh_normal()
            # delete_custom_normals()
        print("[INFO] Scene initialized.")

        if offset is None:
            scale, offset = self.normalize_scene()
            print(f"[INFO] Scene normalized with auto scale: {scale}, offset: {offset}")
        else:
            scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
            if len(scene_root_objects) > 1:
                scene = bpy.data.objects.new("ParentEmpty", None)
                bpy.context.scene.collection.objects.link(scene)
                for obj in scene_root_objects:
                    obj.parent = scene
            else:
                scene = scene_root_objects[0]
            scene.scale = scene.scale * scale
            bpy.context.view_layer.update()
            scene.matrix_world.translation += offset
            bpy.ops.object.select_all(action="DESELECT")
            print(f"[INFO] Scene scaled with specified scale: {scale}, offset: {offset}")

        cam = self.init_camera()
        self.init_lighting()
        print("[INFO] Camera and lighting initialized.")
        if self.geo_mode:
            self.override_material()

        yaws = []
        pitchs = []
        offset_random = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset_random)
            yaws.append(y)
            pitchs.append(p)

        # Dynamic radius and fov calculation (from TRELLIS dataset toolkit)
        # This ensures consistent viewing frustum regardless of object size
        fov_min, fov_max = 10, 70
        radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
        radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
        k_min = 1 / radius_max**2
        k_max = 1 / radius_min**2
        ks = np.random.uniform(k_min, k_max, (num_views,))
        radius = [1 / np.sqrt(k) for k in ks]
        fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]

        views = [{"yaw": y, "pitch": p, "radius": r, "fov": f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
        to_export = {
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "scale": scale,
            "offset": [offset.x, offset.y, offset.z],
            "frames": [],
        }

        for i, view in enumerate(views):
            cam.location = (
                view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
                view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
                view["radius"] * np.sin(view["pitch"]),
            )
            cam.data.lens = 16 / np.tan(view["fov"] / 2)
            bpy.context.scene.render.filepath = os.path.join(output_dir, f"{i:03d}.png")
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            metadata = {
                "file_path": f"{i:03d}.png",
                "camera_angle_x": view["fov"],
                "transform_matrix": self.get_transform_matrix(cam),
            }
            to_export["frames"].append(metadata)
        with open(os.path.join(output_dir, "transforms.json"), "w") as f:
            json.dump(to_export, f, indent=4)

        mesh_file_path = None
        # if save_mesh:
        #     try:
        #         self.unhide_all_objects()
        #         self.convert_to_meshes()
        #         self.triangulate_meshes()
        #         print("[INFO] Meshes triangulated.")
        #         ply_path = os.path.join(output_dir, "mesh.ply")
        #         try:
        #             bpy.ops.wm.ply_export(filepath=ply_path)
        #             mesh_file_path = ply_path
        #             print("[INFO] Mesh file saved.")
        #         except AttributeError:
        #             try:
        #                 bpy.ops.export_mesh.ply(filepath=ply_path)
        #                 mesh_file_path = ply_path
        #                 print("[INFO] Mesh file saved.")
        #             except AttributeError:
        #                 print("[WARNING] PLY export not available, skipping mesh export")
        #     except Exception as e:
        #         print(f"[WARNING] Mesh export failed: {e}")

        if save_mesh:
            try:
                import trimesh
                import trimesh.transformations as tf
                if file_path.lower().endswith(('.glb', '.gltf')):
                    scene = trimesh.load(file_path, force='scene')
                    mesh = scene.dump(concatenate=True)
                    mat_y_to_z = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
                    mesh.apply_transform(mat_y_to_z)
                    if self.split_normal:
                        mesh.unmerge_vertices()
                    mesh.apply_scale(scale)
                    translation = [offset.x, offset.y, offset.z]
                    mesh.apply_translation(translation)
                    ply_path = os.path.join(output_dir, "mesh.ply")
                    mesh.export(ply_path)
                    mesh_file_path = ply_path
            except Exception as e:
                print(f"[ERROR] Mesh export failed: {e}")
        return {
            "rendered": True,
            "num_views": num_views,
            "output_dir": output_dir,
            "transforms_file": os.path.join(output_dir, "transforms.json"),
            "mesh_file": mesh_file_path,
        }


def render_3d_model(file_path: str, output_dir: str, num_views: int = 150, scale: float = 1.0, offset: Vector = None, resolution: int = 512, engine: str = "CYCLES", geo_mode: bool = False, split_normal: bool = False, save_mesh: bool = True) -> Dict:
    renderer = BpyRenderer(resolution=resolution, engine=engine, geo_mode=geo_mode, split_normal=split_normal)
    return renderer.render_object(file_path, output_dir, num_views, scale, offset, save_mesh)


def get_image_data(frames, output_dir):
    """
    Load and preprocess rendered images with camera parameters

    Args:
        frames: List of frame metadata from transforms.json
        output_dir: Directory containing rendered images

    Yields:
        Dictionaries with 'image', 'extrinsics', 'intrinsics'
    """
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(output_dir, view["file_path"])
            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None

            # Resize to 518x518 (DINOv2 expects 518 patches)
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255

            # Remove alpha channel
            if image.shape[2] == 4:
                image = image[:, :, :3] * image[:, :, 3:]
            else:
                image = image[:, :, :3]

            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Get camera parameters
            c2w = torch.tensor(view["transform_matrix"])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)

            fov = view["camera_angle_x"]
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(
                torch.tensor(fov), torch.tensor(fov)
            )

            return {
                "image": image,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics
            }

        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data


def render_3d_asset(
    model_path,
    output_dir,
    num_views=150,
    resolution=512,
    engine="CYCLES",
    **kwargs
):
    """
    Render a 3D asset to multi-view images

    Args:
        model_path: Path to 3D model file
        output_dir: Directory to save rendered outputs
        num_views: Number of views to render
        resolution: Image resolution
        engine: Rendering engine (CYCLES or EEVEE)
        **kwargs: Additional rendering parameters

    Returns:
        Dictionary with rendering results
    """
    print("=" * 60)
    print("STEP 1: 3D Model Rendering")
    print("=" * 60)

    # Check if already rendered
    if os.path.exists(os.path.join(output_dir, "transforms.json")) and \
       os.path.exists(os.path.join(output_dir, "mesh.ply")):
        print(f"Render directory {output_dir} already exists, skipping rendering")
        return {
            "rendered": True,
            "num_views": num_views,
            "output_dir": output_dir,
            "transforms_file": os.path.join(output_dir, "transforms.json"),
            "mesh_file": os.path.join(output_dir, "mesh.ply"),
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up rendering parameters
    render_params = {
        "num_views": num_views,
        "scale": 1.0,
        "offset": None,
        "resolution": resolution,
        "engine": engine,
        "geo_mode": False,
        "split_normal": False,
        "save_mesh": True,
    }
    render_params.update(kwargs)

    print(f"Input model: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Rendering parameters: {render_params}")

    # Render using Blender
    result = render_3d_model(
        file_path=model_path,
        output_dir=output_dir,
        **render_params
    )

    print(f"Rendering completed successfully!")
    print(f"Generated {result['num_views']} views")
    print(f"Transforms file: {result['transforms_file']}")
    if result.get("mesh_file"):
        print(f"Mesh file: {result['mesh_file']}")

    return result


