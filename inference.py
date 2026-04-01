#!/usr/bin/env python3

import os
import sys
import bpy
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

# Import all utility functions from submodules
from inference.image_processing import bg_to_white, resize_to_512
from inference.rendering import render_front_view, render_3d_model
from inference.voxelization import process_3d_asset
from inference.voxel_encoding import voxel_grid_to_ply, VoxelProcessor
from inference.model_utils import extract_and_decode_voxel, load_sparse_structure_encoder, inject_methods, feats_to_slat
from inference.sampling import sample_sparse_structure, sample
from inference.qwen_image_edit import qwen_image_edit_main, load_qwen_image

# ============================================================================
# STEP-0: Load pipeline and model
# ============================================================================
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline = load_sparse_structure_encoder(pipeline)
pipeline = inject_methods(pipeline)
print(f"\nLoading TRELLIS pipeline Done")

dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
dinov2_model.eval().cuda()
print(f"\nLoading DINOv2 model Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nano3D"
    )

    # Required arguments
    parser.add_argument(
        "--src_mesh_path",
        type     = str,
        required = True,
        default  = "assets/source.glb",
        help     = "path of source mesh"
    )

    parser.add_argument(
        "--output_dir",
        type     = str,
        required = True,
        default  = "assets/output",
        help     = "path of output dir"
    )

    parser.add_argument(
        "--editing_mode",
        type     = str,
        required = True,
        default  = "add",
        help     = "editing mode: 'add', 'remove', or 'replace'"
    )

    parser.add_argument(
        "--using_qwen_image", 
        action   = "store_true",
        help     = "Whether to use qwen_image for image editing. If False, you should provide your own edited images."
    )

    parser.add_argument(
        "--edit_instruction",
        type     = str,
        required = True,
        default  = "add a hat on the head.",
        help     = "editing instruction for image editing"
    )

    parser.add_argument(
        "--lora_path",
        type     = str,
        required = True,
        default  = "./Qwen-Image-Lightning/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors",
        help     = "path to the LoRA weights for Qwen Image Edit"
    )

    args             = parser.parse_args()
    src_mesh_path    = args.src_mesh_path
    output_dir       = args.output_dir
    editing_mode     = args.editing_mode
    using_qwen_image = args.using_qwen_image
    edit_instruction = args.edit_instruction
    lora_path        = args.lora_path

    if using_qwen_image == True:
        print("Loading Qwen-Image for image editing...")
        qwen_image_pipeline = load_qwen_image("Qwen/Qwen-Image-Edit", lora_path)

    render_path   = f"{output_dir}/render"
    # STEP-1: render 3d model to multi-view images
    results1 = process_3d_asset(
            model_path   = src_mesh_path,
            output_dir   = render_path,
            num_views    = 150,
            resolution   = 512,
            engine       = "CYCLES",
            dinov2_model = dinov2_model,
            batch_size   = 16,
            voxel_size   = 1/64
        )

    # STEP-2: extract and decode slat and voxel
    results2 = extract_and_decode_voxel(
            pipeline   = pipeline,
            render_dir = render_path,
            output_dir = f"{output_dir}/src_voxel"
        )

    # STEP-3: render front view
    render_front_view(
            file_path   = src_mesh_path, 
            output_dir  = f"{output_dir}/image", 
            output_name = "front.png"
        )

    # STEP-4: process image
    src_image_path = bg_to_white(f"{output_dir}/image/front.png")
    if using_qwen_image == True:
        print("Using Qwen-Image for image editing...")
        tar_image_path = f"{output_dir}/image/edited.png"
        qwen_image_edit_main(
            pipe                = qwen_image_pipeline,
            model_name          = "Qwen/Qwen-Image-Edit",
            image_path          = src_image_path,
            edit_instruction    = edit_instruction,
            save_path           = tar_image_path,
            base_seed           = 42,
            num_inference_steps = 8,
            true_cfg_scale      = 1.0,
        )
    else:
        tar_image_path = input("请输入编辑后图像的路径: ")

    tar_image_path = resize_to_512(
            tar_image_path, 
            f"{output_dir}/image"
        )

    # STEP-5: Nano3D Editing
    outputs = pipeline.run(
        src_image_path,
        tar_image_path,
        source_ply_path          = f"{output_dir}/render/voxels.ply",
        source_voxel_latent_path = f"{output_dir}/src_voxel/latent.pt",  
        source_slat_path         = f"{output_dir}/render/features.npz",
        editing_mode             = editing_mode,
        seed                     = 1,
        output_path              = output_dir
    )

    # GLB files can be extracted from the outputs
    with torch.enable_grad():
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify     = 0.95,          # Ratio of triangles to remove in the simplification process
            texture_size = 1024,      # Size of the texture used for the GLB
        )
    glb.export(f"{output_dir}/edit_mesh.glb")