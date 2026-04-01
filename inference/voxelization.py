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

# Import rendering functions
from inference.rendering import render_3d_asset, get_image_data


def voxelize_mesh(output_dir, voxel_size=1/64):
    """
    Convert mesh to voxels
    Args:
        output_dir: Directory containing mesh.ply
        voxel_size: Size of each voxel (default 1/64 for 64^3 grid)

    Returns:
        Path to voxels.ply file
    """
    mesh_path = os.path.join(output_dir, "mesh.ply")
    voxels_path = os.path.join(output_dir, "voxels.ply")

    if not os.path.exists(mesh_path):
        raise ValueError(f"Mesh file not found: {mesh_path}")

    print(f"Voxelizing mesh from: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Clip vertices to [-0.5, 0.5] bounds
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=voxel_size, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)
    )

    # Get voxel vertices
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])

    # Normalize to [-0.5, 0.5] range
    grid_size = int(1 / voxel_size)
    vertices = (vertices + 0.5) / grid_size - 0.5

    # Save voxels
    utils3d.io.write_ply(voxels_path, vertices)
    print(f"Voxelized mesh saved to: {voxels_path}")
    print(f"Total voxels: {len(vertices)}")

    return voxels_path


def extract_features(
    output_dir,
    model,
    batch_size=16,
    voxel_size=1/64
):
    """
    Extract DINOv2 features from rendered images and project to voxels

    Args:
        output_dir: Directory containing render outputs
        model: DINOv2 model name
        batch_size: Batch size for feature extraction
        voxel_size: Size of voxels (default 1/64 for 64^3 grid)

    Returns:
        Dictionary with paths to voxels and features
    """
    print("=" * 60)
    print("STEP 2: Feature Extraction with DINOv2")
    print("=" * 60)

    dinov2_model = model

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # DINOv2 output patch size
    n_patch = 518 // 14

    # Load metadata
    transforms_path = os.path.join(output_dir, "transforms.json")
    mesh_path       = os.path.join(output_dir, "mesh.ply")
    voxels_path     = os.path.join(output_dir, "voxels.ply")

    if not os.path.exists(transforms_path):
        raise ValueError(f"Transforms file not found: {transforms_path}")
    if not os.path.exists(mesh_path):
        raise ValueError(f"Mesh file not found: {mesh_path}")

    # Voxelize if needed
    if not os.path.exists(voxels_path):
        print("Voxelizing mesh...")
        voxelize_mesh(output_dir, voxel_size=voxel_size)

    # Load voxel positions
    print("Loading voxel positions...")
    positions = utils3d.io.read_ply(voxels_path)[0]
    positions = torch.from_numpy(positions).float().cuda()

    # Get voxel grid indices
    grid_size = int(1 / voxel_size)
    indices = ((positions + 0.5) * grid_size).long()

    assert torch.all(indices >= 0) and torch.all(indices < grid_size), \
        "Some vertices are out of bounds"

    n_voxels = positions.shape[0]
    print(f"Total voxels: {n_voxels}")

    # Load image data
    print("Loading rendered images...")
    with open(transforms_path, "r") as f:
        metadata = json.load(f)
    frames = metadata["frames"]

    image_data = []
    for datum in get_image_data(frames, output_dir):
        datum["image"] = transform(datum["image"])
        image_data.append(datum)

    n_views = len(image_data)
    print(f"Total views: {n_views}")

    # Extract features
    print("Extracting DINOv2 features...")
    patchtokens_lst = []
    uv_lst = []

    for i in tqdm(range(0, n_views, batch_size), desc="Processing batches"):
        batch_data = image_data[i : i + batch_size]
        bs = len(batch_data)

        batch_images = torch.stack([d["image"] for d in batch_data]).cuda()
        batch_extrinsics = torch.stack([d["extrinsics"] for d in batch_data]).cuda()
        batch_intrinsics = torch.stack([d["intrinsics"] for d in batch_data]).cuda()

        # Extract features
        with torch.no_grad():
            features = dinov2_model(batch_images, is_training=True)

        # Get patch tokens and project to voxels
        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        patchtokens = features["x_prenorm"][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(
            bs, 1024, n_patch, n_patch
        )

        patchtokens_lst.append(patchtokens)
        uv_lst.append(uv)

    # Aggregate features
    print("Aggregating features across views...")
    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)

    # Interpolate features to voxel positions
    patchtokens_interp = F.grid_sample(
        patchtokens,
        uv.unsqueeze(1),
        mode="bilinear",
        align_corners=False
    ).squeeze(2).permute(0, 2, 1).cpu().numpy()

    # Average across views
    latent_features = np.mean(patchtokens_interp, axis=0).astype(np.float16)

    print(f"Latent features shape: {latent_features.shape}")

    # Save features
    features_path = os.path.join(output_dir, "features.npz")
    np.savez_compressed(
        features_path,
        indices=indices.cpu().numpy().astype(np.uint8),
        patchtokens=latent_features
    )

    print(f"Features saved to: {features_path}")

    return {
        "voxels_path": voxels_path,
        "features_path": features_path,
        "n_voxels": n_voxels,
        "n_views": n_views,
        "latent_dim": latent_features.shape[1]
    }


def process_3d_asset(
    model_path,
    output_dir,
    dinov2_model,
    num_views=150,
    resolution=512,
    engine="CYCLES",
    batch_size=16,
    voxel_size=1/64
):
    """
    Complete pipeline: Render 3D asset and extract voxels + latent features

    Args:
        model_path: Path to input 3D model
        output_dir: Directory for outputs
        num_views: Number of views to render
        resolution: Rendering resolution
        engine: Rendering engine
        dinov2_model: DINOv2 model to use
        batch_size: Batch size for feature extraction
        voxel_size: Voxel grid resolution

    Returns:
        Dictionary with complete results
    """
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE: Render + Extract Features")
    print("=" * 60 + "\n")

    # Step 1: Render
    render_results = render_3d_asset(
        model_path = model_path,
        output_dir = output_dir,
        num_views  = num_views,
        resolution = resolution,
        engine     = engine
    )

    # Step 2: Extract features
    feature_results = extract_features(
        output_dir = output_dir,
        model      = dinov2_model,
        batch_size = batch_size,
        voxel_size = voxel_size
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Voxels: {feature_results['voxels_path']}")
    print(f"  Latent features: {feature_results['features_path']}")
    print(f"  Total voxels: {feature_results['n_voxels']}")
    print(f"  Total views: {feature_results['n_views']}")
    print(f"  Latent dimension: {feature_results['latent_dim']}")

    return {
        "render": render_results,
        "features": feature_results
    }


