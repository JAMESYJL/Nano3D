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


def load_voxel_features(render_dir: str) -> Dict:
    """
    Load voxel features from npz file

    Args:
        render_dir: Directory containing features.npz

    Returns:
        Dictionary with 'patchtokens' and 'indices'
    """
    features_path = os.path.join(render_dir, "features.npz")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"Loading features from: {features_path}")
    features = np.load(features_path)

    print(f"  - Patchtokens shape: {features['patchtokens'].shape}")
    print(f"  - Indices shape: {features['indices'].shape}")

    return features


def encode_voxel_grid(
    pipeline,
    render_dir: str,
    encoder_name: str = "sparse_structure_encoder"
) -> torch.Tensor:
    """
    Encode voxel grid to latent space using VAE encoder

    Args:
        pipeline: TRELLIS pipeline with models
        render_dir: Directory containing voxels.ply
        encoder_name: Name of encoder in pipeline.models

    Returns:
        Latent tensor
    """
    print(f"\nEncoding voxel grid with {encoder_name}...")

    # Load voxel coordinates and convert to grid
    voxels_path = os.path.join(render_dir, "voxels.ply")
    position = utils3d.io.read_ply(voxels_path)[0]
    coords = ((torch.tensor(position) + 0.5) * 64).int().contiguous().cuda()

    # Convert coords to voxel grid (1, 1, 64, 64, 64)
    voxel_grid = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float).cuda()
    voxel_grid[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

    print(f"Voxel grid shape: {voxel_grid.shape}")

    # Get encoder from pipeline
    if encoder_name not in pipeline.models:
        print(f"⚠ Warning: {encoder_name} not found in pipeline.models")
        print(f"Available models: {list(pipeline.models.keys())}")
        print(f"Using fallback: Using voxel grid directly as latent...")
        return voxel_grid

    encoder = pipeline.models[encoder_name]

    # Encode to latent space
    with torch.no_grad():
        latent = encoder(voxel_grid, sample_posterior=False)

    print(f"Latent shape: {latent.shape}")
    print(f"Latent dtype: {latent.dtype}")

    return latent


def decode_latent_to_voxel(
    pipeline,
    latent: torch.Tensor,
    decoder_name: str = "sparse_structure_decoder"
) -> torch.Tensor:
    """
    Decode latent representation back to voxel space

    Args:
        pipeline: TRELLIS pipeline with models
        latent: Latent tensor from encoder
        decoder_name: Name of decoder in pipeline.models

    Returns:
        Decoded voxel tensor of shape (1, 1, 64, 64, 64)
    """
    print(f"\nDecoding latent to voxel with {decoder_name}...")

    # Get decoder from pipeline
    if decoder_name not in pipeline.models:
        raise KeyError(f"Decoder {decoder_name} not found in pipeline.models")

    decoder = pipeline.models[decoder_name]

    # Decode latent to voxel space
    with torch.no_grad():
        decoded_voxel = decoder(latent)

    print(f"Decoded voxel shape: {decoded_voxel.shape}")
    print(f"Decoded voxel dtype: {decoded_voxel.dtype}")
    print(f"Decoded voxel value range: [{decoded_voxel.min():.4f}, {decoded_voxel.max():.4f}]")

    return decoded_voxel


def voxel_grid_to_ply(
    voxel_grid: torch.Tensor,
    output_path: str,
    threshold: float = 0.5
) -> str:
    """
    Convert voxel grid to ply format with coordinates

    Args:
        voxel_grid: Tensor of shape (1, 1, 64, 64, 64) or (1, 64, 64, 64)
        output_path: Path to save ply file
        threshold: Threshold for voxel activation

    Returns:
        Path to saved ply file
    """
    # Ensure correct shape
    if voxel_grid.dim() == 5:
        voxel_grid = voxel_grid.squeeze(0).squeeze(0)  # Remove batch and channel dims
    elif voxel_grid.dim() == 4:
        voxel_grid = voxel_grid.squeeze(0)  # Remove batch dim

    voxel_grid = voxel_grid.float().cpu()

    # Get active voxel positions
    active_mask = voxel_grid > threshold
    active_indices = torch.where(active_mask)

    if len(active_indices[0]) == 0:
        print("Warning: No active voxels found!")
        return output_path

    # Convert grid indices to coordinates [-0.5, 0.5]
    x = active_indices[0].float()
    y = active_indices[1].float()
    z = active_indices[2].float()

    coordinates = torch.stack([x, y, z], dim=-1) / 64.0 - 0.5
    coordinates = coordinates.numpy()

    print(f"Active voxels: {len(coordinates)}")

    # Save to ply
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    utils3d.io.write_ply(output_path, coordinates)

    print(f"Voxel ply saved to: {output_path}")

    return output_path


class VoxelProcessor:
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        # 预计算 3D 连通域结构
        self.struct = generate_binary_structure(3, 1) # 6-connectivity
    def ply_to_voxel(self, ply_path: str) -> np.ndarray:
        """读取点云并体素化: [-0.5, 0.5] -> [0, grid_size-1]"""
        try:
            data = PlyData.read(ply_path)['vertex'].data
            # 兼容大小写坐标轴
            pts = np.stack([data[k] for k in ('x', 'y', 'z')], axis=-1).astype(np.float32)
        except (KeyError, ValueError):
            # 尝试大写
            pts = np.stack([data[k] for k in ('X', 'Y', 'Z')], axis=-1).astype(np.float32)
        # 核心变换逻辑：归一化坐标反推索引
        indices = np.floor((pts + 0.5) * self.grid_size).astype(np.int32)
        
        # 过滤越界点并去重
        mask = np.all((indices >= 0) & (indices < self.grid_size), axis=1)
        indices = indices[mask]
        
        voxel = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)
        voxel[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        return voxel
    def voxel_to_ply(self, voxel: np.ndarray, save_path: str):
        """体素转点云并保存: [0, grid_size-1] -> [-0.5, 0.5]"""
        indices = np.argwhere(voxel > 0)
        if len(indices) == 0:
            # 保存空 PLY
            v = np.empty(0, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        else:
            # 还原坐标中心点
            pos = (indices.astype(np.float32) + 0.5) / self.grid_size - 0.5
            v = np.array([tuple(p) for p in pos], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(v, 'vertex')
        PlyData([el], text=False).write(save_path) # text=False 节省空间且更快
    def filter_edit_regions(self, voxel_src: np.ndarray, voxel_tar: np.ndarray, min_size: int = 100):
        """
        核心优化逻辑：
        通过 XOR 找出差异，利用区域大小过滤噪点，最后合并合法的编辑。
        """
        # 1. 计算差异区域 (XOR)
        diff = (voxel_src ^ voxel_tar).astype(np.uint8)
        
        # 2. 连通域分析
        labels, num = label(diff, structure=self.struct)
        if num == 0:
            return voxel_src.copy(), np.zeros_like(voxel_src)
        # 3. 统计各区域大小 (一次性统计，比 np.isin 快得多)
        counts = np.bincount(labels.ravel())
        
        # 4. 核心优化：找出所有大于阈值的区域 ID (跳过背景 0)
        # 这里原来的逻辑是取 top_n 且 > 100，这里直接矢量化处理
        valid_indices = np.where(counts[1:] > min_size)[0] + 1
        
        if len(valid_indices) == 0:
            return voxel_src.copy(), np.zeros_like(voxel_src)
        # 5. 构造合法编辑的 Mask (使用 map 直接映射，避免 Python 循环)
        # 将不在 valid_indices 里的全设为 0
        mask_map = np.zeros(num + 1, dtype=np.uint8)
        mask_map[valid_indices] = 1
        keep_mask = mask_map[labels] # 极其高效的查表法填充
        # 6. 应用修改：在原图基础上，仅在合法差异区进行翻转
        refined_voxel = voxel_src.copy()
        refined_voxel[keep_mask == 1] ^= 1
        
        return refined_voxel, keep_mask
