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

# Import sampling functions
from inference.sampling import sample_sparse_structure, sample
# Import voxel encoding functions
from inference.voxel_encoding import encode_voxel_grid, decode_latent_to_voxel, voxel_grid_to_ply

from scipy.ndimage import label, generate_binary_structure
from plyfile import PlyData, PlyElement


def load_sparse_structure_encoder(pipeline):
    """
    Dynamically load sparse_structure_encoder and slat_encoder if not present in pipeline
    Args:
        pipeline: TRELLIS pipeline
    Returns:
        Updated pipeline with both encoders
    """
    if "sparse_structure_encoder" in pipeline.models and "slat_encoder" in pipeline.models:
        print("sparse_structure_encoder and slat_encoder already in pipeline")
        return pipeline
    print("Attempting to load sparse_structure_encoder and slat_encoder...")

    from trellis.models import SparseStructureEncoder, SLatEncoder

    repo_id = "microsoft/TRELLIS-image-large"
    # ==========================================
    # 1. Load Sparse Structure Encoder
    # ==========================================
    if "sparse_structure_encoder" not in pipeline.models:
        print("  Downloading sparse_structure_encoder config and weights from Hugging Face...")
        ss_config_file = hf_hub_download(
            repo_id,
            "ckpts/ss_enc_conv3d_16l8_fp16.json"
        )
        ss_weights_file = hf_hub_download(
            repo_id,
            "ckpts/ss_enc_conv3d_16l8_fp16.safetensors"
        )
        with open(ss_config_file, 'r') as f:
            ss_config = json.load(f)
        print(f"  Creating sparse_structure_encoder with config: {ss_config.get('name', 'unknown')}")
        ss_encoder = SparseStructureEncoder(**ss_config['args'])
        
        ss_state_dict = load_file(ss_weights_file)
        ss_encoder.load_state_dict(ss_state_dict)
        ss_encoder.eval()
        ss_encoder.to(pipeline.device)
        pipeline.models["sparse_structure_encoder"] = ss_encoder
        print("  ✓ sparse_structure_encoder loaded successfully!")

    # ==========================================
    # 2. Load SLAT Encoder
    # ==========================================
    if "slat_encoder" not in pipeline.models:
        print("  Downloading slat_encoder config and weights from Hugging Face...")
        slat_config_file = hf_hub_download(
            repo_id,
            "ckpts/slat_enc_swin8_B_64l8_fp16.json"
        )
        slat_weights_file = hf_hub_download(
            repo_id,
            "ckpts/slat_enc_swin8_B_64l8_fp16.safetensors"
        )
        with open(slat_config_file, 'r') as f:
            slat_config = json.load(f)
        print(f"  Creating slat_encoder with config: {slat_config.get('name', 'unknown')}")
        slat_encoder = SLatEncoder(**slat_config['args'])
        
        slat_state_dict = load_file(slat_weights_file)
        slat_encoder.load_state_dict(slat_state_dict)
        slat_encoder.eval()
        slat_encoder.to(pipeline.device)
        pipeline.models["slat_encoder"] = slat_encoder
        print("  ✓ slat_encoder loaded successfully!")

    print("=" * 60)

    return pipeline


def extract_and_decode_voxel(
    pipeline,
    render_dir: str,
    output_dir: str,
    threshold: float = 0.5
) -> Dict:
    """
    Complete pipeline: Encode voxel grid -> Decode -> Save ply

    Args:
        render_dir: Directory containing render results (voxels.ply)
        output_dir: Directory to save outputs
        threshold: Threshold for voxel activation in decoder output

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("VOXEL FEATURE ENCODING AND DECODING")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Encode voxel grid to latent space
    latent = encode_voxel_grid(pipeline, render_dir)

    # Step 2: Decode latent back to voxel space
    decoded_voxel = decode_latent_to_voxel(pipeline, latent)

    # # Step 3: Convert decoded voxel to ply format
    # output_ply = os.path.join(output_dir, "decoded_voxel.ply")
    # voxel_grid_to_ply(decoded_voxel, output_ply, threshold=threshold)

    # Also save the latent features for potential future use
    latent_path = os.path.join(output_dir, "latent.pt")
    torch.save(latent, latent_path)
    print(f"Latent saved to: {latent_path}")

    # Save decoded voxel tensor
    # voxel_tensor_path = os.path.join(output_dir, "decoded_voxel_tensor.pt")
    # torch.save(decoded_voxel, voxel_tensor_path)
    # print(f"Decoded voxel tensor saved to: {voxel_tensor_path}")

    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return {
        "latent": latent_path,
        # "decoded_voxel_ply": output_ply,
        # "decoded_voxel_tensor": voxel_tensor_path,
    }

def get_dense_cube_fast(mask: np.ndarray):
    # 检查是否有任何 1
    if not np.any(mask):
        return np.zeros_like(mask)
    # 找到每个轴向包含 1 的范围
    # 比如在 z 轴上，只要 (y, x) 平面有 1，该 z 层就是有效的
    z_any = np.any(mask, axis=(1, 2))
    y_any = np.any(mask, axis=(0, 2))
    x_any = np.any(mask, axis=(1, 0))
    # 找到第一个和最后一个非零值的索引
    z_min, z_max = np.where(z_any)[0][[0, -1]]
    y_min, y_max = np.where(y_any)[0][[0, -1]]
    x_min, x_max = np.where(x_any)[0][[0, -1]]
    dense_mask = np.zeros_like(mask)
    dense_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = 1
    return dense_mask

def feats_to_slat(pipeline, feats_path):
    feats = np.load(feats_path)
    feats_tensor = sp.SparseTensor(
        feats  = torch.from_numpy(feats["patchtokens"]).float(),
        coords = torch.cat(
            [
                    torch.zeros(feats["patchtokens"].shape[0], 1).int(), 
                    torch.from_numpy(feats["indices"]).int()
            ], dim=1)
        ).cuda()
    feats_encoder = pipeline.models["slat_encoder"]
    slat = feats_encoder(feats_tensor, sample_posterior=False)
    return slat

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
        # 3. 统计各区域大小
        counts = np.bincount(labels.ravel())
        if len(counts) <= 1:
            return voxel_src.copy(), np.zeros_like(voxel_src)
        # 找到最大的【编辑区域】大小 (必须跳过背景 counts[0])
        max_edit_size = np.max(counts[1:])
        while max_edit_size < min_size and min_size >= 1:
            min_size = min_size / 2
        # 4. 核心优化：找出所有大于阈值的区域 ID
        valid_indices = np.where(counts[1:] > min_size)[0] + 1
        # 5. 构造合法编辑的 Mask (使用 map 直接映射，避免 Python 循环)
        # 将不在 valid_indices 里的全设为 0
        mask_map = np.zeros(num + 1, dtype=np.uint8)
        mask_map[valid_indices] = 1
        keep_mask = mask_map[labels] # 极其高效的查表法填充
        # 6. 应用修改：在原图基础上，仅在合法差异区进行翻转
        refined_voxel = voxel_src.copy()
        refined_voxel[keep_mask == 1] ^= 1
        
        return refined_voxel, keep_mask

# ============================================================================
# 从 trellis_image_to_3d.py 中提取的 run 函数
# ============================================================================
@torch.no_grad()
def run(
    self,
    source_image_path:        str,
    target_image_path:        str,
    source_ply_path:          str,
    source_voxel_latent_path: str,
    source_slat                   = None,              
    source_slat_path:         str = "",
    editing_mode:             str = "add",
    num_samples: int = 1,
    seed: int = 42,
    sparse_structure_sampler_params: dict = {},
    slat_sampler_params: dict = {},
    formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    preprocess_image: bool = True,
    output_path: str = "",
    st_step: int = 12,
    back_render_path: str = "",
) -> dict:
    assert editing_mode in ["add", "remove", "replace"], "editing_mode must be 'add' or 'remove' or 'replace'"
    src_img             = Image.open(source_image_path)
    tar_img             = Image.open(target_image_path)
    if back_render_path != "":
        back_img = Image.open(back_render_path)
        back_img = [tar_img, back_img]
        if preprocess_image == True:
            back_img = [self.preprocess_image(img) for img in back_img]
    else:
        back_img = None

    source_voxel_latent = torch.load(
            source_voxel_latent_path,
            weights_only = False
        )
    if preprocess_image == True:
        src_img = self.preprocess_image(src_img)
        tar_img = self.preprocess_image(tar_img)
    src_cond = self.get_cond([src_img])
    tar_cond = self.get_cond([tar_img])

    torch.manual_seed(seed)
    sparse_structure_sampler_params = {
        "source_voxel_latent": source_voxel_latent,
        "src_cond": src_cond,
        "tar_cond": tar_cond,
        "output_path": output_path,
        "st_step" : st_step,
        **sparse_structure_sampler_params
    }
    tar_coords = self.sample_sparse_structure(
            src_cond,
            num_samples,
            sparse_structure_sampler_params
        )

    src_ply_path    = source_ply_path
    tar_ply_path    = os.path.join(output_path, "edit_voxel.ply")

    proc            = VoxelProcessor(grid_size=64)
    # 1. Load
    v_a = proc.ply_to_voxel(src_ply_path)
    v_b = proc.ply_to_voxel(tar_ply_path)

    # 2. Process (去噪并合并编辑)
    v_refined, edit_mask = proc.filter_edit_regions(v_a, v_b, min_size=250)
    edit_cube_mask = get_dense_cube_fast(edit_mask)

    # 3. Save
    proc.voxel_to_ply(
            v_refined,
            os.path.join(output_path, "edit_voxel_post.ply")
        )
    proc.voxel_to_ply(
            edit_mask,
            os.path.join(output_path, "mask.ply")
        )
    proc.voxel_to_ply(
            edit_cube_mask,
            os.path.join(output_path, "cube_mask.ply")
        )

    if source_slat_path == "" and source_slat is None:
        raise ValueError("Either source_slat_path or source_slat must be provided")
    elif source_slat is not None:
        src_slat   = source_slat
    else:
        src_slat = feats_to_slat(
            self,
            source_slat_path
        )
    v_refined  = torch.nonzero(torch.from_numpy(v_refined) > 0).to(torch.int32)
    batch_idx  = torch.zeros((v_refined.shape[0], 1), dtype=torch.int32, device=v_refined.device)
    v_refined  = torch.cat([batch_idx, v_refined ], dim=1)
    tar_coords = v_refined.to('cuda')

    if back_img is None:
        tar_slat = self.sample_slat(
                tar_cond,
                tar_coords, 
                slat_sampler_params
            )
    else:
        # 'stochastic', 'multidiffusion'

        tar_cond = self.get_cond(back_img)
        tar_cond['neg_cond'] = tar_cond['neg_cond'][:1]

        with self.inject_sampler_multi_image(
                'slat_sampler', 
                len(back_img), 
                12, 
                mode='stochastic'
            ):
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            }
            tar_slat = self.sample_slat(
                tar_cond, 
                tar_coords, 
                slat_sampler_params
            )

    if editing_mode == "add" or editing_mode == "remove":
        # 1. 搬到 CPU 并转为元组提高查询效率
        src_np = src_slat.coords.cpu().numpy()
        tar_np = tar_slat.coords.cpu().numpy()
        src_c = [tuple(c) for c in src_np]
        tar_c = [tuple(c) for c in tar_np]

        # 2. 建立索引映射
        pos_to_idx2 = {pos: i for i, pos in enumerate(tar_c)}
        colors = np.zeros((len(tar_np), 3))
        colors[:] = [1.0, 0.0, 0.0]

        # 3. 直接赋值
        merge_num = 0
        all_num   = 0
        for i1, pos in enumerate(src_c):
            if pos in pos_to_idx2:
                idx2 = pos_to_idx2[pos]
                tar_slat.feats[idx2] = src_slat.feats[i1]
                colors[idx2] = [0.0, 1.0, 0.0]
                merge_num += 1
            all_num += 1
        print("="*40)
        print(f"SLAT Merge: Merged {merge_num} / {all_num} features from source to target")
        print("="*40)

        tar_xyz    = tar_np[:, -3:]
        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tar_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(
                os.path.join(output_path, "diff_show.ply"),
                pcd
            )
        print("可视化点云已保存至 merge_visualization.ply (绿:已合并, 红:未变)")
    elif editing_mode == "replace":
        # 1. 搬到 CPU 并转为集合/字典提高查询效率
        src_np = src_slat.coords.cpu().numpy()
        tar_np = tar_slat.coords.cpu().numpy()
        # 假设 mask_coords 也是一个 [N, 3] 的 Tensor 或 Array

        edit_cube_mask  = torch.nonzero(torch.from_numpy(edit_cube_mask) > 0).to(torch.int32)
        batch_idx       = torch.zeros(
            (edit_cube_mask.shape[0], 1), 
            dtype=torch.int32, 
            device=edit_cube_mask.device
        )
        mask_coords     = torch.cat([batch_idx, edit_cube_mask ], dim=1)

        mask_np = mask_coords.cpu().numpy() if hasattr(mask_coords, 'cpu') else mask_coords

        # 转为 tuple 以便进行哈希查询
        src_lookup = {tuple(pos): i for i, pos in enumerate(src_np)}
        mask_set = set(tuple(pos) for pos in mask_np)

        # 2. 准备颜色数组 (用于可视化)
        # 红色 [1, 0, 0] 表示保留了 tar 特征
        # 绿色 [0, 1, 0] 表示替换成了 src 特征
        colors = np.zeros((len(tar_np), 3))
        colors[:] = [1.0, 0.0, 0.0] 

        # 3. 执行特征替换逻辑
        merge_num = 0
        all_num = len(tar_np)

        for i_tar, pos in enumerate(tar_np):
            pos_tuple = tuple(pos)
            
            # 逻辑判断：
            # 如果在 mask_coords 中 -> 采用 tar (即不做操作)
            # 如果不在 mask_coords 中 -> 尝试采用 src
            if pos_tuple in mask_set:
                # 落在 Mask 内，强制保留原始 tar 特征
                colors[i_tar] = [1.0, 0.0, 0.0] # 红色
            else:
                # 落在 Mask 外，尝试从 src 获取特征
                if pos_tuple in src_lookup:
                    i_src = src_lookup[pos_tuple]
                    # 替换特征
                    tar_slat.feats[i_tar] = src_slat.feats[i_src]
                    colors[i_tar] = [0.0, 1.0, 0.0] # 绿色
                    merge_num += 1
                else:
                    # 如果不在 mask 里，但 src 里也没有这个点，通常只能保留 tar
                    colors[i_tar] = [0.4, 0.4, 0.4] # 灰色表示既不在mask也不在src

        print("="*40)
        print(f"SLAT Mask Merge: Replaced {merge_num} / {all_num} features from source")
        print(f"Mask Protected: {len(mask_set)} points potentially protected")
        print("="*40)

        # 4. 可视化
        tar_xyz = tar_np[:, -3:] # 假设最后三列是 XYZ，如果只有三列则直接用 tar_np
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tar_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        output_file = os.path.join(output_path, "mask_merge_diff.ply")
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"可视化已保存至 {output_file} (红:Mask保护区域, 绿:已替换为Src, 灰:无匹配)")
    return self.decode_slat(tar_slat, formats)

@torch.no_grad()
def run_custom(
    self,
    image_path:  str,
    num_samples: int = 1,
    seed: int = 42,
    sparse_structure_sampler_params: dict = {},
    slat_sampler_params: dict = {},
    formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    preprocess_image: bool = True,
    output_path: str = ""
) -> dict:
    image             = Image.open(image_path)

    if preprocess_image == True:
        image = self.preprocess_image(image)
    cond = self.get_cond([image])

    torch.manual_seed(seed)

    sparse_structure_sampler_params = {
        "output_path": output_path,
        **sparse_structure_sampler_params
    }

    coords = self.sample_sparse_structure_custom(
            cond,
            num_samples,
            sparse_structure_sampler_params
        )

    slat = self.sample_slat(
            cond,
            coords, 
            slat_sampler_params
        )

    # coords = slat.coords.cpu()
    # feats  = slat.feats.cpu()
    # result = {"coords": coords, "feats": feats}
    result = {
        "src_mesh": self.decode_slat(slat, formats),
        "src_slat": slat,
    }
    return result

def sample_sparse_structure_custom(
    self,
    cond: dict,
    num_samples: int = 1,
    sampler_params: dict = {},
) -> torch.Tensor:

    flow_model = self.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
    sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
    z_s = self.sparse_structure_sampler.sample(
        flow_model,
        noise,
        **cond,
        **sampler_params,
        verbose=True
    ).samples # torch.Size([1, 8, 16, 16, 16])
    
    torch.save(z_s, os.path.join(sampler_params["output_path"], "latent.pt"))
    decoder       = self.models['sparse_structure_decoder']
    decoded_voxel = decoder(z_s)
    output_ply    = os.path.join(sampler_params["output_path"], "voxels.ply")
    voxel_grid_to_ply(
            decoded_voxel, 
            output_ply, 
            threshold=0.5
        )

    coords = torch.argwhere(decoded_voxel>0)[:, [0, 2, 3, 4]].int()
    return coords

def inject_methods(pipeline):
    # 将 run 绑定到 pipeline
    pipeline.run                     = MethodType(run, pipeline)
    pipeline.sample_sparse_structure = MethodType(sample_sparse_structure, pipeline)

    pipeline.run_custom                     = MethodType(run_custom, pipeline)
    pipeline.sample_sparse_structure_custom = MethodType(sample_sparse_structure_custom, pipeline)

    # 从 pipeline 中提取两个 sampler，给它们都注入 sample
    sparse_structure_sampler = pipeline.sparse_structure_sampler
    slat_sampler = pipeline.slat_sampler

    # 将 sample 绑定到 sparse_structure_sampler
    sparse_structure_sampler.sample = MethodType(sample, sparse_structure_sampler)

    # 将 sample 绑定到 slat_sampler
    # slat_sampler.sample = MethodType(sample, slat_sampler)

    print("✓ Custom methods injected successfully")
    return pipeline


