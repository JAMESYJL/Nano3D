"""Nano3D Inference Pipeline - Modular Components"""

from .image_processing import bg_to_white, resize_to_512
from .rendering import (
    sphere_hammersley_sequence,
    render_front_view,
    render_3d_model,
    get_image_data,
    render_3d_asset,
)
from .voxelization import voxelize_mesh, extract_features, process_3d_asset
from .voxel_encoding import (
    load_voxel_features,
    encode_voxel_grid,
    decode_latent_to_voxel,
    voxel_grid_to_ply,
    VoxelProcessor,
)
from .model_utils import (
    load_sparse_structure_encoder,
    extract_and_decode_voxel,
    feats_to_slat,
    inject_methods,
    run,
)
from .sampling import sample_sparse_structure, sample
from .qwen_image_edit import qwen_image_edit_main, load_qwen_image

__all__ = [
    # Image processing
    "bg_to_white",
    "resize_to_512",
    # Rendering
    "sphere_hammersley_sequence",
    "render_front_view",
    "render_3d_model",
    "get_image_data",
    "render_3d_asset",
    # Voxelization
    "voxelize_mesh",
    "extract_features",
    "process_3d_asset",
    # Voxel encoding
    "load_voxel_features",
    "encode_voxel_grid",
    "decode_latent_to_voxel",
    "voxel_grid_to_ply",
    "VoxelProcessor",
    # Model utilities
    "load_sparse_structure_encoder",
    "extract_and_decode_voxel",
    "feats_to_slat",
    "inject_methods",
    "run",
    # Sampling
    "sample_sparse_structure",
    "sample",
    # Qwen-Image editing
    "qwen_image_edit_main",
    "load_qwen_image",
]
