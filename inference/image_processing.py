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


def bg_to_white(input_path):
    """背景转白色，保存到原路径"""
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_white{ext}"

    img = Image.open(input_path).convert('RGBA')
    white_bg = Image.new('RGB', img.size, color='white')
    white_bg.paste(img, mask=img.split()[3])
    white_bg.save(output_path, 'PNG')
    print(f"✅ 背景转白色: {output_path}")
    return output_path


def resize_to_512(input_path, output_path):
    """缩放到512x512，保存到原路径"""
    base, ext = os.path.splitext(input_path)
    output_path = f"{output_path}/edit_512{ext}"

    img = Image.open(input_path)
    result = img.resize((512, 512), Image.Resampling.LANCZOS)
    result.save(output_path, 'PNG')
    print(f"✅ 缩放到512x512: {output_path}")
    return output_path


