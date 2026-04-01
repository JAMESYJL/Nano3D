import argparse
import math
import os
from PIL import Image

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
)
from diffusers.models import QwenImageTransformer2DModel
import torch


def qwen_image_edit_main(
    pipe,
    model_name: str,
    image_path: str,
    edit_instruction: str,
    save_path: str,
    base_seed: int = 42,
    num_inference_steps: int = 8,
    true_cfg_scale: float = 1.0,
    device = "cuda"
):
    # Load the input image
    assert os.path.exists(image_path), f"Image path {image_path} does not exist"
    input_image = Image.open(image_path).convert("RGB")

    if "Qwen-Image-2512" in model_name:
        negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    else:
        negative_prompt = " "

    # Prepare input arguments
    input_args = {
        "image": input_image,
        "prompt": edit_instruction,
        "generator": torch.Generator(device=device).manual_seed(base_seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
    }

    # Generate edited image
    edited_image = pipe(**input_args).images[0]

    # Create directory if needed and save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    edited_image.save(save_path)
    print(f"Image saved to {save_path}")

def load_qwen_image(
            model_name = "Qwen/Qwen-Image-Edit", 
            lora_path = ""
        ):

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    # Check if model supports image editing
    if "Qwen-Image-Edit-2509" in model_name or "Qwen-Image-Edit-2511" in model_name:
        pipe_cls = QwenImageEditPlusPipeline
    else:
        pipe_cls = QwenImageEditPipeline

    if lora_path is not None:
        model = QwenImageTransformer2DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch_dtype
        )
        assert os.path.exists(lora_path), f"Lora path {lora_path} does not exist"
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        pipe = pipe_cls.from_pretrained(
            model_name, transformer=model, scheduler=scheduler, torch_dtype=torch_dtype
        )
        pipe.load_lora_weights(lora_path)
    else:
        pipe = pipe_cls.from_pretrained(model_name, torch_dtype=torch_dtype)

    pipe = pipe.to(device)

    return pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit an image using Qwen Image Edit model")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--prompt", type=str, help="Text instruction for editing the image")
    parser.add_argument("--save_path", type=str, help="Path to save the edited image")
    parser.add_argument("--lora_path", type=str, help="Path to the LoRA weights")

    args = parser.parse_args()
    model_name          = "Qwen/Qwen-Image-Edit"
    steps               = 8
    num_inference_steps = steps
    true_cfg_scale      = 1.0
    base_seed           = 42
    # lora_path           = args.lora_path
    lora_path           = "./Qwen-Image-Lightning/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"

    pipe = load_qwen_image(model_name, lora_path)

    qwen_image_edit_main(
        pipe             = pipe,
        model_name       = model_name,
        image_path       = args.image_path,
        edit_instruction = args.prompt,
        save_path        = args.save_path,
        base_seed        = base_seed,
        num_inference_steps = num_inference_steps,
        true_cfg_scale   = true_cfg_scale,
    )
