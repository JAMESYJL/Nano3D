#!/usr/bin/env python3

import os
import sys
import time
import torch
import tempfile
import shutil
import gradio as gr
from pathlib import Path

import trimesh
import numpy as np
import trimesh.transformations as tf 

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

from inference.image_processing import bg_to_white, resize_to_512
from inference.rendering import render_front_view
from inference.voxelization import process_3d_asset
from inference.model_utils import extract_and_decode_voxel, load_sparse_structure_encoder, inject_methods

torch.set_grad_enabled(False)

def create_gradio1(pipeline, dinov2_model, qwen_image_pipeline):

    class Nano3DProcessor:
        def __init__(self, pipeline, dinov2_model, qwen_image_pipeline):
            self.pipeline = pipeline
            self.dinov2_model = dinov2_model
            self.qwen_image_pipeline = qwen_image_pipeline
            self.template = """Carefully edit the image to only perform the following change: [Edit content: "{}"], while strictly keeping the rest of the original image unchanged.
                        Do not alter the shape, proportions, colors, textures, pose/structure, composition, or lighting of the original subjects.
                        Do not change any elements that are not directly related to [Edit content].
                        The overall style, sharpness, and level of detail must remain perfectly consistent with the original image.
                        keep white background."""

        def process(self,
                    src_mesh_file,
                    editing_mode: str = "add",
                    edit_instruction: str = "add a hat on the head.",
                    num_views: int = 150,
                    resolution: int = 512,
                    st_step: int = 12,
                    multi_cond_stage2: bool = False):
            """
            Process 3D mesh editing pipeline with Qwen-Image

            Yields:
                Progress messages
            """

            # Track completed steps with timing
            completed_steps = []
            step_start_time = None
            current_step_name = None

            def format_progress():
                """Format all completed steps with timing"""
                lines = completed_steps.copy()
                return "\n".join(lines)

            edit_instruction = self.template.format(edit_instruction)
            output_dir = tempfile.mkdtemp(prefix="nano3d")
            yield f"📁 Working directory: {output_dir}"

            src_mesh_path = src_mesh_file.name if hasattr(src_mesh_file, 'name') else str(src_mesh_file)
            yield f"📥 Input mesh: {os.path.basename(src_mesh_path)}"

            render_path = f"{output_dir}/render"

            # STEP-1: Render 3D model to multi-view images
            step_start_time = time.time()
            current_step_name = "Step 1"
            yield "🎬 Step 1/5: Rendering 3D model to multi-view images..."
            results1 = process_3d_asset(
                model_path=src_mesh_path,
                output_dir=render_path,
                num_views=num_views,
                resolution=resolution,
                engine="CYCLES",
                dinov2_model=self.dinov2_model,
                batch_size=16,
                voxel_size=1/64
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 1 [Rendering 3D model to multi-view images] completed [{elapsed}s]")
            yield format_progress()

            # STEP-2: Extract and decode voxel
            step_start_time = time.time()
            current_step_name = "Step 2"
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🔍 Step 2/5: Extracting and decoding voxel..."
            results2 = extract_and_decode_voxel(
                pipeline=self.pipeline,
                render_dir=render_path,
                output_dir=f"{output_dir}/src_voxel"
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 2 [Extracting and decoding voxel] completed [{elapsed}s]")
            yield format_progress()

            # STEP-3: Render front view
            step_start_time = time.time()
            current_step_name = "Step 3"
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🖼️ Step 3/5: Rendering front view..."
            # ===============================================================
            render_front_view(
                file_path=src_mesh_path,
                output_dir=f"{output_dir}/image",
                output_name="front.png"
            )
            src_image_path   = bg_to_white(f"{output_dir}/image/front.png")
            # ===============================================================
            if multi_cond_stage2 == True:
                render_front_view(
                    file_path=src_mesh_path,
                    output_dir=f"{output_dir}/image",
                    output_name="back.png"
                )
                back_render_path = bg_to_white(f"{output_dir}/image/back.png")
            else:
                back_render_path = ""
            # ===============================================================
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 3 [Rendering front view] completed [{elapsed}s]")
            yield format_progress()

            # STEP-4: Qwen-Image editing
            step_start_time = time.time()
            current_step_name = "Step 4"
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🎨 Step 4/5: Qwen-Image editing..."
            tar_image_path = f"{output_dir}/image/edited.png"
            qwen_image_edit_main(
                pipe=self.qwen_image_pipeline,
                model_name="Qwen/Qwen-Image-Edit-2509",
                image_path=src_image_path,
                edit_instruction=edit_instruction,
                save_path=tar_image_path,
                base_seed=42,
                num_inference_steps=8,
                true_cfg_scale=1.0,
            )

            tar_image_path = resize_to_512(
                    tar_image_path, 
                    f"{output_dir}/image"
                )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 4 [Qwen-Image editing] completed [{elapsed}s]")
            yield format_progress()

            # STEP-5: Nano3D Editing
            step_start_time = time.time()
            current_step_name = "Step 5"
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🚀 Step 5/5: Running Nano3D pipeline..."
            outputs = self.pipeline.run(
                src_image_path,
                tar_image_path,
                source_ply_path  = f"{output_dir}/render/voxels.ply",
                source_voxel_latent_path = f"{output_dir}/src_voxel/latent.pt",
                source_slat_path = f"{output_dir}/render/features.npz",
                editing_mode     = editing_mode,
                seed             = 1,
                output_path      = output_dir,
                st_step          = st_step,
                back_render_path = back_render_path
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 5 [Running Nano3D pipeline] completed [{elapsed}s]")
            yield format_progress()

            # Export to GLB
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "💾 Exporting to GLB..."

            with torch.enable_grad():
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )

            output_glb_path = f"{output_dir}/edit_mesh.glb"
            # mesh = update_mesh_preview(glb)
            glb.export(output_glb_path)
            elapsed = int(time.time() - step_start_time)
            completed_steps.append("✅ Export completed [%ds]" % elapsed)
            yield format_progress()

            progress = format_progress() + "\n" if completed_steps else ""
            # Return results
            yield {
                "status": progress + "✨ Nano3D editing completed!",
                "output_mesh": output_glb_path,
                "src_image": src_image_path,
                "edited_image": tar_image_path
            }

    def process_wrapper(src_mesh, editing_mode, edit_instruction, num_views, resolution, st_step):
        final_status = ""
        final_mesh = None
        final_src_image = None
        final_edited_image = None

        for result in processor.process(
            src_mesh,
            editing_mode=editing_mode,
            edit_instruction=edit_instruction,
            num_views=num_views,
            resolution=resolution,
            st_step=st_step
        ):
            if isinstance(result, dict):
                # Save final result
                final_status = result.get("status", "")
                final_mesh = result.get("output_mesh")
                final_src_image = result.get("src_image")
                final_edited_image = result.get("edited_image")
            else:
                # Progress message - keep previous results visible
                yield result, final_mesh, final_src_image, final_edited_image

        # Yield final result
        yield final_status, final_mesh, final_src_image, final_edited_image

    """Create Gradio interface for Nano3D"""
    processor = Nano3DProcessor(pipeline, dinov2_model, qwen_image_pipeline)

    with gr.Blocks(title="Nano3D Mesh Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Nano3D Mesh Editor")
        gr.Markdown("Upload a 3D mesh and edit it with Qwen-Image + Nano3D")

        with gr.Row():
            # Left panel - Input
            with gr.Column(scale=1):
                gr.Markdown("## 📥 Input")

                src_mesh = gr.File(
                    label="🔼 Upload Source Mesh",
                    file_types=[".glb", ".obj", ".fbx", ".ply"]
                )

                def update_mesh_preview(file):
                    if file is None:
                        return None
                    return file.name if hasattr(file, 'name') else str(file)

                src_mesh_preview = gr.Model3D(
                    label="📦 Source Mesh Preview",
                    interactive=False
                )
                src_mesh.change(
                    fn=update_mesh_preview,
                    inputs=[src_mesh],
                    outputs=[src_mesh_preview]
                )

                gr.Markdown("### ⚙️ Parameters")

                editing_mode = gr.Radio(
                    choices=["add", "remove", "replace"],
                    value="add",
                    label="Editing Mode"
                )

                edit_instruction = gr.Textbox(
                    label="Edit Instruction",
                    value="add a hat on the head.",
                    lines=3
                )

                st_step = gr.Slider(
                    minimum=5,
                    maximum=14,
                    value=11,
                    step=1,
                    label="Editing Aggressiveness",
                    info="Higher values mean closer to source mesh; Lower values mean more aggressive edits"
                )

                with gr.Accordion("Render Parameter", open=False):
                    num_views = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=30,
                        step=10,
                        label="Number of Views"
                    )

                    resolution = gr.Radio(
                        choices=[256, 512],
                        value=512,
                        label="Resolution"
                    )

                process_btn = gr.Button(
                    "🚀 Start Nano3D Editing",
                    variant="primary",
                    scale=3,
                )

            # Right panel - Output
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Output")

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                    max_lines=10
                )

                gr.Markdown("### Images")

                with gr.Row():
                    src_image_output = gr.Image(label="Source Image", interactive=False)
                    edited_image_output = gr.Image(label="Edited Image", interactive=False)

                output_mesh = gr.Model3D(
                    label="Output Mesh",
                    interactive=False
                )

        # Connect processing
        process_btn.click(
            fn=process_wrapper,
            inputs=[src_mesh, editing_mode, edit_instruction, num_views, resolution,st_step],
            outputs=[status_output, output_mesh, src_image_output, edited_image_output]
        )

    return demo

def create_gradio2(pipeline, dinov2_model, qwen_image_pipeline):

    class Nano3DGeneratorProcessor:
        def __init__(self, pipeline, dinov2_model, qwen_image_pipeline):
            self.pipeline = pipeline
            self.dinov2_model = dinov2_model
            self.qwen_image_pipeline = qwen_image_pipeline
            self.template = """Carefully edit the image to only perform the following change: [Edit content: "{}"], while strictly keeping the rest of the original image unchanged.
                        Do not alter the shape, proportions, colors, textures, pose/structure, composition, or lighting of the original subjects.
                        Do not change any elements that are not directly related to [Edit content].
                        The overall style, sharpness, and level of detail must remain perfectly consistent with the original image.
                        keep white background."""

        def process(self,
                    src_input_image,
                    editing_mode: str = "add",
                    edit_instruction: str = "add a hat on the head.",
                    st_step: int = 12):
            """
            Process 3D generation and editing pipeline with Qwen-Image

            Yields:
                Progress messages or final result dict
            """

            # Track completed steps with timing
            completed_steps = []
            step_start_time = None
            current_step_name = None

            def format_progress():
                """Format all completed steps with timing"""
                lines = completed_steps.copy()
                return "\n".join(lines)

            edit_instruction = self.template.format(edit_instruction)
            output_dir = tempfile.mkdtemp(prefix="nano3d_gen_")
            yield f"📁 Working directory: {output_dir}"

            src_image_path = src_input_image.name if hasattr(src_input_image, 'name') else str(src_input_image)
            yield f"📥 Input image: {os.path.basename(src_image_path)}"

            # Create necessary directories
            os.makedirs(f"{output_dir}/image", exist_ok=True)

            # STEP-0: Source mesh generation and feature extraction
            step_start_time = time.time()
            current_step_name = "Step 0"
            yield "🔧 Step 0/4: Generating source mesh from image..."
            result = self.pipeline.run_custom(
                src_image_path,
                seed=1,
                output_path=output_dir,
            )
            # Export source mesh to GLB
            with torch.enable_grad():
                src_glb = postprocessing_utils.to_glb(
                    result["src_mesh"]['gaussian'][0],
                    result["src_mesh"]['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )
            src_mesh_path = f"{output_dir}/src_mesh.glb"
            src_glb.export(src_mesh_path)
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 0 [Generating source mesh] completed [{elapsed}s]")
            yield format_progress()

            # STEP-1: Render front view
            step_start_time = time.time()
            current_step_name = "Step 1"
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🖼️ Step 1/4: Rendering front view..."
            render_front_view(
                file_path=src_mesh_path,
                output_dir=f"{output_dir}/image",
                output_name="front.png"
            )
            src_image_rendered = bg_to_white(f"{output_dir}/image/front.png")
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 1 [Rendering front view] completed [{elapsed}s]")
            yield format_progress()

            # STEP-2: Image processing and editing
            step_start_time = time.time()
            current_step_name = "Step 2"
            progress = format_progress() + "\n" if completed_steps else ""

            yield progress + "🎨 Step 2/4: Qwen-Image editing..."
            tar_image_path = f"{output_dir}/image/edited.png"
            qwen_image_edit_main(
                pipe=self.qwen_image_pipeline,
                model_name="Qwen/Qwen-Image-Edit-2509",
                image_path=src_image_rendered,
                edit_instruction=edit_instruction,
                save_path=tar_image_path,
                base_seed=42,
                num_inference_steps=8,
                true_cfg_scale=1.0,
            )

            tar_image_path = resize_to_512(
                tar_image_path,
                f"{output_dir}/image"
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 2 [Image processing] completed [{elapsed}s]")
            yield format_progress()

            # STEP-3: Nano3D Editing
            step_start_time = time.time()
            current_step_name = "Step 3"
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🚀 Step 3/4: Running Nano3D pipeline..."
            outputs = self.pipeline.run(
                src_image_rendered,
                tar_image_path,
                source_ply_path  = f"{output_dir}/voxels.ply",
                source_voxel_latent_path=f"{output_dir}/latent.pt",
                source_slat      = result["src_slat"],
                editing_mode     = editing_mode,
                seed             = 1,
                output_path      = output_dir,
                st_step          = st_step,
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 3 [Running Nano3D pipeline] completed [{elapsed}s]")
            yield format_progress()

            # Export to GLB
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "💾 Exporting to GLB..."

            with torch.enable_grad():
                tar_glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )

            output_glb_path = f"{output_dir}/edit_mesh.glb"
            tar_glb.export(output_glb_path)
            elapsed = int(time.time() - step_start_time)
            completed_steps.append("✅ Export completed [%ds]" % elapsed)
            yield format_progress()

            progress = format_progress() + "\n" if completed_steps else ""
            # Return results
            yield {
                "status": progress + "✨ Nano3D generation and editing completed!",
                "src_mesh": src_mesh_path,
                "output_mesh": output_glb_path,
                "src_image": src_image_rendered,
                "edited_image": tar_image_path
            }

    def process_wrapper(src_image, editing_mode, edit_instruction, st_step):
        final_status       = ""
        final_src_mesh     = None
        final_output_mesh  = None
        final_src_image    = None
        final_edited_image = None

        for result in processor.process(
            src_image,
            editing_mode=editing_mode,
            edit_instruction=edit_instruction,
            st_step=st_step
        ):
            if isinstance(result, dict):
                # Save final result
                final_status = result.get("status", "")
                final_src_mesh = result.get("src_mesh")
                final_output_mesh = result.get("output_mesh")
                final_src_image = result.get("src_image")
                final_edited_image = result.get("edited_image")
            else:
                # Progress message - keep previous results visible
                yield result, final_src_mesh, final_output_mesh, final_src_image, final_edited_image

        # Yield final result
        yield final_status, final_src_mesh, final_output_mesh, final_src_image, final_edited_image

    """Create Gradio interface for Nano3D Generator"""
    processor = Nano3DGeneratorProcessor(pipeline, dinov2_model, qwen_image_pipeline)

    with gr.Blocks(title="Nano3D Generator with Editing", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Nano3D Generator with Image Editing")
        gr.Markdown("Upload an image, generate 3D mesh, and edit it with Qwen-Image + Nano3D")

        with gr.Row():
            # Left panel - Input (compact)
            with gr.Column(scale=0.35):
                gr.Markdown("## 📥 Input")

                src_image = gr.Image(
                    label="🖼️ Upload Source Image",
                    type="filepath",
                    scale=1
                )

                gr.Markdown("### ⚙️ Parameters")

                editing_mode = gr.Radio(
                    choices=["add", "remove", "replace"],
                    value="add",
                    label="Editing Mode"
                )

                edit_instruction = gr.Textbox(
                    label="Edit Instruction",
                    value="add a hat on the head.",
                    lines=3
                )

                st_step = gr.Slider(
                    minimum=5,
                    maximum=14,
                    value=11,
                    step=1,
                    label="Editing Aggressiveness",
                    info="Higher values mean closer to source mesh; Lower values mean more aggressive edits"
                )

                process_btn = gr.Button(
                    "🚀 Start Nano3D Generation",
                    variant="primary",
                    scale=3,
                )

            # Right panel - Output (expanded)
            with gr.Column(scale=1.65):
                gr.Markdown("## 📤 Output")

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                    max_lines=15
                )

                with gr.Accordion("📸 Images (click to expand)", open=False):
                    with gr.Row():
                        src_image_output = gr.Image(label="Rendered Source Image", interactive=False, scale=1)
                        edited_image_output = gr.Image(label="Edited Image", interactive=False, scale=1)
                        
                gr.Markdown("### 3D Meshes")
                with gr.Row():
                    src_mesh_output = gr.Model3D(
                        label="Generated Source Mesh",
                        interactive=False,
                        scale=1,
                        height=600
                    )
                    output_mesh = gr.Model3D(
                        label="Edited Output Mesh",
                        interactive=False,
                        scale=1,
                        height=600
                    )

        # Connect processing
        process_btn.click(
            fn=process_wrapper,
            inputs=[src_image, editing_mode, edit_instruction, st_step],
            outputs=[status_output, src_mesh_output, output_mesh, src_image_output, edited_image_output]
        )

    return demo

def create_gradio3(pipeline, dinov2_model, qwen_image_pipeline = None):

    class Nano3DProcessor:
        def __init__(self, pipeline, dinov2_model, qwen_image_pipeline):
            self.pipeline = pipeline
            self.dinov2_model = dinov2_model
            self.qwen_image_pipeline = qwen_image_pipeline
            self.template = """Carefully edit the image to only perform the following change: [Edit content: "{}"], while strictly keeping the rest of the original image unchanged.
                        Do not alter the shape, proportions, colors, textures, pose/structure, composition, or lighting of the original subjects.
                        Do not change any elements that are not directly related to [Edit content].
                        The overall style, sharpness, and level of detail must remain perfectly consistent with the original image.
                        keep white background."""
            self.step1_3_result = None

        def process_step1_to_3(self,
                    src_mesh_file,
                    num_views: int  = 150,
                    resolution: int = 512,
                    multi_cond_stage2 = False):
            """
            Process STEP 1-3: Render and prepare source image
            Yields output_dir and src_image_path when complete
            """
            completed_steps = []

            def format_progress():
                lines = completed_steps.copy()
                return "\n".join(lines)

            output_dir = tempfile.mkdtemp(prefix="nano3d")
            yield f"📁 Working directory: {output_dir}"

            src_mesh_path = src_mesh_file.name if hasattr(src_mesh_file, 'name') else str(src_mesh_file)
            yield f"📥 Input mesh: {os.path.basename(src_mesh_path)}"

            render_path = f"{output_dir}/render"

            # STEP-1: Render 3D model to multi-view images
            step_start_time = time.time()
            yield "🎬 Step 1/5: Rendering 3D model to multi-view images..."
            results1 = process_3d_asset(
                model_path=src_mesh_path,
                output_dir=render_path,
                num_views=num_views,
                resolution=resolution,
                engine="CYCLES",
                dinov2_model=self.dinov2_model,
                batch_size=16,
                voxel_size=1/64
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 1 [Rendering 3D model to multi-view images] completed [{elapsed}s]")
            yield format_progress()

            # STEP-2: Extract and decode voxel
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🔍 Step 2/5: Extracting and decoding voxel..."
            results2 = extract_and_decode_voxel(
                pipeline=self.pipeline,
                render_dir=render_path,
                output_dir=f"{output_dir}/src_voxel"
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 2 [Extracting and decoding voxel] completed [{elapsed}s]")
            yield format_progress()

            # STEP-3: Render front view
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🖼️ Step 3/5: Rendering front view..."
            # ===============================================================
            render_front_view(
                file_path=src_mesh_path,
                output_dir=f"{output_dir}/image",
                output_name="front.png"
            )
            src_image_path   = bg_to_white(f"{output_dir}/image/front.png")
            # ===============================================================
            if multi_cond_stage2 == True:
                render_front_view(
                    file_path=src_mesh_path,
                    output_dir=f"{output_dir}/image",
                    output_name="back.png"
                )
                back_render_path = bg_to_white(f"{output_dir}/image/back.png")
            else:
                back_render_path = ""
            # ===============================================================
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 3 [Rendering front view] completed [{elapsed}s]")
            yield format_progress()

            # Return the output_dir and src_image_path for next stage
            progress = format_progress() + "\n" if completed_steps else ""
            yield {
                "output_dir":       output_dir,
                "src_image_path":   src_image_path,
                "back_render_path": back_render_path,
                "status": progress + "✨ Ready for image editing! Please upload the edited image."
            }

        def process_step4_to_5(self,
                    output_dir: str,
                    src_image_path: str,
                    edited_image_file,
                    editing_mode: str = "add",
                    st_step: int = 12,
                    back_render_path: str = ""):
            """
            Process STEP 4-5: Use uploaded edited image and run Nano3D pipeline
            """
            completed_steps = [
                "✅ Step 1 [Rendering 3D model to multi-view images] completed",
                "✅ Step 2 [Extracting and decoding voxel] completed",
                "✅ Step 3 [Rendering front view] completed"
            ]

            def format_progress():
                lines = completed_steps.copy()
                return "\n".join(lines)

            # STEP-4: Use uploaded edited image
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🎨 Step 4/5: Loading edited image..."
            tar_image_path = f"{output_dir}/image/edited.png"

            if edited_image_file is not None:
                edited_image_file_path = edited_image_file.name if hasattr(edited_image_file, 'name') else str(edited_image_file)
                shutil.copy(edited_image_file_path, tar_image_path)
            else:
                raise ValueError("Edited image is required. Please upload an edited image.")

            tar_image_path = resize_to_512(
                    tar_image_path, 
                    f"{output_dir}/image"
                )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 4 [Loading edited image] completed [{elapsed}s]")
            yield format_progress()

            # STEP-5: Nano3D Editing
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🚀 Step 5/5: Running Nano3D pipeline..."
            outputs = self.pipeline.run(
                src_image_path,
                tar_image_path,
                source_ply_path  = f"{output_dir}/render/voxels.ply",
                source_voxel_latent_path = f"{output_dir}/src_voxel/latent.pt",
                source_slat_path = f"{output_dir}/render/features.npz",
                editing_mode     = editing_mode,
                seed             = 42,
                output_path      = output_dir,
                st_step          = st_step,
                back_render_path = back_render_path
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 5 [Running Nano3D pipeline] completed [{elapsed}s]")
            yield format_progress()

            # Export to GLB
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "💾 Exporting to GLB..."

            with torch.enable_grad():
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )

            output_glb_path = f"{output_dir}/edit_mesh.glb"
            glb.export(output_glb_path)
            elapsed = int(time.time() - step_start_time)
            completed_steps.append("✅ Export completed [%ds]" % elapsed)
            yield format_progress()

            progress = format_progress() + "\n" if completed_steps else ""
            yield {
                "status": progress + "✨ Nano3D editing completed!",
                "output_mesh": output_glb_path,
                "src_image": src_image_path,
                "edited_image": tar_image_path
            }

    def process_step1_3_wrapper(src_mesh, num_views, resolution, editing_mode):
        nonlocal saved_state
        final_status = ""
        final_src_image = None

        for result in processor.process_step1_to_3(
            src_mesh,
            num_views  = num_views,
            resolution = resolution
        ):
            if isinstance(result, dict):
                # Save state to closure variable
                saved_state     = result
                final_status    = result.get("status", "")
                final_src_image = result.get("src_image_path")
            else:
                # 中间进度更新：只更新 status_output
                yield result, final_src_image

        # 最终更新：设置 status 和 image
        yield final_status, final_src_image

    def process_step4_5_wrapper(edited_image_file, editing_mode, st_step):
        if not saved_state or "output_dir" not in saved_state:
            yield "❌ Error: No previous state. Please run STEP 1-3 first.", None
            return

        final_status       = ""
        final_mesh         = None
        final_src_image    = None
        final_edited_image = None

        for result in processor.process_step4_to_5(
            output_dir        = saved_state["output_dir"],
            src_image_path    = saved_state["src_image_path"],
            edited_image_file = edited_image_file,
            editing_mode      = editing_mode,
            st_step           = st_step,
            back_render_path  = saved_state["back_render_path"]
        ):
            if isinstance(result, dict):
                final_status = result.get("status", "")
                final_mesh = result.get("output_mesh")
                final_src_image = result.get("src_image")
                final_edited_image = result.get("edited_image")
            else:
                # 中间进度更新：只更新 status_output，其他保持不变
                yield result, None

        # 最终更新：设置所有输出
        yield final_status, final_mesh

    """Create Gradio interface for Nano3D"""
    processor = Nano3DProcessor(pipeline, dinov2_model, None)

    # Use closure variable to store intermediate state
    saved_state = {}
    
    with gr.Blocks(title="Nano3D Mesh Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Nano3D Mesh Editor")
        gr.Markdown("Upload a 3D mesh and edit it with Image Upload + Nano3D")

        with gr.Row():
            # Left panel - Input
            with gr.Column(scale=1):
                gr.Markdown("## 📥 Input")

                src_mesh = gr.File(
                    label="🔼 Upload Source Mesh",
                    file_types=[".glb", ".obj", ".fbx", ".ply"]
                )

                def update_mesh_preview(file):
                    if file is None:
                        return None
                    return file.name if hasattr(file, 'name') else str(file)

                src_mesh_preview = gr.Model3D(
                    label="📦 Source Mesh Preview",
                    interactive=False
                )
                src_mesh.change(
                    fn=update_mesh_preview,
                    inputs=[src_mesh],
                    outputs=[src_mesh_preview]
                )

                gr.Markdown("### ⚙️ Parameters")

                editing_mode = gr.Radio(
                    choices=["add", "remove", "replace"],
                    value="add",
                    label="Editing Mode"
                )

                st_step = gr.Slider(
                    minimum=5,
                    maximum=14,
                    value=11,
                    step=1,
                    label="Editing Aggressiveness",
                    info="Higher values mean closer to source mesh; Lower values mean more aggressive edits"
                )

                with gr.Accordion("Render Parameter", open=False):
                    num_views = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=30,
                        step=10,
                        label="Number of Views"
                    )

                    resolution = gr.Radio(
                        choices=[256, 512],
                        value=512,
                        label="Resolution"
                    )

                # Step 1-3 processing button
                process_btn_1_3 = gr.Button(
                    "🚀 Step 1: Prepare Source Image",
                    variant="primary",
                    scale=1,
                )

            # Right panel - Output
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Output")

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                    max_lines=10
                )

                gr.Markdown("### Images")

                with gr.Row():
                    src_image_output = gr.Image(
                        label="Source Image", 
                        interactive=False
                    )
                    edited_image_file = gr.Image(
                        label="📤 Upload Edited Image",
                        type="filepath"
                    )
                process_btn_4_5 = gr.Button(
                    "🚀 Step 2: Complete Nano3D Editing",
                    variant="primary",
                    scale=1,
                )

                output_mesh = gr.Model3D(
                    label="Output Mesh",
                    interactive=False
                )

        # Connect processing - Stage 1-3
        process_btn_1_3.click(
            fn=process_step1_3_wrapper,
            inputs=[src_mesh, num_views, resolution, editing_mode],
            outputs=[status_output, src_image_output]
        )

        # Connect processing - Stage 4-5
        process_btn_4_5.click(
            fn=process_step4_5_wrapper,
            inputs=[edited_image_file, editing_mode, st_step],
            outputs=[status_output, output_mesh]
        )

    return demo

def create_gradio4(pipeline, dinov2_model, qwen_image_pipeline = None):

    class Nano3DGeneratorWithoutQwenProcessor:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def process_step0_to_1(self, src_input_image):
            """
            Process STEP 0-1: Generate source mesh from image and render front view
            Yields status and outputs when complete
            """
            completed_steps = []

            def format_progress():
                lines = completed_steps.copy()
                return "\n".join(lines)

            output_dir = tempfile.mkdtemp(prefix="nano3d_gen_")
            yield f"📁 Working directory: {output_dir}"

            src_image_path = src_input_image.name if hasattr(src_input_image, 'name') else str(src_input_image)
            yield f"📥 Input image: {os.path.basename(src_image_path)}"

            os.makedirs(f"{output_dir}/image", exist_ok=True)

            # STEP-0: Source mesh generation and feature extraction
            step_start_time = time.time()
            yield "🔧 Step 0/2: Generating source mesh from image..."
            result = self.pipeline.run_custom(
                src_image_path,
                seed=1,
                output_path=output_dir,
            )
            # Export source mesh to GLB
            with torch.enable_grad():
                src_glb = postprocessing_utils.to_glb(
                    result["src_mesh"]['gaussian'][0],
                    result["src_mesh"]['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )
            src_mesh_path = f"{output_dir}/src_mesh.glb"
            src_glb.export(src_mesh_path)
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 0 [Generating source mesh] completed [{elapsed}s]")
            yield format_progress()

            # STEP-1: Render front view
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🖼️ Step 1/2: Rendering front view..."
            render_front_view(
                file_path=src_mesh_path,
                output_dir=f"{output_dir}/image",
                output_name="front.png"
            )
            src_image_rendered = bg_to_white(f"{output_dir}/image/front.png")
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 1 [Rendering front view] completed [{elapsed}s]")
            yield format_progress()

            progress = format_progress() + "\n" if completed_steps else ""
            yield {
                "output_dir": output_dir,
                "src_mesh_path": src_mesh_path,
                "src_image_rendered": src_image_rendered,
                "src_slat": result["src_slat"],
                "status": progress + "✨ Ready for image editing! Please upload the edited image."
            }

        def process_step2_to_3(self,
                            output_dir: str,
                            src_mesh_path: str,
                            src_image_rendered: str,
                            src_slat,
                            edited_image_file,
                            editing_mode: str = "add",
                            st_step: int = 12):
            """
            Process STEP 2-3: Use uploaded edited image and run Nano3D pipeline
            """
            completed_steps = [
                "✅ Step 0 [Generating source mesh] completed",
                "✅ Step 1 [Rendering front view] completed"
            ]

            def format_progress():
                lines = completed_steps.copy()
                return "\n".join(lines)

            # STEP-2: Use uploaded edited image
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🎨 Step 2/4: Loading edited image..."
            tar_image_path = f"{output_dir}/image/edited.png"

            if edited_image_file is not None:
                edited_image_file_path = edited_image_file.name if hasattr(edited_image_file, 'name') else str(edited_image_file)
                shutil.copy(edited_image_file_path, tar_image_path)
            else:
                raise ValueError("Edited image is required. Please upload an edited image.")

            tar_image_path = resize_to_512(
                tar_image_path,
                f"{output_dir}/image"
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 2 [Loading edited image] completed [{elapsed}s]")
            yield format_progress()

            # STEP-3: Nano3D Editing
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "🚀 Step 3/4: Running Nano3D pipeline..."
            outputs = self.pipeline.run(
                src_image_rendered,
                tar_image_path,
                source_ply_path=f"{output_dir}/voxels.ply",
                source_voxel_latent_path=f"{output_dir}/latent.pt",
                source_slat=src_slat,
                editing_mode=editing_mode,
                seed=1,
                output_path=output_dir
            )
            elapsed = int(time.time() - step_start_time)
            completed_steps.append(f"✅ Step 3 [Running Nano3D pipeline] completed [{elapsed}s]")
            yield format_progress()

            # Export to GLB
            step_start_time = time.time()
            progress = format_progress() + "\n" if completed_steps else ""
            yield progress + "💾 Exporting to GLB..."

            with torch.enable_grad():
                tar_glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )

            output_glb_path = f"{output_dir}/edit_mesh.glb"
            tar_glb.export(output_glb_path)
            elapsed = int(time.time() - step_start_time)
            completed_steps.append("✅ Export completed [%ds]" % elapsed)
            yield format_progress()

            progress = format_progress() + "\n" if completed_steps else ""
            yield {
                "status": progress + "✨ Nano3D generation and editing completed!",
                "src_mesh": src_mesh_path,
                "output_mesh": output_glb_path,
                "src_image": src_image_rendered,
                "edited_image": tar_image_path
            }

    def process_step0_1_wrapper(src_image):
        nonlocal saved_state
        final_status = ""
        final_src_mesh = None
        final_src_image = None

        for result in processor.process_step0_to_1(src_image):
            if isinstance(result, dict):
                # Save state to closure variable
                saved_state = result
                final_status = result.get("status", "")
                final_src_mesh = result.get("src_mesh_path")
                final_src_image = result.get("src_image_rendered")
            else:
                # Progress message
                yield result, final_src_mesh, final_src_image

        # Final update
        yield final_status, final_src_mesh, final_src_image

    def process_step2_3_wrapper(edited_image_file, editing_mode, st_step):
        if not saved_state or "output_dir" not in saved_state:
            yield "❌ Error: No previous state. Please run STEP 0-1 first.", None, None, None
            return

        final_status = ""
        final_src_mesh = None
        final_output_mesh = None
        final_src_image = None
        final_edited_image = None

        for result in processor.process_step2_to_3(
            output_dir=saved_state["output_dir"],
            src_mesh_path=saved_state["src_mesh_path"],
            src_image_rendered=saved_state["src_image_rendered"],
            src_slat=saved_state["src_slat"],
            edited_image_file=edited_image_file,
            editing_mode=editing_mode,
            st_step=st_step
        ):
            if isinstance(result, dict):
                final_status = result.get("status", "")
                final_src_mesh = result.get("src_mesh")
                final_output_mesh = result.get("output_mesh")
                final_src_image = result.get("src_image")
                final_edited_image = result.get("edited_image")
            else:
                # Progress message
                yield result, final_src_mesh, final_output_mesh, final_src_image, final_edited_image

        # Final update
        yield final_status, final_src_mesh, final_output_mesh, final_src_image, final_edited_image

    """Create Gradio interface for Nano3D Generator (without Qwen-Image)"""
    processor = Nano3DGeneratorWithoutQwenProcessor(pipeline)

    # Use closure variable to store intermediate state
    saved_state = {}
    
    with gr.Blocks(title="Nano3D Generator (No Qwen-Image)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Nano3D Generator & Editor (Manual Image Editing)")
        gr.Markdown("Upload an image → Generate 3D mesh → Upload edited image → Complete Nano3D editing")

        with gr.Row():
            # Left panel - Input
            with gr.Column(scale=0.35):
                gr.Markdown("## 📥 Input")

                src_image = gr.Image(
                    label="🖼️ Upload Source Image",
                    type="filepath"
                )

                gr.Markdown("### ⚙️ Parameters")

                editing_mode = gr.Radio(
                    choices=["add", "remove", "replace"],
                    value="add",
                    label="Editing Mode"
                )

                st_step = gr.Slider(
                    minimum=5,
                    maximum=14,
                    value=11,
                    step=1,
                    label="Editing Aggressiveness",
                    info="Higher values = closer to source; Lower values = more aggressive"
                )

                gr.Markdown("### 📋 Workflow")
                process_btn_0_1 = gr.Button(
                    "🚀 Step 1: Generate Mesh",
                    variant="primary",
                    scale=3,
                )

                gr.Markdown("---")

                edited_image_file = gr.Image(
                    label="📤 Upload Edited Image",
                    type="filepath"
                )

                process_btn_2_3 = gr.Button(
                    "🚀 Step 2: Complete Editing",
                    variant="primary",
                    scale=3,
                )

            # Right panel - Output
            with gr.Column(scale=1.65):
                gr.Markdown("## 📤 Output")

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                    max_lines=15
                )

                with gr.Accordion("📸 Images (click to expand)", open=False):
                    with gr.Row():
                        src_image_output = gr.Image(label="Rendered Source Image", interactive=False, scale=1)
                        edited_image_output = gr.Image(label="Edited Image", interactive=False, scale=1)

                gr.Markdown("### 3D Meshes")
                with gr.Row():
                    src_mesh_output = gr.Model3D(
                        label="Generated Source Mesh",
                        interactive=False,
                        scale=1,
                        height=600
                    )
                    output_mesh = gr.Model3D(
                        label="Edited Output Mesh",
                        interactive=False,
                        scale=1,
                        height=600
                    )

        # Connect processing - Step 0-1
        process_btn_0_1.click(
            fn=process_step0_1_wrapper,
            inputs=[src_image],
            outputs=[status_output, src_mesh_output, src_image_output]
        )

        # Connect processing - Step 2-3
        process_btn_2_3.click(
            fn=process_step2_3_wrapper,
            inputs=[edited_image_file, editing_mode, st_step],
            outputs=[status_output, src_mesh_output, output_mesh, src_image_output, edited_image_output]
        )

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nano3D Gradio Interface with optional Qwen-Image support")
    parser.add_argument(
        "--use-qwen-image",
        action="store_true",
        help="Enable Qwen-Image for automatic image editing"
    )
    parser.add_argument(
        "--input-mesh",
        action="store_true",
        help="Use 3D mesh as input (uncheck for image)"
    )
    parser.add_argument(
        "--qwen-image-lora-path",
        type=str,
        help="Path to Qwen-Image LoRA weights"
    )
    args = parser.parse_args()
    # ============================================================================
    # Load pipeline and model
    # ============================================================================
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    pipeline = load_sparse_structure_encoder(pipeline)
    pipeline = inject_methods(pipeline)
    print(f"Loading TRELLIS pipeline Done\n")

    # Load DINOv2 model
    print(f"Loading DINOv2 model...")
    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
    dinov2_model.eval().cuda()
    print("DINOv2 loaded\n")

    # Load Qwen-Image pipeline once
    if args.use_qwen_image:
        print("Loading Qwen-Image model...")
        from inference.qwen_image_edit import qwen_image_edit_main, load_qwen_image
        qwen_image_pipeline = load_qwen_image(
            model_name = "Qwen/Qwen-Image-Edit-2509",
            lora_path  = args.qwen_image_lora_path
        )
        print("Qwen-Image loaded\n")

    if args.use_qwen_image and args.input_mesh:
        print("Using 3D mesh as input")
        demo = create_gradio1(
            pipeline,
            dinov2_model,
            qwen_image_pipeline
        )
    elif args.use_qwen_image and not args.input_mesh:
        print("Using image as input")
        demo = create_gradio2(
            pipeline,
            dinov2_model,
            qwen_image_pipeline
        )
    elif not args.use_qwen_image and args.input_mesh:
        print("Using 3D mesh as input without Qwen-Image")
        demo = create_gradio3(
            pipeline,
            dinov2_model
        )
    else:
        print("Using image as input without Qwen-Image")
        demo = create_gradio4(
            pipeline,
            dinov2_model
        )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )