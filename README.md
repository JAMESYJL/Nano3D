# Nano3D: A Training-Free Approach for Efficient 3D Editing Without Masks

[**Paper**] | [**Project Page**]

Official implementation of Nano3D: A Training-Free Approach for Efficient 3D Editing Without Masks

[Junliang Ye*](https://jamesyjl.github.io/), [Shenghao Xie*](https://shxie2020.github.io/), [Ruowen Zhao](https://zhaorw02.github.io/), [Zhengyi Wang](https://thuwzy.github.io/), Hongyu Yan, Wenqiang Zu, Lei Ma, Jun Zhu.

<p align="center"> All Code will be released soon... 🏗️ 🚧 🔨</p>

Abstract: *3D object editing is essential for interactive content creation in gaming, animation, and robotics, yet current approaches remain inefficient, inconsistent, and often fail to preserve unedited regions. Most methods rely on editing multi-view renderings followed by reconstruction, which introduces artifacts and limits practicality. To address these challenges, we propose \textbf{Nano3D}, a training-free framework for precise and coherent 3D object editing without masks. Nano3D integrates FlowEdit into TRELLIS to perform localized edits guided by front-view renderings, and further introduces region-aware merging strategies, Voxel/Slat-Merge, which adaptively preserve structural fidelity by ensuring consistency between edited and unedited areas. Experiments demonstrate that Nano3D achieves superior 3D consistency and visual quality compared with existing methods. Based on this framework, we construct the first large-scale 3D editing datasets \textbf{Nano3D-Edit-100k}, which contains over 100,000 high-quality 3D editing pairs. This work addresses long-standing challenges in both algorithm design and data availability, significantly improving the generality and reliability of 3D editing, and laying the groundwork for the development of feed-forward 3D editing models.*

<p align="center">
    <img src="assets/teaser.png">
</p>

## Method

Overall Framework of Nano3D. The original 3D object is voxelized and encoded into sparse structure and structured latent respectively. Stage 1 modifies geometry via Flow Transformer with FlowEdit, guided by Nano Banana–edited images. Stage 2 generates structured latents with Sparse Flow Transformer, supporting TRELLIS-inherent appearance editing. Voxel/Slat-Merge further ensures consistency across both stages before decoding the final 3D object.
<p align="center">
    <img src="assets/method.png">
</p>

## BibTeX

```bibtex
@article{ye2025shapellm,
  title={ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding},
  author={Ye, Junliang and Wang, Zhengyi and Zhao, Ruowen and Xie, Shenghao and Zhu, Jun},
  journal={arXiv preprint arXiv:2506.01853},
  year={2025}
}
```
