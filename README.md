# ViTacGen Code and Data
Official Repository of **ViTacGen: Robotic Pushing with Vision-to-Touch Generation** (RA-L 2025)

<span class="author-block">
  <a href="https://georgewuzy.github.io/" style="color: #4A90E2 !important;">Zhiyuan Wu</a><sup>1</sup>,
</span>
<span class="author-block">
  <a href="https://yijionglin.github.io/" style="color: #4A90E2 !important;">Yijiong Lin</a><sup>2</sup>,
</span>
<span class="author-block">
  <a href="https://rancho-zhao.github.io/" style="color: #4A90E2 !important;">Yongqiang Zhao</a><sup>1</sup>,
</span>
<span class="author-block">
  <a href="https://scholar.google.com/citations?user=osQ5dAkAAAAJ&hl=zh-CN&oi=ao" style="color: #4A90E2 !important;">Xuyang Zhang</a><sup>1</sup>,
</span>
<span class="author-block">
  <a href="https://zhuochenn.github.io/" style="color: #4A90E2 !important;">Zhuo Chen</a><sup>1</sup>,
</span>
<span class="author-block">
  <a href="https://lepora.com/" style="color: #4A90E2 !important;">Nathan Lepora</a><sup>2</sup>,
</span>
<span class="author-block">
  <a href="https://shanluo.github.io/" style="color: #4A90E2 !important;">Shan Luo</a><sup>1</sup>
</span>

<sup>1</sup> King's College London, <sup>2</sup> University of Bristol

<p align="center">
    <a href='https://arxiv.org/abs/2510.14117'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://arxiv.org/pdf/2510.14117'>
      <img src='https://img.shields.io/badge/Paper-PDF-FF9547?style=plastic&logo=adobeacrobatreader&logoColor=FF9547' alt='Paper PDF'>
    </a>
    <a href='https://robot-perception-lab.github.io/vitacgen-website/'>
      <img src='https://img.shields.io/badge/Project-Page-66C0FF?style=plastic&logo=Google%20chrome&logoColor=66C0FF' alt='Project Page'>
    </a>
</p>

## Checkpoints and Dataset Release

We will soon release our **checkpoints** and **dataset**. Stay tuned! 

## Overview
![Simulation Results](assets/sim.gif)
![Unseen Objects](assets/unseen.gif)

In this paper, we propose **ViTacGen**, a novel robot manipulation framework designed for visual robotic pushing with vision-to-touch generation in reinforcement learning to eliminate the reliance on high-resolution real tactile sensors, enabling effective zero-shot deployment on visual-only robotic systems.

![Pipeline](assets/pipeline.gif)

ViTacGen comprises two components: 1) **VT-Gen** for vision-to-touch generation, and 2) **VT-Con** for reinforcement learning on visual and generated tactile contact depth images with contrastive learning.

## Dependencies


## Usage


### Pretrain expert policy


### Collect visual-tactile data pairs


### Train vision-to-touch generation (VT-Gen)


### Train policy with vision-to-touch generation (VT-Con)


### Evaluation and inference


## Citation

If you find this work helpful, please consider citing us using the following BibTeX entry:

```bibtex  
@article{wu2025vitacgen,  
  title={ViTacGen: Robotic Pushing with Vision-to-Touch Generation},  
  author={Wu, Zhiyuan and Lin, Yijiong and Zhao, Yongqiang and Zhang, Xuyang and Chen, Zhuo and Lepora, Nathan and Luo, Shan},  
  journal={IEEE Robotics and Automation Letters},  
  year={2025},  
  publisher={IEEE}
}  
```

## Acknowledgement

Our code is based on [Tactile Gym](https://github.com/ac-93/tactile_gym) and [M2CURL](https://github.com/ligerfotis/M2CURL). We sincerely appreciate their excellent works.

## Contact

If you have any questions, feel free to contact me through email at [zhiyuan.1.wu@kcl.ac.uk](mailto:zhiyuan.1.wu@kcl.ac.uk).
