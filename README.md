# ViTacGen Code and Data
Official Repository of **ViTacGen: Robotic Pushing with Vision-to-Touch Generation (RA-L 2025)**

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
    <a href='#'>
      <img src='https://img.shields.io/badge/Project-Page-66C0FF?style=plastic&logo=Google%20chrome&logoColor=66C0FF' alt='Project Page'>
    </a>
</p>

## Overview


## Checkpoints and Dataset Release

We will soon release the checkpoints and dataset for vision-to-touch generation training. Stay tuned! 

## Dependencies

Create a new conda environment. My CUDA version (nvcc --version) is 12.4
```bash  
conda create -n vitacgen python=3.8  
conda activate vitacgen
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
``` 

Install required packages.
```bash  
pip install -r requirements.txt
``` 

## Usage


### Data Usage


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

## Contact

If you have any questions, feel free to contact me through email at [zhiyuan.1.wu@kcl.ac.uk](mailto:zhiyuan.1.wu@kcl.ac.uk).
