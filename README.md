# Custom Tip-Adapter + TIMO: Training-Free Few-Shot Classification with CLIP

This repository implements a custom few-shot classification model by integrating:

1. **Tip-Adapter**: A training-free adapter for CLIP that constructs a key-value cache from few-shot visual features, enhancing zero-shot performance with minimal computational overhead.  
2. **TIMO**: A training-free framework introducing Image-Guided Text (IGT) and Text-Guided Image (TGI) modules to refine prompts and image embeddings via cross-modal guidance.

By integrating TIMO’s prompt and feature refinement into Tip-Adapter’s cache pipeline, we achieve better semantic alignment and stronger few-shot accuracy with only minor code changes.

At its core, Tip-Adapter builds a key-value cache of support image features and connects them directly to their labels. During inference, it fuses these adapted logits with CLIP’s zero-shot predictions using a residual weighting scheme.
But while Tip-Adapter is simple and effective, it doesn’t fully exploit the cross-modal relationship between images and their textual labels.

---
<div align="center">
  <img width=900 src="Custom Tip-TIMO.png"/>
</div>

## 📦 Installation  


## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
git clone https://github.com/gaopengcuhk/Tip-Adapter.git
cd Tip-Adapter

conda create -n tip_adapter python=3.7
conda activate tip_adapter

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Follow [DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to install ImageNet and other 10 datasets referring to CoOp.

## Get Started
### Configs
The running configurations can be modified in `configs/dataset.yaml`, including shot numbers, visual encoders, and hyperparamters. 

For simplicity, we provide the hyperparamters achieving the overall best performance on 1\~16 shots for a dataset, which accord with the scores reported in the paper. If respectively tuned for different shot numbers, the 1\~16-shot performance can be further improved. You can edit the `search_scale`, `search_step`, `init_beta` and `init_alpha` for fine-grained tuning.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.

### Numerical Results
We provide Tip-Adapter's **numerical results** in Figure 4 and 5 of the paper at [exp.log](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/exp.log).

 CLIP-Adapter's numerical results are also updated for comparison.

### 📊 Model Performance Summary

| Model                 | Validation Accuracy (%) | Test Accuracy (%)     |
|----------------------|--------------------------|------------------------|
| Zero-shot CLIP       | 86.23                    | 85.84                  |
| Tip-Adapter          | 87.75                    | 87.51                  |
| Tip-Adapter-F        | —                        | 90.02                  |
| Tip-Adapter-F (FT)   | —                        | 90.06 @ epoch 14       |
| Tip-Adapter-F (2nd)  | 92.66 (val)              | 90.79                  |

### Running

Caltech datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dataset.yaml
```
The fine-tuning of Tip-Adapter-F will be automatically conducted after the training-free Tip-Adapter.

# 📖 Usage

1. Prepare your dataset (e.g., Caltech101) under `~/datasets/caltech101/` with class-folder structure.

2. Configure `configs/caltech101.yaml`:

```yaml
dataset: caltech101
root_path: ~/datasets/caltech101
shots: 16
backbone: ViT-B/16
init_alpha: 0.1
init_beta: 20.0
augment_epoch: 1



## Acknowledgement
This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter). Thanks for their wonderful works.

## Citation
```bash
@article{zhang2021tip,
  title={Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling},
  author={Zhang, Renrui and Fang, Rongyao and Gao, Peng and Zhang, Wei and Li, Kunchang and Dai, Jifeng and Qiao, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2111.03930},
  year={2021}
}
```


