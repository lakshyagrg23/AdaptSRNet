# AdaptSRNet: Enhancing Image Steganalysis via Adaptive Filter-Attention Fusion

[![Framework](https://img.shields.io/badge/PyTorch-Lightning-purple)](https://www.pytorchlightning.ai/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Submitted-green)](https://link.springer.com/journal/371)

This repository contains the official PyTorch implementation of the paper **"AdaptSRNet: Enhancing Image Steganalysis via Adaptive Filter-Attention Fusion"**, currently submitted to *The Visual Computer* (Springer).

## Abstract
Binary image steganalysis aims to detect hidden information within images. This study introduces **AdaptSRNet**, a model integrating learnable SRM-based filters, multi-scale feature extraction, and attention mechanisms. Evaluated on BOSSBase 1.01 at a payload of 0.4 bpp across four steganographic schemes (WOW, S-UNIWARD, HILL, and HUGO), AdaptSRNet achieved peak accuracy of **86.42%** on WOW, with AUC values consistently above 0.90. These results position AdaptSRNet competitively among state-of-the-art methods while maintaining computational efficiency.

## Architecture
AdaptSRNet consists of five main components:
1.  **Learnable SRM Front-End:** Initialized with domain-specific filters (KV, Edge, Laplacian) but fine-tuned during training.
2.  **Multi-Scale Feature Extractor:** Captures residuals at various resolutions.
3.  **Enhanced Residual Backbone:** Utilizes SE (Squeeze-and-Excitation) blocks.
4.  **CBAM Attention:** Applies spatial and channel attention.
5.  **Dual Global Pooling:** Combines AvgPool and MaxPool statistics.

## Installation

### Prerequisites
* Python >= 3.8
* PyTorch >= 1.10
* CUDA (for GPU support)

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/AdaptSRNet.git](https://github.com/YOUR_USERNAME/AdaptSRNet.git)
    cd AdaptSRNet
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation
For this study, we utilized the pre-processed dataset hosted on Kaggle:
* **Dataset Link:** [BOSSBase & BOWS2 (Kaggle)](https://www.kaggle.com/datasets/zapak1010/bossbase-bows2)

After downloading, ensure your directory structure matches the format below. The code expects the `cover` images to be in the root `cover` folder and algorithm-specific stego images to be nested under `stego/{ALGORITHM}/{PAYLOAD}/stego`.

```text
GBRASNET/
└── BOSSBase-1.01/
    ├── cover/
    └── stego/
        ├── HILL/
        │   ├── 0.2bpp/
        │   └── 0.4bpp/
        │       └── stego/
        ├── HUGO/
        ├── MiPOD/
        ├── S-UNIWARD/
        └── WOW/
```

## Usage

### Training
To train the model on a specific algorithm (e.g., WOW), run:

```bash
python train.py --algo WOW --data_dir ./data --batch_size 16 --epochs 100
```

**Key Arguments:**
* `--algo`: Steganography algorithm (WOW, S-UNIWARD, HILL, HUGO).
* `--payload`: Payload size (default: 0.4).
* `--lr`: Learning rate (default: 1e-3 with Cosine Annealing).

### Evaluation
To evaluate a trained checkpoint:

```bash
python test.py --checkpoint_path lightning_logs/version_0/checkpoints/best.ckpt
```

## Results (BOSSBase @ 0.4 bpp)

| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WOW** | 86.42% | 88.15% | 84.10% | 86.08% | 0.94 |
| **S-UNIWARD** | 85.10% | 86.50% | 83.15% | 84.79% | 0.93 |
| **HILL** | 84.85% | 86.20% | 82.95% | 84.54% | 0.92 |
| **HUGO** | 82.35% | 83.90% | 80.05% | 81.93% | 0.90 |

## Citation
If you find this code or research useful, please cite our manuscript:

```bibtex
@article{AdaptSRNet2025,
  title={AdaptSRNet: Enhancing Image Steganalysis via Adaptive Filter-Attention Fusion},
  author={Vollala, Satyanarayana and Varshney, Tarang and Garg, Lakshya and Patel, Parth},
  journal={The Visual Computer},
  year={2025},
  note={Submitted}
}
```
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
