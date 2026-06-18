# CLIP-RD

CLIP-RD is a knowledge distillation framework for CLIP that extends relational distillation beyond the conventional image-text relation. It distills a large teacher CLIP model into a smaller student model by using feature distillation, interactive contrastive learning, and relational distillation losses including HRD, VRD, and XRD.

## About source code

This repository contains the source code for training and evaluating CLIP-RD.

The main components are organized as follows:

* `src/open_clip/`: CLIP model implementation based on OpenCLIP.
* `src/training/`: training, evaluation, loss computation, distributed training, and zero-shot evaluation code.
* `src/data/`: dataset-related utilities.
* `script/ViT_B_16_Laion400M/`: training scripts for baseline, KD, and CLIP-RD settings.
* `script/eval/`: evaluation scripts for image-text retrieval and zero-shot classification.
* `tests/`: unit and integration tests for model loading, inference, and training.

The provided scripts include:

* `ViT_T_16_baseline.sh`: trains the student CLIP model without distillation.
* `ViT_T_16_KD.sh`: trains the student model with the KD baseline.
* `ViT_T_16_RD.sh`: trains the student model with the proposed CLIP-RD losses.
* `eval_coco.sh`: evaluates image-text retrieval on MSCOCO.
* `eval_flickr.sh`: evaluates image-text retrieval on Flickr.
* `eval_imagenet.sh`: evaluates zero-shot classification on ImageNet-related datasets.

## How to build

This project does not require a separate build step. After installing the required Python packages, the code can be executed directly from the source directory.

```bash
git clone https://github.com/Rosieyang01/CLIP-RD.git
cd CLIP-RD
```

If needed, set the Python path before running training or evaluation:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## How to install

We recommend using a virtual environment.

```bash
conda create -n clip-rd python=3.9
conda activate clip-rd
```

Install the training dependencies:

```bash
pip install -r requirements-training.txt
```

Install the test dependencies:

```bash
pip install -r requirements-test.txt
```

The main dependencies include PyTorch, torchvision, WebDataset, pandas, tqdm, Hugging Face Hub, and Transformers.

## How to test

Run the test suite with `pytest`:

```bash
pytest tests
```

For faster testing, multiple workers can be used:

```bash
pytest tests -n auto
```

The tests cover basic model loading, pretrained checkpoint downloading, inference, and simple training behavior.

## Description of Data

### Conceptual Captions 3M

OpenCLIP reads a CSV file containing two columns: an image path and its corresponding text caption.

First, download the Conceptual Captions 3M URL files. For simplicity, rename `Train_GCC-training` to `cc3m_train.tsv` and `Validation_GCC-1.1.0-Validation` to `cc3m_val.tsv`. Then, run:

```bash
python src/data/gather_cc.py \
    path/to/cc3m/images/ \
    path/to/cc3m_train.tsv \
    path/to/cc3m_val.tsv
```

The generated `cc3m_train.csv` and `cc3m_val.csv` files have the following format:

| title | filepath |
|---|---|
| XXXXXX | train/X/X.jpg |
| ... | ... |

Our downloaded CC3M dataset contains approximately **2.89M training images** and **13K validation images**.

### Conceptual Captions 12M

First, download the Conceptual Captions 12M URL file. Then, run:

```bash
python src/data/gather_cc12m.py \
    path/to/cc12m/images/ \
    path/to/cc12m.tsv
```

The generated `cc12m.csv` file has the following format:

| title | filepath |
|---|---|
| XXXXXX | train/X/X.jpg |
| ... | ... |

Our downloaded CC12M training dataset contains approximately **9.97M images**.

> Replace all `path/to/...` entries with the corresponding local dataset paths before running the scripts.

## Result

We evaluate CLIP-RD on zero-shot classification and zero-shot cross-modal retrieval tasks. The teacher model is ViT-B/16, and the student model is ViT-T/16. For retrieval tasks, we report Recall@1 (R@1) for Image-to-Text (I2T) and Text-to-Image (T2I).

### Main Results

| Method | IN-1K | MSCOCO I2T | MSCOCO T2I | Flickr I2T | Flickr T2I |
|---|---:|---:|---:|---:|---:|
| T: ViT-B/16 | 67.1 | 39.5 | 36.5 | 76.5 | 75.5 |
| S: ViT-T/16 | 29.3 | 18.2 | 17.9 | 39.3 | 42.0 |
| TinyCLIP | 40.8 | 26.8 | 24.7 | 58.6 | 58.5 |
| CLIP-KD* | 41.3 | 27.4 | 24.1 | 58.4 | 56.4 |
| CLIP-RD (Ours) | 42.1 | 27.8 | 25.1 | 58.3 | 58.6 |

CLIP-RD achieves 42.1% zero-shot accuracy on ImageNet-1K, outperforming the ViT-T/16 student baseline by 12.8%p, TinyCLIP by 1.3%p, and CLIP-KD by 0.8%p. On MSCOCO, CLIP-RD improves I2T and T2I retrieval over CLIP-KD by 0.4%p and 1.0%p, respectively. On Flickr, CLIP-RD achieves a 2.2%p improvement over CLIP-KD in T2I retrieval.

### Additional Zero-Shot Results

| Method | IN-1K | IN-V2 | IN-R | IN-S | CC3M I2T | CC3M T2I |
|---|---:|---:|---:|---:|---:|---:|
| T: ViT-B/16 | 67.1 | 59.6 | 77.9 | 52.4 | 42.8 | 42.2 |
| S: ViT-T/16 | 29.3 | 24.9 | 34.2 | 16.9 | 33.6 | 34.0 |
| CLIP-KD | 41.3 | 35.5 | 46.3 | 26.3 | 40.2 | 38.7 |
| CLIP-RD (Ours) | 42.1 | 36.2 | 48.3 | 27.3 | 40.6 | 39.3 |

CLIP-RD consistently outperforms CLIP-KD on ImageNet variants and CC3M retrieval. In particular, it improves ImageNet-R by 2.0%p and CC3M retrieval by 0.4%p / 0.6%p on I2T / T2I.

### Zero-Shot Classification on Various Datasets

| Method | IN | CIFAR-10 | CIFAR-100 | EuroSAT | Food101 | RESISC45 | Sun397 | Caltech101 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T: ViT-B/16 | 67.1 | 91.1 | 71.3 | 33.8 | 80.2 | 57.8 | 69.0 | 87.0 |
| S: ViT-T/16 | 29.3 | 66.9 | 27.5 | 11.6 | 27.8 | 24.4 | 38.1 | 66.2 |
| CLIP-KD | 41.3 | 74.2 | 39.9 | 18.0 | 41.6 | 32.6 | 51.6 | 75.9 |
| CLIP-RD (Ours) | 42.1 | 75.5 | 42.3 | 25.5 | 43.2 | 32.6 | 52.0 | 78.0 |

Across diverse zero-shot classification benchmarks, CLIP-RD improves over CLIP-KD by up to 7.5%p, with strong gains on EuroSAT, CIFAR-100, Food101, Sun397, and Caltech101.
