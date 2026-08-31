# CLIP-RD

CLIP-RD is a knowledge distillation framework for CLIP that extends relational distillation beyond the conventional image-text relation. It distills a large teacher CLIP model into a smaller student model by using feature distillation, interactive contrastive learning, and relational distillation losses including HRD, VRD, and XRD.

## About source code

This repository contains the source code for training and evaluating CLIP-RD.

The main components are organized as follows:

* `src/open_clip/`: CLIP model implementation based on OpenCLIP.
* `src/training/`: training, evaluation, loss computation, distributed training, and zero-shot evaluation code.
* `src/data/`: dataset-related utilities.
* `script/distillation/`: training scripts for baseline, KD, and CLIP-RD settings.
* `script/eval/`: evaluation scripts for image-text retrieval and zero-shot classification.
* `tests/`: unit and integration tests for model loading, inference, and training.

The provided scripts include:

* `student_baseline.sh`: trains the student CLIP model without distillation.
* `student_KD.sh`: trains the student model with the KD baseline.
* `student_RD.sh`: trains the student model with the proposed CLIP-RD losses.
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

We evaluate CLIP-RD on zero-shot classification and zero-shot cross-modal retrieval tasks. The teacher model is ViT-L/14, and the student model is ViT-B/16. For retrieval tasks, we report Recall@1 (R@1) for Image-to-Text (I2T) and Text-to-Image (T2I).

### Main Results

| Method | IN-1K | MSCOCO I2T | MSCOCO T2I | Flickr I2T | Flickr T2I |
|---|---:|---:|---:|---:|---:|
| T: ViT-L/14 | 72.8 | 42.6 | 40.9 | 80.7 | 79.4 |
| S: ViT-B/16 | 35.5 | 23.0 | 22.9 | 49.8 | 49.5 |
| CLIP-KD | 55.4 | 37.1 | 35.1 | 73.3 | 69.7 |
| CLIP-RD (Ours) | 57.2 | 37.8 | 36.7 | 73.7 | 71.7 |

CLIP-RD achieves 57.2% accuracy and outperforms the baseline of ViT-B/16 by 21.7%p and CLIP-KD by 1.8\%p. We observe that CLIP-RD improves I2T retrieval R@1 by 0.7%p on MSCOCO and 0.4%p on Flickr over CLIP-KD. For T2I retrieval, our framework also outperforms CLIP-KD by 1.6%p on MSCOCO and 2.0%p on Flickr.

### Robustness to Domain Shifts

| Method | IN-1K | IN-V2 | IN-R | IN-S |
|---|---:|---:|---:|---:|
| T: ViT-L/14 | 72.8 | 65.5 | 84.7 | 59.6 |
| S: ViT-B/16 | 35.5 | 31.1 | 46.8 | 24.5 |
| CLIP-KD | 55.4 | 48.3 | 69.8 | 43.5 |
| CLIP-RD (Ours) | 57.2 | 49.4 | 71.8 | 44.8 |

CLIP-RD consistently outperforms CLIP-KD on ImageNet variants. CLIP-RD consistently outperforms CLIP-KD across all datasets, yielding improvements ranging from 1.1%p to 2.0%p.

### Zero-Shot Classification on Various Datasets

| Method | CIFAR-10 | CIFAR-100 | Caltech101 | EuroSAT | Food101 | RESISC45 | Sun397 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T: ViT-L/14 | 94.7 | 77.7 | 88.4 | 41.3 | 84.9 | 63.2 | 71.5 |
| S: ViT-B/16 | 79.5 | 39.4 | 75.8 | 19.1 | 36.2 | 33.0 | 45.6 |
| CLIP-KD | 87.5 | 61.7 | 84.7 | 28.1 | 58.1 | 46.9 | 61.1 |
| CLIP-RD (Ours) | 87.9 | 62.7 | 85.3 | 34.8 | 60.0 | 48.7 | 62.7 |

CLIP-RD outperforms CLIP-KD across all datasets. On object classification benchmarks such as CIFAR-10/100 and Caltech101, it surpasses CLIP-KD by 0.4%p to 1.0%p. Notably, on EuroSAT, a challenging satellite image dataset, CLIP-RD demonstrates a substantial improvement of 6.7%p. We observe similar robust improvements on Food101 (+1.9%p), Sun397 (+1.6%p), and RESISC45 (+1.8%p), confirming the strong zero-shot capability of our model.
