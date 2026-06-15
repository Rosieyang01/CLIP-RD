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

The training scripts are configured for image-text pretraining data such as CC3M and CC12M. Each dataset should be prepared with a CSV annotation file and an image directory.

The CSV file should contain image paths and captions. By default, the scripts use:

* `filepath`: image file path column
* `title`: text caption column

Example structure:

```text
data/
├── cc3m/
│   ├── cc3m_train.csv
│   ├── cc3m_val.csv
│   └── images/
├── cc12m/
│   ├── cc12m.csv
│   └── images/
└── imagenet/
    └── val/
```

For retrieval evaluation, MSCOCO and Flickr should be preprocessed using the Karpathy split. For zero-shot classification, ImageNet validation data and related variants such as ImageNet-V2, ImageNet-R, and ImageNet-Sketch can be used.

Before running the scripts, replace all `path/to/...` entries with the actual local paths to the datasets, teacher checkpoint, model checkpoint, and log directory.

## Result

CLIP-RD aims to improve the student CLIP model by preserving relational knowledge from the teacher model in multiple directions. Compared with the baseline and conventional KD setting, CLIP-RD additionally considers vertical and cross relational distillation, allowing the student model to better inherit the teacher's image-text representation structure.

The experimental results can be summarized as follows:

| Method   | Training Setting                 | Evaluation Dataset         | Metric     | Result        |
| -------- | -------------------------------- | -------------------------- | ---------- | ------------- |
| Baseline | Student only                     | MSCOCO / Flickr / ImageNet | R@K / Acc. | To be updated |
| KD       | Feature + relational KD baseline | MSCOCO / Flickr / ImageNet | R@K / Acc. | To be updated |
| CLIP-RD  | HRD + VRD + XRD                  | MSCOCO / Flickr / ImageNet | R@K / Acc. | To be updated |

To reproduce CLIP-RD training, run:

```bash
bash script/ViT_B_16_Laion400M/ViT_T_16_RD.sh
```

To evaluate a trained checkpoint, update the `--resume` path in the evaluation script and run:

```bash
bash script/eval/eval_coco.sh
bash script/eval/eval_flickr.sh
bash script/eval/eval_imagenet.sh
```

