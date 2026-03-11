# CLAUDE.md — V-JEPA 2 Codebase Guide

This file documents the V-JEPA 2 repository for AI assistants. It covers project structure, development workflows, code conventions, and key architectural patterns.

## Project Overview

**V-JEPA 2** is Meta FAIR's PyTorch implementation of a self-supervised video understanding and prediction model. It learns visual representations from internet-scale video data without labels, achieving state-of-the-art results on video understanding benchmarks (SSv2, Diving48, EK100) and supporting robot manipulation through an action-conditioned variant (V-JEPA 2-AC).

- **Package name:** `vjepa2`, version `0.0.1`
- **Python requirement:** >= 3.11 (CI uses 3.12)
- **License:** MIT (some utility files under Apache 2.0)

---

## Repository Structure

```
jepa/
├── app/                        # Training loop implementations
│   ├── vjepa/                  # Standard video JEPA pre-training
│   ├── vjepa_droid/            # Action-conditioned (robot) post-training
│   ├── vjepa_cowa/             # COWA variant
│   ├── vjepa_cowa_planner/     # COWA planner variant
│   ├── main.py                 # Local training entrypoint
│   ├── main_distributed.py     # SLURM distributed training entrypoint
│   └── scaffold.py             # Dynamic app loader via importlib
├── src/                        # Core library package
│   ├── models/                 # Neural network architectures
│   │   ├── vision_transformer.py       # Main ViT encoder (3D video, RoPE)
│   │   ├── predictor.py                # Masked latent predictor
│   │   ├── ac_predictor.py             # Action-conditioned predictor
│   │   ├── attentive_pooler.py         # Pooling/classification heads
│   │   └── utils/
│   │       ├── modules.py              # Transformer blocks
│   │       ├── patch_embed.py          # 2D/3D patch embedding
│   │       └── pos_embs.py             # Sincos and RoPE positional embeddings
│   ├── datasets/               # Data loading and preprocessing
│   │   ├── video_dataset.py            # Video loader (uses decord)
│   │   ├── data_manager.py             # Multi-dataset management and weighted sampling
│   │   ├── imagenet1k.py               # ImageNet support
│   │   └── utils/                      # Transforms, samplers, dataloader utilities
│   ├── masks/                  # Masking strategies
│   │   ├── multiseq_multiblock3d.py    # Spatial-temporal block masking
│   │   └── default.py                  # Default masking utilities
│   ├── hub/
│   │   └── backbones.py                # PyTorch Hub loaders (ViT-L/H/G)
│   └── utils/                  # Shared utilities
│       ├── distributed.py              # Distributed training initialization
│       ├── logging.py                  # Logging and CSV monitoring
│       ├── monitoring.py               # GPU memory/process monitoring
│       ├── schedulers.py               # LR and weight decay schedulers
│       └── checkpoint_loader.py        # Checkpoint management
├── evals/                      # Evaluation frameworks
│   ├── video_classification_frozen/    # SSv2, Diving48 classification
│   ├── action_anticipation_frozen/     # EPIC-KITCHENS action anticipation
│   ├── image_classification_frozen/    # ImageNet probes
│   ├── main.py                 # Local eval entrypoint
│   └── main_distributed.py     # Distributed eval entrypoint
├── configs/                    # YAML configuration files
│   ├── train/                  # Pre-training and post-training configs
│   ├── eval/                   # Evaluation configs
│   └── inference/              # Inference configs
├── tests/                      # Unit tests (9 files)
│   ├── models/                 # ViT, predictor tests
│   └── datasets/               # Dataloader, transform, sampler tests
├── notebooks/                  # Demo notebooks and example scripts
├── .github/workflows/          # CI/CD (unit tests + linters)
├── setup.py
├── pyproject.toml
├── requirements.txt
└── requirements-test.txt
```

---

## Development Commands

### Installation

```bash
# Create conda environment (recommended)
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312

# Install package in editable mode
pip install -e .

# Install linting tools for development
pip install -r requirements-test.txt
```

> **Note:** `decord` (the video reading library) does not support macOS natively. Use `eva-decord` or `decord2` as alternatives on macOS.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/models/test_vision_transformer.py

# Run a specific test
pytest tests/models/test_vision_transformer.py::TestVisionTransformer::test_forward
```

Most model tests require CUDA. Tests will be skipped automatically if no GPU is available.

### Linting and Formatting

Before committing, run all three linters in this order. CI enforces all three:

```bash
# Fix import ordering
python -m isort app evals/*.py src tests

# Format code (119-char line length)
python -m black app evals/*.py src tests

# Check for style/logic issues
python -m flake8 --config .flake8 --show-source --statistics app evals/*.py src tests
```

To check without modifying (as CI does):
```bash
python -m isort app evals/*.py src tests --check
python -m black --check app evals/*.py src tests
python -m flake8 --config .flake8 --show-source --statistics app evals/*.py src tests
```

### Training

**Local (single node, specify GPU devices):**
```bash
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml --devices cuda:0 cuda:1
```

**Distributed (SLURM via submitit):**
```bash
python -m app.main_distributed --fname configs/train/vitl16/pretrain-256px-16f.yaml --time 6000 --account <slurm_account>
```

The `--fname` argument points to a YAML config. The `app` field in the YAML selects which training module to use (e.g., `vjepa`, `vjepa_droid`). `app/scaffold.py` dynamically imports `app.<app_name>.train.main()`.

### Evaluation

```bash
# Local evaluation
python -m evals.main --fname configs/eval/vitl16/ssv2.yaml --devices cuda:0

# Distributed evaluation
python -m evals.main_distributed --fname configs/eval/vitl16/ssv2.yaml
```

### Using Pre-trained Models via PyTorch Hub

```python
import torch

# Load preprocessor and model
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
```

Available hub entry points are defined in `src/hub/backbones.py`.

---

## Code Conventions

### Style Rules

- **Indentation:** 4 spaces (no tabs)
- **Line length:** 119 characters maximum (enforced by black and flake8)
- **Formatting:** PEP8, enforced via `black` (profile: black) and `flake8`
- **Import sorting:** `isort` with `profile = "black"`, `line_length = 119`
- **Ignored flake8 rules:** E203 (whitespace before ':'), E701 (multiple statements), W503 (line break before binary operator)
- **`__init__.py` and `version.py`:** F401 (unused imports) is ignored

### File Headers

All source files carry a Meta FAIR copyright header:
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
```

Always include this header when creating new source files in `app/`, `src/`, `evals/`, or `tests/`.

### Module Patterns

- All neural network components subclass `torch.nn.Module`
- Constructors use keyword arguments with defaults; `**kwargs` accepted where extensibility is needed
- Type annotations in function signatures are encouraged
- Logging uses `logging.getLogger()`. Only rank-0 process logs at INFO level; other ranks log at ERROR level to avoid redundant output
- Configuration is YAML-driven; avoid hardcoding hyperparameters in source files

### Distributed Training

- `src/utils/distributed.py` provides initialization helpers
- `submitit` handles SLURM job scheduling for distributed runs
- Training apps receive a device list from the entrypoint; do not assume single-GPU

### Key Libraries

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Core deep learning |
| `timm` | Vision Transformer building blocks |
| `einops` | Readable tensor reshaping |
| `decord` | Video frame extraction |
| `webdataset` | Web-scale dataset streaming |
| `submitit` | SLURM job scheduling |
| `wandb`, `tensorboard` | Experiment tracking |
| `fire` | CLI argument parsing |
| `pyyaml`, `python-box` | YAML config loading |
| `beartype` | Runtime type checking |
| `peft`, `transformers` | Fine-tuning utilities |

---

## Configuration Files

YAML configs under `configs/` drive all training and evaluation runs. Key top-level fields:

- `app`: Training app name (maps to `app/<name>/train.py`)
- `meta`: Output directories, checkpoint frequency, logging settings
- `data`: Dataset paths, sampling weights, number of workers
- `mask`: Masking strategy and parameters
- `model`: Architecture (ViT size, patch size, etc.)
- `optimization`: Learning rate, weight decay, schedulers, batch size, epochs

When adding new configs, follow the naming pattern `configs/train/<model>/description.yaml`.

---

## Testing Guidelines

- Add tests for new model components in `tests/models/`
- Add tests for new dataset utilities in `tests/datasets/`
- Model tests that require CUDA should guard with `pytest.mark.skipif(not torch.cuda.is_available(), ...)`
- Test files are named `test_<module>.py`
- CI runs `pytest tests` on every push — all tests must pass

---

## CI/CD

Two GitHub Actions workflows run on PRs and pushes:

1. **Unit Tests** (`.github/workflows/base_tests.yaml`): Runs `pytest tests` on every push using Python 3.12.
2. **Linters** (`.github/workflows/linters.yaml`): Runs `isort --check`, `flake8`, and `black --check` on changes to `app/`, `evals/*.py`, `src/`, or `tests/`. Triggered on pushes to `master` and PRs targeting `master` or `gh/**` branches.

Both workflows must pass before merging to `master`.

---

## Contributing

1. Fork the repo and create a feature branch from `master`
2. Add tests for new functionality
3. Update documentation if APIs change
4. Run linters and tests locally before pushing
5. Complete the [Meta CLA](https://code.facebook.com/cla) if not already done
6. Open a PR with reviewers assigned

Security issues should be reported via [Meta's bug bounty program](https://bugbounty.meta.com/), not as public GitHub issues.
