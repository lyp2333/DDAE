# DDAE: Dual-Flow Adaptive Diffusion Autoencoder

Official implementation of **"DDAE: Dual-Flow Adaptive Diffusion Autoencoder for High-Fidelity Image Reconstruction and Editing"**.

DDAE is a diffusion-based autoencoder that couples a compact semantic encoder with a **dual-flow decoder**: instead of predicting only the noise `ε`, the decoder jointly estimates the clean image `x₀` and the noise through two prediction heads fused by a **learned, timestep-conditioned gating factor `γ(t)`**. Both targets share a single U-Net trunk (key/value reuse), which enforces trajectory consistency without any extra consistency loss, and enables single-forward-pass dual-target prediction at near single-decoder cost.

> Method details, full training configuration, and extended experiments are documented in [`Supplementary Material for DDAE.pdf`](./Supplementary%20Material%20for%20DDAE.pdf) and [`The extended version of DDAE.pdf`](./The%20extended%20version%20of%20DDAE.pdf) in this repository.

## 1. Overview & Core Features

**Method highlights**

- **Dual-flow prediction** (`pred_mode: adaptive_form1`): the decoder outputs both `x₀` and `ε` per timestep, blended by a learned gating variable `γ_t` that adapts to the noise level.
- **Shared-trunk architecture**: target-specific parameters are confined to the last decoder blocks (`use_ada_model` / `adaptive_model_type` / `num_adaptive_blocks`), while key/value activations are reused across flows for implicit consistency.
- **Joint training** of the semantic encoder and the diffusion decoder (`train_pattern: encoder+ddpm`), with EMA target network, gradient clipping, and optional `torch.compile` acceleration.

**Features provided by this repository**

| Category | What you get |
|---|---|
| Training | Hydra-configured PyTorch Lightning training on iLab-20M / FFHQ / CelebA-HQ (plus fonts / dSprites templates), fp16/bf16 mixed precision, single- or multi-GPU (DDP), optional DeepSpeed ZeRO offload |
| Sampling | DDIM / DDPM / DPM-Solver backends, with `random-x_T` or **inferred-`x_T`** (reverse-encoded) reconstruction modes |
| Evaluation | Reconstruction (SSIM / PSNR / LPIPS), FID, Perceptual Path Length (PPL), latent interpolation, latent recombination, attribute manipulation, disentanglement consistency matrix |
| Baselines | Group-Supervised Learning (GSL) comparison scripts and GAE latent-prior utilities |

## 2. Requirements & Dependencies

**Hardware & OS**

- NVIDIA GPU with CUDA (a GPU is **required** — CPU-only training/evaluation is not supported). Multi-GPU is optional.
- Linux (tested on Ubuntu). The CUDA extension in `third_party/` additionally requires a CUDA toolkit (`nvcc`) matching your PyTorch build — only needed for PPL evaluation.

**Software**

- Python ≥ 3.9 (3.10 recommended)
- PyTorch ≥ 2.0 (CUDA build; `torch.compile` is used when enabled)

**Python packages**

```bash
pip install torch torchvision pytorch-lightning>=2.0 hydra-core omegaconf \
            lmdb lpips diffusers pytorch-fid clean-fidelity \
            scipy numpy matplotlib pillow tqdm requests tensorboard
# optional: DeepSpeed ZeRO-3 offload
pip install deepspeed
```

`clean-fidelity` provides the `fidelity` CLI used by `scripts/fid.sh`; `pytorch-fid` backs `main/evaluation/evaluate_fid.py`.

## 3. Installation & Configuration

### 3.1 Clone and set up the environment

```bash
git clone https://github.com/lyp2333/DDAE.git
cd DDAE
conda create -n ddae python=3.10 -y
conda activate ddae
pip install <packages above>
```

### 3.2 Prepare datasets (LMDB format)

All datasets are consumed as **LMDB** databases of images. To convert an image folder into LMDB:

```bash
python data_resize_celeba.py
```

Edit the `__main__` block first: set `in_path` (source image folder), `out_path` (target `.lmdb`), `ext`, and `size`. Datasets used in the paper:

- **iLab-20M** (vehicles) — download from the official iLab-20M release, then convert to `ilab_128.lmdb` (128×128).
- **FFHQ / CelebA-HQ** (faces) — download from their official sources and convert the same way.

### 3.3 Point the configs to your machine

Everything is driven by Hydra YAML files under `main/configs/`. Before the first run, edit the corresponding `dataset/<name>/train.yaml` and `test.yaml`:

| Key | Meaning | Example |
|---|---|---|
| `joint.data.root` | Path to the LMDB dataset | `/data/ilab/ilab_128.lmdb` |
| `joint.training.results_dir` | Output root (checkpoints, logs) | `/runs/ddae` |
| `joint.training.workers` | DataLoader workers (default 72 — tune to your CPU) | `16` |
| `joint.training.Gae_chkpt_path` / `diffae_chkpt_path` | Pretrained checkpoints (only needed for GAE-related patterns) | — |
| `joint.evaluation.ckpt_path` | Checkpoint used by evaluation scripts (in `test.yaml`) | `/runs/ddae/.../last.ckpt` |
| `joint.evaluation.save_path` / `recon_save_path` | Output dirs for evaluation results | — |

⚠️ **Crucial**: several Python files contain **hard-coded absolute `sys.path` entries** from the original development machine. Search and replace these with your local repo path before running:

```bash
grep -rn "/home/lyp/Code" --include="*.py" .     # find all occurrences
# main/trainer/joint_adaptive_trainer.py, main/evaluation/*.py, test.py, ...
```

The shell scripts in `scripts/` likewise contain absolute paths (`/home/lyp/...`) — adapt them to your layout or call the Python commands directly as shown below.

## 4. Usage & Examples

All entry points use Hydra, so any config value can be overridden on the command line.

### 4.1 Training

```bash
# DDAE on iLab-20M (default dataset config)
python main.py dataset=ilab_20M/train

# common overrides
python main.py dataset=ilab_20M/train \
    dataset.joint.training.batch_size=64 \
    dataset.joint.training.fp16=bf16-mixed \
    dataset.joint.training.device=gpu:0,1
```

Key training-time switches (see `main/configs/dataset/ilab_20M/train.yaml`):

| Config | Options | Note |
|---|---|---|
| `joint.training.train_pattern` | `encoder+ddpm` (DDAE), `encoder+GAE+ddpm` | The latter requires a pretrained GAE checkpoint |
| `joint.model_ddpm.pred_mode` | `adaptive_form1` (DDAE), `eps`, `x0`, `adaptive_form2` | Prediction parameterization |
| `joint.model_ddpm.use_ada_model` + `adaptive_model_type` + `num_adaptive_blocks` | `last` / `symmetric`, block count | Placement of target-specific parameters (e.g. last 3 decoder blocks) |
| `joint.training.fp16` | `16-mixed`, `bf16-mixed`, `False` | Mixed-precision mode |
| `joint.training.use_torch_compile` | `True` / `False` | Requires PyTorch ≥ 2.0 |
| `joint.training.lambda_eps / lambda_x0 / lambda_adaptive_factor` | float | Loss weights (paper: all 1.0; LPIPS weight 2.0 is hard-coded in the loss) |

**Outputs**: checkpoints and logs are written to
`<results_dir>/checkpoints/<training_name>--checkpoints/` — `Model_weights/` (best + `last.ckpt`) and `Tensorboard_logs/`. Training **auto-resumes** from `last.ckpt` if present in the same directory; change `training_name` for a fresh run. Monitor with `tensorboard --logdir <results_dir>`.

### 4.2 Reconstruction

```bash
# via wrapper script (args: recon_model, recon_mode, pred_steps)
bash scripts/recon.sh encoder_out step 100

# or directly
python main/evaluation/generate_recons.py dataset=ilab_20M/test \
    dataset.joint.evaluation.recon_model=encoder_out \
    dataset.joint.evaluation.recon_mode=step \
    dataset.joint.evaluation.pred_steps=100
```

- `recon_mode=step` — **inferred-`x_T`**: DDIM-reverse-encode the input (`T_encode_reversed` steps), then denoise (`pred_steps` steps).
- `recon_mode=random` — **random-`x_T`**: denoise from pure noise, conditioned on the semantic code (`use_xT=False`).

### 4.3 Quantitative evaluation

```bash
# SSIM / PSNR / LPIPS between two image folders (original vs. recon)
python main/evaluation/autoencoding_evaluation.py dataset=ilab_20M/test \
    dataset.joint.data.autoencoding_imgs_root0=/path/to/original \
    dataset.joint.data.autoencoding_imgs_root1=/path/to/recon

# FID (pytorch-fid based)
python main/evaluation/evaluate_fid.py   # or adapt scripts/fid.sh (uses the `fidelity` CLI)

# Perceptual Path Length — needs third_party CUDA ops and a StyleGAN VGG16 .pkl
bash scripts/ppl_evaluation.sh encoder_out full
```

### 4.4 Latent-space editing & analysis

```bash
# latent interpolation between image pairs (set imgs_root to a folder of l.png/r.png pairs)
python main/evaluation/interpolation.py dataset=ilab_20M/test \
    dataset.joint.data.imgs_root=/path/to/imgs_interpolation \
    dataset.joint.evaluation.num_intp=10

# latent recombination (swap pose / background / identity across images)
python main/evaluation/recombine.py dataset=ilab_20M/test
# example input groups are provided under main/evaluation/imgs_recombination/

# attribute manipulation & disentanglement consistency matrix
python main/evaluation/manipulate.py dataset=ilab_20M/test
python main/evaluation/calculate_cmatrix.py dataset=ilab_20M/test
```

### 4.5 Auxiliary utilities

```bash
python -m main.trainer.train_ae                 # train the GAE latent prior
python -m main.trainer.train_classifier         # train attribute classifiers
python -m main.trainer.get_and_save_latent_code # dump semantic latents to disk
python main/evaluation/visualization.py         # t-SNE of the semantic space
python main/evaluation/plots/figure_ppl_distance.py  # plot PPL-vs-timesteps curves
```

GSL baseline comparisons live in `main/evaluation/GSL/` (reconstruction, recombination, interpolation with a pretrained Group-Supervised Learning checkpoint).

## 5. Project Structure

```text
DDAE/
├── main.py                     # DDAE training entry point
├── test.py                     # developer scratchpad (not an entry point)
├── data_resize_celeba.py       # image folder → LMDB converter
├── scripts/                    # ready-made shell wrappers (paths need adapting)
│   ├── recon.sh                #   reconstruction
│   ├── fid.sh                  #   FID via `fidelity` CLI
│   ├── ppl_evaluation.sh       #   PPL
│   ├── autoencoding_eval.sh    #   SSIM/PSNR/LPIPS
│   └── test_interpolation.sh   #   interpolation
├── main/
│   ├── configs/                # Hydra configs
│   │   ├── default_conf.yaml   #   default dataset group = ilab_20M/train
│   │   └── dataset/            #   per-dataset train/test yaml (ilab_20M, ffhq,
│   │                           #   celeba, celebahq256, fonts, dsprites)
│   ├── datasets/               # LMDB dataset classes + eval helpers
│   ├── models/
│   │   ├── joint_dual_model.py # ModelWrapperV2 — DDAE Lightning module (dual-flow loss)
│   │   ├── joint_model.py      # single-flow variant
│   │   ├── autoencoders/GAE.py # GAE latent prior
│   │   ├── gsl_net/            # GSL baseline network
│   │   ├── classifier.py       # attribute classifiers
│   │   ├── callbacks.py        # EMA update, training-time sampling
│   │   └── diffusion/          # DDPM, DDIM, DPM-Solver, U-Net variants, Encoder
│   ├── trainer/
│   │   ├── joint_adaptive_trainer.py  # main DDAE trainer (used by main.py)
│   │   ├── train_ae.py                # GAE trainer
│   │   ├── train_classifier.py        # classifier trainer
│   │   ├── get_and_save_latent_code.py
│   │   └── metric.py                  # SSIM / PSNR
│   └── evaluation/             # all eval scripts (see Section 4)
│       ├── GSL/                #   Group-Supervised Learning comparisons
│       ├── imgs_recombination/ #   example inputs for recombination
│       └── plots/              #   figure scripts (PPL curve)
└── third_party/                # NVlabs StyleGAN2-ADA ops (PPL evaluation)
    ├── dnnlib/
    └── torch_utils/            # CUDA extensions — compiled on first use
```

## 6. Important Notes

1. **Hard-coded paths**: as noted in §3.3, `sys.path` entries in `main/trainer/joint_adaptive_trainer.py`, `main/evaluation/*.py`, and `test.py`, plus all absolute paths in `scripts/*.sh` and the YAML configs, must be adapted to your machine before running. This is the single most common source of `ModuleNotFoundError` / `FileNotFoundError` for new users.
2. **Config provenance**: `main/configs/dataset/ilab_20M/train.yaml` is the reference configuration matching the paper (128×128, Adam, lr 6×10⁻⁵, batch 32, mixed precision, `adaptive_form1`, 3 adaptive blocks). The `ffhq/` and `celebahq256/` templates additionally contain 256-resolution development settings — if you train at 128, keep `image_size` and the corresponding model keys consistent with the paper setup.
3. **`test.py` is not a test suite** — it is a developer scratchpad; ignore it.
4. **GPU is mandatory** (`device: "gpu:0"`; use `gpu:0,1` for DDP). PPL evaluation additionally JIT-compiles the CUDA ops in `third_party/` on first run and downloads/loads a StyleGAN VGG16 feature network (`vgg16_path` in the eval config).
5. **Auto-resume**: training resumes from `last.ckpt` inside the run directory whenever it exists. To start over, change `training_name` or move the old checkpoint.
6. **Checkpoint compatibility**: evaluation scripts load `ModelWrapperV2` with `strict=False`; when switching `pred_mode` / adaptive-block settings between train and eval, make sure the YAML model section matches the checkpoint, or loading will silently skip mismatched weights.
7. **Batch size vs. dataset size**: the loader caps `batch_size` at `len(dataset)`; tiny debug datasets are fine but reduce `workers` accordingly.

## Citation

If you find this code useful, please cite the paper (bibtex will be added upon publication) and refer to the supplementary material in this repository for full experimental details.

## License

The code is released for research purposes. Contact the authors before any commercial use.
