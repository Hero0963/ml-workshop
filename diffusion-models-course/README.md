# Diffusion Models Course

A hands-on course on diffusion models, from first principles to the 2026 research frontier.
Every idea is derived, implemented from scratch in PyTorch, and checked in a lab notebook that
runs on a laptop CPU in minutes.

The lessons are written in Traditional Chinese (each one explains the intuition in plain
language first, then gives the formal derivation); the code and its comments are in English.
Start with [`lessons/00_course_map.md`](lessons/00_course_map.md).

## Syllabus

| # | Lesson | Lab notebook |
|---|--------|--------------|
| 00 | [Course map and setup](lessons/00_course_map.md) | — |
| 01 | [Generative models and the idea behind diffusion](lessons/01_generative_models_and_intuition.md) | — |
| 02 | [Math toolbox: Gaussians, scores, Langevin dynamics](lessons/02_math_toolbox.md) | [`02_langevin.ipynb`](notebooks/02_langevin.ipynb) |
| 03 | [The forward process: turning data into noise](lessons/03_forward_process.md) | [`03_forward_process.ipynb`](notebooks/03_forward_process.ipynb) |
| 04 | [DDPM: the reverse process, training and sampling](lessons/04_ddpm_training_and_sampling.md) | [`04_ddpm_2d.ipynb`](notebooks/04_ddpm_2d.ipynb) |
| 05 | [Scores and SDEs: the same model, seen differently](lessons/05_score_and_sde.md) | [`05_score_field.ipynb`](notebooks/05_score_field.ipynb) |
| 06 | [Image diffusion with a U-Net](lessons/06_unet_image_diffusion.md) | [`06_mnist_ddpm.ipynb`](notebooks/06_mnist_ddpm.ipynb) |
| 07 | [Fast sampling: DDIM and ODE solvers](lessons/07_fast_sampling_ddim.md) | [`07_ddim_sampling.ipynb`](notebooks/07_ddim_sampling.ipynb) |
| 08 | [Conditioning and classifier-free guidance](lessons/08_guidance.md) | [`08_classifier_free_guidance.ipynb`](notebooks/08_classifier_free_guidance.ipynb) |
| 09 | [Flow matching and rectified flow](lessons/09_flow_matching.md) | [`09_flow_matching.ipynb`](notebooks/09_flow_matching.ipynb) |
| 10 | [Latent diffusion, diffusion transformers, text conditioning](lessons/10_latent_diffusion_and_dit.md) | [`10_tiny_dit.ipynb`](notebooks/10_tiny_dit.ipynb) |
| 11 | [The 2026 frontier: a map for further reading](lessons/11_frontier_2026.md) | — |
| A | [Notation and formula cheat sheet](lessons/A_notation.md) | — |

Curated reading list with verification dates: [`references.md`](references.md).

## Setup

The project is self-contained, like every project in this repository. Dependencies are added
by hand (repository rule), and `pyproject.toml` already routes `torch` / `torchvision` to the
PyTorch CUDA 12.6 wheel index, so the same commands give a GPU build on Windows and Linux:

```bash
cd diffusion-models-course
uv add torch torchvision numpy matplotlib loguru ipykernel
uv add --dev pytest
uv run pytest
```

Then open any notebook in `notebooks/` with the `.venv` of this project as the kernel.
MNIST is downloaded to `data/` on first use (about 60 MB, ignored by git).

**Hardware.** Every lab runs on a CPU. Measured on a 4-core cloud CPU: the labs without image
training finish in under 4 minutes, the MNIST labs in 7-30 minutes (lab 06 is the longest: about
16 minutes of training plus 1000-step sampling). With a GPU, raise the step counts at the top of
each notebook, or use the scripts for longer runs:

```bash
uv run python scripts/train_mnist.py --method ddpm --arch unet --steps 20000
uv run python scripts/train_mnist.py --method flow --arch dit --conditional --steps 20000
uv run python scripts/sample_mnist.py checkpoints/mnist_flow_dit_cond.pt --guidance-scale 3
```

## Project structure

```
diffusion-models-course/
├── lessons/                 # the course text (Traditional Chinese)
├── notebooks/               # executed lab notebooks, one per lesson with a lab
├── src/diffusion_course/    # reference implementation used by every lab
│   ├── schedules.py         #   noise schedules (linear, cosine)
│   ├── ddpm.py              #   forward process, epsilon loss, ancestral sampling
│   ├── ddim.py              #   deterministic few-step sampling
│   ├── score.py             #   score functions, (annealed) Langevin dynamics
│   ├── flow_matching.py     #   flow matching loss, Euler / Heun samplers
│   ├── guidance.py          #   classifier-free guidance
│   ├── models/              #   toy MLP, U-Net, tiny DiT -- all called as model(x, t, y)
│   ├── data.py              #   2D toy data, Gaussian mixture with exact score, MNIST
│   ├── training.py          #   one training loop with EMA, checkpoints
│   └── viz.py               #   plotting helpers
├── scripts/                 # longer training runs and sampling from checkpoints
├── tests/                   # checks against closed-form answers (pytest)
├── references.md            # reading list
├── NOTICE.md                # third-party licences and attribution
└── ai-collab/               # roadmap, dev log, project guide
```

## Licensing and attribution

All code and text in this project were written for it; no code was copied from the papers'
reference implementations (several of which are licensed for non-commercial use only). The
datasets are downloaded at run time and not redistributed, but the executed notebooks display
some MNIST digits. See [`NOTICE.md`](NOTICE.md) for the dataset licences and the attribution they
require.
