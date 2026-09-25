# NOTICE — third-party material and attribution

Checked on 2026-09-25. This file lists everything in this project that comes from someone else,
under which terms, and how it is used.

## What is original

All code in `src/`, `scripts/`, `tests/` and `notebooks/`, all lesson text in `lessons/`, and all
figures in the executed notebooks were written or generated for this course.

- The algorithms are implemented from the equations in the published papers cited in each lesson
  and in `references.md`. **No code was copied** from the papers' reference implementations.
  Several of those implementations carry non-commercial licences (for example
  [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) is CC BY-NC);
  check each repository's own `LICENSE` before reusing its code.
- No figures from papers, blog posts or videos are embedded; every plot is produced by the lab
  notebooks. External material is only linked.
- Mathematical notation follows the conventions of the cited papers; explanations are our own
  wording.

## Datasets (downloaded at run time, not redistributed)

Neither dataset is stored in this repository (`data/` is ignored by git); `torchvision` downloads
it on first use. The executed notebooks do display some images derived from them, so the
attribution below applies to those notebook outputs.

### MNIST

- Creators: Yann LeCun, Corinna Cortes, Christopher J.C. Burges.
  Reference: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning applied to
  document recognition", *Proceedings of the IEEE* 86(11), 1998.
- Licence: the MNIST page is widely quoted as releasing the dataset under
  **Creative Commons Attribution-Share Alike 3.0** (CC BY-SA 3.0), with the copyright held by
  Yann LeCun and Corinna Cortes.
  - Verification status: the official page (<https://yann.lecun.com/exdb/mnist/>) could not be
    reached on 2026-09-25 (HTTP 503), so this is from secondary sources. The Hugging Face card
    for `ylecun/mnist` lists MIT instead. We follow the **stricter** reading, CC BY-SA 3.0.
- Use here: training data for the lab models. The MNIST digits and their noised versions shown
  in `notebooks/03_forward_process.ipynb` and the other lab outputs are adapted from MNIST and
  are shared under CC BY-SA 3.0 with the attribution above.

### Fashion-MNIST (optional)

- Copyright (c) 2017 Zalando SE, released under the **MIT License**
  (<https://github.com/zalandoresearch/fashion-mnist>).
- Use here: an optional drop-in replacement (`image_loader("fashion_mnist")`) suggested in the
  exercises; none of the committed notebook outputs use it.

## Software dependencies (installed by the user, not redistributed)

| Package | Licence (PyPI metadata, 2026-09-25) |
|---|---|
| PyTorch (`torch`) | BSD-style, plus the licences of bundled components (Apache-2.0, MIT, ...) |
| `torchvision` | BSD |
| NumPy | BSD-3-Clause (plus bundled components) |
| Matplotlib | Matplotlib licence (PSF-based) |
| `loguru` | MIT |
| `ipykernel` | BSD-3-Clause |
| `pytest` | MIT |
