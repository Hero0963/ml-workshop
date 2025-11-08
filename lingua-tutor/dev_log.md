# Dev Log - Lingua Tutor

## 2025-11-08

- **Implemented Core STT Module (`src/stt`):**
  - Integrated `openai-whisper` for local speech-to-text transcription.
  - Developed a script (`check_gpu.py`) to validate GPU availability, ensuring PyTorch can leverage CUDA.
  - Created a runnable script (`whisper_stt.py`) that processes an audio file and saves the transcription, featuring a timestamped output file for versioning.

- **Developed Text Evaluation Module (`src/evaluation`):**
  - Implemented a function to calculate Word Error Rate (WER) based on Levenshtein distance, establishing a standard metric for accuracy.
  - Created a runnable script (`text_evaluator.py`) to compare a ground-truth text with an STT hypothesis file.
  - Refactored both STT and evaluation scripts to use a hardcoded test case, creating a streamlined and repeatable testing workflow.

- **Containerized the Development Environment:**
  - Set up a complete Docker environment using `docker-compose` for consistency and reproducibility.
  - Authored a `Dockerfile.dev` using a clean `nvidia/cuda` base image to ensure a conflict-free Python and PyTorch installation.
  - Implemented a Python-based launcher (`run_docker_dev.py`) to simplify starting the development container.
  - Added a `run_workflow.py` script to automate the end-to-end testing process (GPU check -> STT -> Evaluation) inside the container.
- **Note on Crawler Feature:** The originally planned web crawler feature has potential legal and ethical implications that require careful consideration before implementation.

## 2025-11-05

- Initialized project structure.
- Created READMEs and initial documentation.