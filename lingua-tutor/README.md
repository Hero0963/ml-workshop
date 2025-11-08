# Lingua Tutor

An AI-powered language learning assistant designed to help users improve their language skills through transcription and evaluation.

## ✨ Features

- **Speech-to-Text (STT):** Utilizes a local Whisper model to transcribe user-spoken audio into text.
- **Transcription Evaluation:** Calculates the Word Error Rate (WER) by comparing the STT output against a ground-truth text, providing a standard metric for accuracy.
- **Dockerized Development Environment:** A fully containerized environment using Docker and Docker Compose ensures consistency, reproducibility, and easy setup. Includes GPU support for accelerated transcription.
- **Automated Workflow:** Scripts to automate the entire testing pipeline, from running STT to performing the evaluation.

## 📂 Project Structure

```
lingua-tutor/
├── .devcontainer/
│   └── Dockerfile.dev
├── output/
│   └── stt/
│       └── *.txt
├── src/
│   ├── stt/
│   │   ├── check_gpu.py
│   │   └── whisper_stt.py
│   ├── evaluation/
│   │   └── text_evaluator.py
│   └── utils.py
├── test_data/
│   ├── this_is_a_simple_test.txt
│   └── this_is_a_simple_test.wav
├── .dockerignore
├── docker-compose.dev.yml
├── pyproject.toml
├── pytest.ini
├── run_docker_dev.py
└── run_workflow.py
```

## 🚀 How to Use

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- NVIDIA GPU with appropriate drivers for CUDA support.

### 1. Start the Development Environment

From the project root directory, run the Python script to build and start the container in the background. The first build may take several minutes.

```shell
python run_docker_dev.py
```

### 2. Enter the Container

Once the container is running, open an interactive shell inside it.

```shell
docker compose -f docker-compose.dev.yml exec app bash
```

### 3. Run the Automated Workflow

Inside the container's shell, execute the main workflow script. This will run the hardcoded test case: transcribe `test_data/this_is_a_simple_test.wav` and evaluate it against `test_data/this_is_a_simple_test.txt`.

```shell
python run_workflow.py
```

The script will output the results of each step, culminating in the final Word Error Rate (WER) score.

## 🔮 Future Work

- **Interactive AI Agent:** Introduce an LLM-based agent to provide interactive feedback, generate questions, and offer supplementary learning materials based on the user's performance.
- **Advanced Pronunciation Evaluation:** Move beyond WER to a more granular, phoneme-level analysis for more precise pronunciation scoring and feedback.
- **Service-Oriented Architecture:** Refactor the project into a service with a dedicated UI (e.g., a web interface) and an API for broader accessibility.
- **Learning Material Sourcing:** Investigate legally sound methods for sourcing and managing learning materials, as the originally planned web crawler has potential legal and ethical implications that require careful consideration.