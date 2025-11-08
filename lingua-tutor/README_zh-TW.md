# Lingua Tutor (語言家教)

一個 AI 驅動的語言學習助理，旨在透過語音轉文字及評估，幫助使用者提升他們的語言能力。

## ✨ 功能特色

- **語音轉文字 (STT):** 使用本地端的 Whisper 模型，將使用者的語音音檔轉換為文字。
- **文本評估:** 透過比對 STT 結果與標準答案文本，計算「詞誤率 (Word Error Rate, WER)」，為準確度提供一個標準化的指標。
- **Docker 化的開發環境:** 使用 Docker 與 Docker Compose 建立的完整容器化環境，確保了環境的一致性、可重現性與設定的簡易性。內建 GPU 支援以加速語音辨識。
- **自動化工作流:** 提供腳本以自動執行完整的測試流程，從執行 STT 到完成評估。

## 📂 專案結構

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

## 🚀 如何使用

### 環境需求

- [Docker](https://www.docker.com/get-started)
- 支援 CUDA 的 NVIDIA GPU 及對應的驅動程式。

### 1. 啟動開發環境

在專案根目錄下，執行 Python 腳本來建立並在背景啟動容器。第一次建立可能需要數分鐘。

```shell
python run_docker_dev.py
```

### 2. 進入容器

容器開始運作後，開啟一個與容器互動的 shell 介面。

```shell
docker compose -f docker-compose.dev.yml exec app bash
```

### 3. 執行自動化工作流

在容器的 shell 中，執行主要的工作流腳本。它將會執行我們寫死的測試案例：辨識 `test_data/this_is_a_simple_test.wav`，並與 `test_data/this_is_a_simple_test.txt` 進行比對。

```shell
python run_workflow.py
```

腳本將會輸出每個步驟的結果，並在最後給出「詞誤率 (WER)」的分數。

## 🔮 未來工作

- **互動式 AI 代理人:** 引入一個基於大型語言模型 (LLM) 的代理人，根據使用者的表現提供互動式回饋、生成練習題目，並提供補充學習資料。
- **進階發音評估:** 從目前的詞誤率 (WER) 升級到更細緻的音素級別 (phoneme-level) 分析，以提供更精準的發音評分與回饋。
- **服務化架構:** 將專案重構成一個服務，並提供專門的使用者介面 (例如網頁) 和 API，以擴大其應用範圍。
- **學習材料來源:** 針對原先規劃的爬蟲功能，需謹慎研究其潛在的法律及倫理問題，並探索合法合規的學習材料獲取與管理方法。