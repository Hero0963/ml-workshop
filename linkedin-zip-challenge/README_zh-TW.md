# Zip 謎題求解器挑戰

本專案致力於探索、開發並比較多種演算法，以解決「Zip」益智遊戲。專案包含了一整套的求解器、程序化謎題生成器，並以一個現代化的網頁服務提供互動介面。

## 關於「Zip」謎題

遊戲的目標是畫一條單一、連續的路徑，走過網格中每一個空格，且每個格子只能走一次。

### 規則
*   路徑必須覆蓋所有可走訪的儲存格，且每個儲存格只能造訪一次。
*   路徑必須按照數字順序（1 → 2 → 3 ...）連接所有的路徑點。
*   路徑不能穿過牆壁（以 `|` 或 `—` 標示）。

## 功能

*   **多樣化的使用者介面**:
    *   **Gradio UI**: 一個功能完整的介面，可用於求解謎題、生成新謎題，以及測試後端 API。
    *   **Svelte UI**: 一個進階的、實驗性的前端，提供功能更豐富、基於 Canvas 的所見即所得 (WYSIWYG) 謎題編輯器。
*   **RESTful API**: 由 FastAPI 驅動的後端，為求解器提供程式化的介面。
*   **多種求解演算法**: 提供從精確演算法到元啟發式演算法的多種求解器。
*   **程序化謎題生成**: 一個強大的腳本與 UI，可生成大量新的謎題資料集。
*   **豐富的視覺化**: 可生成解謎過程的詳細 GIF 動畫和靜態圖片。

## 專案結構

專案結構經過精心組織，將核心邏輯、API 應用和使用者介面明確分離。

```
linkedin-zip-challenge/
├── src/
│   ├── app/                # FastAPI 後端應用程式
│   │   ├── routers/        # API 端點定義
│   │   └── main.py         # 主要 FastAPI 應用定義與啟動
│   ├── core/               # 核心謎題邏輯與求解器
│   │   ├── puzzle_generation/ # 程序化謎題生成腳本
│   │   ├── solvers/        # 所有求解演算法實作
│   │   └── utils.py        # 共用工具 (解析、視覺化等)
│   ├── custom_components/  # 包含 Svelte UI 的原始碼
│   │   └── puzzle_editor/
│   │       └── frontend/   # Svelte 原始碼
│   ├── ui/                 # Gradio UI 應用程式
│   │   └── gradio_app.py   # Gradio 介面定義
│   └── settings.py         # 全域應用程式設定
├── .devcontainer/
│   ├── Dockerfile          # 用於 PRODUCTION 的多階段 Dockerfile
│   └── Dockerfile.dev      # 用於 DEVELOPMENT 的 Dockerfile
├── .env                    # 環境變數 (由使用者建立)
├── docker-compose.yml      # 用於 PRODUCTION 的 Docker Compose 檔案
├── docker-compose.dev.yml  # 用於 DEVELOPMENT 的 Docker Compose 檔案
├── run_docker_dev.py       # 用於啟動開發環境的自動化腳本
├── pyproject.toml          # `uv` 的專案依賴性設定
└── README.md               # 本檔案
```

## 開始使用

本專案支援兩種主要的工作流程：基於 Docker 的環境（因其易用性與一致性而推薦）和手動本地設定。

### 工作流程一：Docker 環境 (建議)

Docker 設定經過精心設計，以同時支援快速開發（具備熱重載）和生產級別的建置。

#### 用於開發 (具備熱重載)

這是**進行活躍開發時建議的工作流程**。它使用 `docker-compose.dev.yml` 來啟動兩個容器（後端和前端開發伺服器），並使用 `volumes` 掛載來啟用即時的程式碼變更反應。

**環境需求:**
*   [Docker](https://www.docker.com/get-started) & [Docker Compose](https://docs.docker.com/compose/install/)

**步驟:**

1.  **設定環境:**
    在專案根目錄下建立一個名為 `.env` 的檔案，並填入以下內容。此檔案用於設定埠號和服務 URL。
    ```.env
    # 主應用程式埠號
    APP_PORT=7440

    # Svelte 開發伺服器埠號 (用於熱重載)
    SVELTE_PORT=5173

    # 在 Docker 中，請使用服務名稱 'ollama_server'
    OLLAMA_PROVIDER_URL=http://ollama_server:11434/v1
    OLLAMA_MODEL_NAME=your_model_name_here
    ```

2.  **一鍵啟動:**
    執行專案提供的自動化腳本，它會為您處理所有事情。
    ```bash
    python run_docker_dev.py
    ```

3.  **存取服務:**
    *   **Gradio UI**: `http://localhost:7440/ui`
    *   **Svelte UI (具備熱重載)**: `http://localhost:5173`
    *   **Svelte UI (整合後)**: `http://localhost:7440/svelte-ui` (注意：此視圖僅在 Docker 重建後才會更新)。

    在開發前端時，請使用 `5173` 埠來即時查看您的變更。在開發後端時，對 Python 檔案的變更將會觸發伺服器自動重載。

#### 用於模擬生產環境

此工作流程會建置一個單一的、最佳化過的、自給自足的 Docker 映像，就像您在真實部署時所做的一樣。

**步驟:**

1.  **使用生產環境的 Compose 檔案:**
    在專案根目錄執行以下指令：
    ```bash
    # -f 指向生產環境的設定檔
    # --build 強制重新建置，執行多階段 Dockerfile
    docker compose -f docker-compose.yml up --build -d
    ```

2.  **存取服務:**
    所有服務都由同一個埠號提供。
    *   **Gradio UI**: `http://localhost:7440/ui`
    *   **Svelte UI**: `http://localhost:7440/svelte-ui`

### 工作流程二：手動本地設定

此方法不使用 Docker，直接在您的電腦上運行服務。

**環境需求:**
*   Python 3.11 & `uv`
*   Node.js & `npm`

**步驟:**

1.  **安裝 Python 依賴:**
    ```bash
    uv sync
    ```

2.  **建置前端:**
    Svelte UI 必須先被編譯成靜態檔案。
    ```bash
    cd src/custom_components/puzzle_editor/frontend
    npm install
    npm run build
    cd ../../../../  # 回到專案根目錄
    ```

3.  **設定環境:**
    在專案根目錄建立一個 `.env` 檔案。請注意 `OLLAMA_PROVIDER_URL` 與 Docker 設定不同。
    ```.env
    APP_PORT=7440

    # 在本地，請使用 localhost
    OLLAMA_PROVIDER_URL=http://localhost:11434/v1
    OLLAMA_MODEL_NAME=your_model_name_here
    ```

4.  **執行應用程式:**
    使用 `uv run` 在虛擬環境中執行應用程式。
    ```bash
    uv run python -m src.app.main
    ```

5.  **存取 UI 介面:**
    *   **Gradio UI**: `http://localhost:7440/ui`
    *   **Svelte UI**: `http://localhost:7440/svelte-ui`

## 如何使用操作介面

主要的介面是 Gradio UI，可透過 `/ui` 存取。它提供了數個分頁：

*   **Generate Puzzle (生成謎題)**: 建立一個新的、隨機的 6x6 謎題。您可以選擇障礙物的數量。生成的謎題佈局和牆壁可以被複製到其他求解器分頁中使用。
*   **Puzzle Solver (Naive)**: 以文字形式貼上謎題佈局和牆壁來進行求解。
*   **Puzzle Solver (Interactive)**: 一個強大的所見即所得編輯器，可以透過點擊網格、新增路徑點、障礙物和牆壁來建立或編輯謎題。
*   **Echo Test**: 一個簡單的工具，用以確認後端 API 是否正常回應。

## 未來工作

以下領域已規劃為未來的開發方向，目前尚未整合進主應用程式：

*   **強化學習 (RL) 求解器**: `src/core/rl/` 下的程式碼是一個實驗性的框架，用於訓練強化學習代理來解決謎題。
*   **視覺語言 (VL) 模型**: `src/core/vl_models/` 下的程式碼是一個實驗性的領域，用於探索如何使用多模態 AI 模型從圖片中解析謎題。

## 開發

### 執行測試
若要執行完整的測試套件並生成報告：
```powershell
.\run_tests.bat
```
測試結果和詳細日誌將會儲存在 `src/core/tests/reports/` 目錄下。

### 開發日誌
若要查看詳細的專案開發時程記錄，請參閱 [開發日誌](./dev_log.md)。