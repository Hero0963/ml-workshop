# .devcontainer/Dockerfile

# 步驟 1: 選擇基礎映像
FROM python:3.11-slim

# 步驟 2: 安裝 uv
RUN pip install --no-cache-dir uv

# 步驟 3: 設定工作目錄
WORKDIR /app

# 步驟 4: 複製依賴描述檔
COPY pyproject.toml uv.lock pytest.ini ./


# 步驟 5: 安裝所有依賴 (包含開發依賴)
# 移除了 --no-dev 參數，這樣 pytest 等開發工具就會被安裝
RUN uv sync

# 步驟 6: 複製原始碼 (這一步仍然需要)
# 雖然我們會掛載程式碼，但在第一次建置或沒有掛載時，容器內依然需要有一份程式碼
COPY src ./src

# 步驟 7: 設定預設的環境變數
# 根據您的 .env 檔案，將預設值改為 7440
ENV APP_PORT=7440

# 步驟 8: 開放容器端口
EXPOSE 7440

# # 步驟 9: 定義開發環境的啟動指令
# # - 加上 --reload 參數，以便在程式碼變更時自動重啟服務
# # - 使用 ${APP_PORT} 來動態設定端口
# CMD ["/bin/sh", "-c", ". .venv/bin/activate && uvicorn src.app.main:app --host 0.0.0.0 --port ${APP_PORT} --reload"]
