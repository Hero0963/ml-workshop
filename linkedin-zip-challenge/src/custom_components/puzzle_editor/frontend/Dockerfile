# src/custom_components/puzzle_editor/frontend/Dockerfile

# 步驟 1: 選擇基礎映像
FROM node:lts-alpine

# 步驟 2: 設定工作目錄
WORKDIR /app

# 步驟 3: 複製 Node.js 依賴描述檔
COPY package*.json ./

# 步驟 4: 安裝 Node.js 依賴
RUN npm install

# 步驟 5: 複製 Svelte 原始碼
COPY . ./

# 步驟 6: 設定環境變數 (這會被 vite.config.js 讀取)
ENV SVELTE_PORT=5173

# 步驟 7: 開放容器端口
EXPOSE 5173

# 步驟 8: 定義容器啟動指令 (已修正)
# 我們不再需要傳遞任何參數，
# 因為 vite.config.js 會自動處理 host 和 port
CMD ["npm", "run", "dev"]