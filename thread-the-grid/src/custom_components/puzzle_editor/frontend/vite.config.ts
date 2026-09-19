// src/custom_components/puzzle_editor/frontend/vite.config.js
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vitejs.dev/config/
export default defineConfig({
  base: '/svelte-ui/',
  plugins: [svelte()],

  /**
   * API 的位址不寫在這裡，也不寫死在 Index.svelte：
   * 建置版走同源相對路徑（app 自己在 /svelte-ui/ 上服務它），dev server 則靠環境變數
   * `VITE_API_URL`（compose 會傳進來），因為頁面來自 :5173 而 API 在 app 的埠上。
   * 不透過 Docker 直接 `npm run dev` 的話，要自己設 `VITE_API_URL`。
   */
  server: {
    host: '0.0.0.0', // 依然保持 '0.0.0.0'

    /**
     * 動態讀取環境變數 SVELTE_PORT
     * * 1. process.env.SVELTE_PORT 從容器環境讀取變數。
     * 2. Number(...) 將其從字串 (e.g., "5173") 轉換為數字 (5173)。
     * 3. || 5173 提供一個合理的預設值 (fallback)，
     * 以防萬一 .env 檔案遺失或未設定該變數，
     * Vite 服務也能正常啟動。
     */
    port: Number(process.env.SVELTE_PORT) || 5173,

    strictPort: true, // 保持 true
  }
})