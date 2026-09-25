# 開發日誌（逆時序，最新在上）

## 2026-09-25：建立課程 v1

- **任務**：本人交辦「想學 diffusion model，收集材料後做一份教學課程」，全權處理、分階段 commit／push。
- **做了什麼**：
  - 查證資料來源：約 60 篇 arXiv 論文逐篇核對標題與首次提交日期；教材網頁確認可連線；前沿（2025–2026）只引用論文摘要裡的數字，二手資訊另外標注。
  - 參考實作 `src/diffusion_course/`：schedule、DDPM、DDIM、score／Langevin、flow matching、CFG、ToyMLP／U-Net／TinyDiT、訓練迴圈（EMA）、評估用分類器。50 個 pytest，全部通過。
  - 講義 12 課＋附錄 A；實驗 notebook 9 本，全部在雲端 CPU 實跑。
  - `NOTICE.md`：資料集與相依套件的授權；`references.md`：分級的延伸閱讀。
- **關鍵數字**（實跑）：
  - Lab 02：Langevin 的步長偏差實測 1.049／1.332／2.004，理論 $1/(1-\eta/2)$ 為 1.053／1.333／2.000；兩座不等高的山，普通 Langevin 抓到 51%（目標 80%），annealed Langevin 79%。
  - Lab 04：損失對 $t$ 的曲線，$t=0$ 約 1.0、$t=975$ 約 0.0003，說明 $L_{\text{simple}}$ 有由資料決定的下限。
  - 速度：4 核心 CPU、batch 128，U-Net 32 通道一步 1.2 秒、16 通道 0.47 秒；TinyDiT 0.28 秒。
- **踩到的坑**：
  - 新版 ruff（0.16）會多報 I001／RUF007；repo 的 pre-commit 釘在 0.14.1，以它為準。
  - Lab 05 的誤差地圖原本用逐點相對誤差，在真實 score 接近 0 的地方（密度山頂）會爆掉、畫出假的高誤差；改成「絕對誤差 ÷ 該雜訊等級 score 的均方根」。
  - Lab 04 的 $t=999$ 時 $\hat x_0$ 散得到處都是：除以 $\sqrt{\bar\alpha_{999}} \approx 0.006$ 放大誤差，正好拿來說明 $\epsilon$-prediction 在高雜訊端的數值問題。
  - MNIST 官方頁面當天 503，無法一手查證授權；依較嚴格的 CC BY-SA 3.0 處理並記在 `NOTICE.md`。
