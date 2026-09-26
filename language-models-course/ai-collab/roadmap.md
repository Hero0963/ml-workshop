# Roadmap — language-models-course

> **新 session 第一站**：現況、下一步、已定案不要再重開的決策。
> 做完一件事就更新本檔（現況＋下一步），細節寫進 [`dev_log.md`](dev_log.md)。架構與執行方式見 [`project_guide.md`](project_guide.md)。
> Last Updated: 2026-09-26

## 現況（2026-09-26）

- **課程 v1 完成**：16 課講義（`lessons/00`–`15` ＋ 附錄 A）、13 個已執行的實驗 notebook（`notebooks/01`–`13`）、參考實作 `src/lm_course/`（有測試）、參考資料 `references.md`、授權說明 `NOTICE.md`。
- 整合的五個來源：word2vec、GPT-2、Stanford CS336、nanochat、text embedding 模型；依「做出一個模型的順序」重排，重疊處只講一次（對照表在 `lessons/00_course_map.md`）。
- 所有 notebook 都在**雲端 4 核心 CPU**上實跑過（torch 2.14.0），輸出就是那次執行的紀錄；每本執行後都逐條核對過 markdown 的敘述與輸出（2026-09-26，細節見 `dev_log.md`）。各 notebook 的實測時間見 `project_guide.md`。
- 本人的機器上**還沒有建過環境**：`pyproject.toml` 的 `dependencies` 是空的，等本人 `uv add`（見「下一步」第 1 項）。

## 下一步

1. **本人建立基線** — `cd language-models-course && uv add torch numpy matplotlib loguru regex ipykernel && uv add --dev pytest && uv run pytest`。
2. **有 GPU 時加大實驗** — Lab 06 用 `scripts/pretrain.py --steps 3000`（或更大的 `base_model_config`），再重跑 Lab 09、11、12、13；Lab 07 可加更大的算力預算。
3. **補練習題參考解答** — 目前每課只有題目；可放在 `lessons/solutions/`。
4. **第 13、15 課保鮮** — embedding 模型與前沿模型的版本約半年重查一次（照 `AGENTS.md` §4，用 Hugging Face 組織頁依建立日期排序），更新查證日期。
5. **加法題的 RL 重試（有 GPU 時）** — Lab 12 §7 的反例：用更大的 base model、每步更多樣本，看 temperature 1 的正確率（pass@1）要高到什麼程度，GRPO 才開始有效。
6. **決定授權** — 子專案尚未放 `LICENSE`；要不要比照 `thread-the-grid`（Apache-2.0）由本人決定。注意 notebook 輸出中的 TinyStories 片段屬 CDLA-Sharing-1.0（見 `NOTICE.md`）。

## 已定案（不要再重開）

| 日期 | 決策 | 理由 |
|---|---|---|
| 2026-09-25 | 講義用 Markdown（繁中）＋ 已執行的 notebook；程式與註解用英文 | 與 repo 其他教材一致（`rules.md`）、比照 `diffusion-models-course` |
| 2026-09-25 | 依建構順序排課，不依來源分章 | 五個來源重疊很多（BPE、Transformer、後訓練）；重疊處只講一次，並用表格比較各來源的差異 |
| 2026-09-25 | 一個可切換的 `GPT`（GPT-2／Llama 類／nanochat 類）取代三份模型程式 | 第 04–06 課的比較只需改 config；GPT-2 權重載入同一個類別驗證實作正確 |
| 2026-09-25 | 資料只用 TinyStories（GPT-4 驗證檔 22.5 MB，CDLA-Sharing-1.0），執行時下載 | 授權明確、CPU 上小模型也能學出通順文字；網頁資料的授權與大小都不適合 |
| 2026-09-25 | 課程 tokenizer：4,096 個 token、GPT-4 式預切但數字單一位數、訓練時就放入 9 個 chat 特殊 token | CPU 小模型的 embedding 不能太大；單一位數讓加法任務可學；特殊 token 名稱沿用 nanochat |
| 2026-09-25 | base model：nanochat 類、4 層、寬 256、context 256、Muon＋AdamW、WSD、1,500 步 | CPU 約 27 分鐘；驗證 bits-per-byte 0.705（GPT-2 124M 零樣本 0.872），生成的故事通順 |
| 2026-09-25 | SFT 任務用「兩數相加」，三種回答格式（直接／step by step／計算機） | 三種格式的正確率差異本身就是 chain of thought 與工具使用的教材 |
| 2026-09-26 | RL（Lab 12）主任務改為「主題跟隨」：故事開頭有沒有提到使用者給的主題字；加法 RL 留作 §7 的反例 | 加法的 GRPO 在 5M 模型、CPU 規模學不起來（試過多組學習率與設定；梯度方向對，但訊號被雜訊淹沒）。主題跟隨 60 步就學會，還能示範 reward hacking、KL 懲罰與推廣到沒看過的主題 |
| 2026-09-26 | n-gram 用 Witten–Bell 插值，前文用乘法雜湊當鍵 | 固定權重的插值在短前文時輸給加 k 平滑；256 進位整數鍵最多 7 階，看不到驗證 bpb 的 U 形曲線 |
| 2026-09-26 | Lab 07 的 IsoFLOP 保留 3 個預算 × 7 種大小（CPU 約 80 分鐘） | 三條曲線才看得出最佳點隨算力移動；notebook 開頭註明「只跑前兩個預算約 25 分鐘」的省時做法 |
| 2026-09-25 | 程式依論文與公開文件自寫，不複製 nanochat／nanoGPT／CS336 的程式；CS336 講義只連結 | 著作權與授權（CS336 講義未附授權；作業有 honor code） |
| 2026-09-25 | 相依套件由本人 `uv add`；`pyproject.toml` 預先把 `torch` 導向 cu126 explicit index | repo 規則 `AGENTS.md` §5；與 `diffusion-models-course` 相同 |
| 2026-09-25 | GPT-2 與 tokenizer 的參考值（token id、logits）寫死在測試裡，來自開發時用 transformers／tiktoken 的交叉驗證 | 不把 transformers、tiktoken 加成相依套件；需要網路的測試標記 `network`，離線自動略過 |
| 2026-09-25 | 前沿（第 15 課）與 embedding 模型清單（第 13 課 §2.8）只寫一手來源查得到的版本與規格 | `AGENTS.md` §4 的保鮮紀律；查證時發現許多 2026 年的模型（DeepSeek-V4、Qwen3.8、Kimi K3…）比訓練資料新 |
