# 任務計畫 — 下一輪的三條平行 track（2026-09-12）

> **給接手的 agent：只讀屬於你那一節就夠了。** 平行開發的規則（worktree、資源號誌、
> 共用檔案協定）在 [`../../../AGENTS.md` §10](../../../AGENTS.md)。
> 這三條 track **刻意切成檔案不重疊**，所以可以同時進行。

---

## 0. 起點與前提

- 三條都從 **`main`** 長新 worktree／分支（`feat/rl-a2-training` 已完成並待合併，
  它帶著 Docker 修復、solver registry、RL solver 上線、多尺寸模型）。
- **先合併 `feat/rl-a2-training` 再開這三條**，否則 Track B 會缺 Docker 那批修復。
- 每條 track 的第一件事都一樣：`uv sync` → `uv run pytest` → `uv run ruff check .` 建立基線
  （2026-09-12 基準：**276 passed, 8 xfailed**）。

**檔案所有權（切分的依據）**

| Track | 擁有 | **不准動** |
|---|---|---|
| A｜RL 微調 | `src/core/rl/`（`solver_service.py` 除外）| `src/app/`、`src/core/solvers/`、Docker 檔 |
| B｜基礎設施與 solver | `.devcontainer/`、`docker-compose*.yml`、`start.py`、`src/core/solvers/`、`src/app/schemas/` | `src/core/rl/`、`src/core/vl_models/` |
| C｜視覺評估 | `src/core/vl_models/`、`notebooks/` | `src/core/rl/`、`src/core/solvers/`、Docker 檔 |

三條都會碰 `ai-collab/dev_log.md` 與 `roadmap.md` ⇒ **各加各的 `###`、只改自己那一項**。

---

## Track A — BC → PPO 微調（RL 主線）

**為什麼做**：本人 2026-09-11 定案「**就是要用 RL 做**」。目前最好的模型是**監督式**訓練出來的，
而 **BC → PPO 微調是唯一能推翻「這題不該用 RL」的實驗**。

**先讀**：[`../handover-rl-solver.md`](../handover-rl-solver.md) §1–§3 ＋
[`../notes/01-rl-methods-explained.md`](../notes/01-rl-methods-explained.md) §4。

**要做的事**

1. 實作 `train_maskable_ppo.py` 的 `--init-from <run-id>`：載入 BC 的 checkpoint 當 PPO 起點。
2. 微調設定：**全長、不用 curriculum**（`CurriculumState(current_k=None)`，
   `_maybe_promote` 對 `None` 是 no-op）。
3. 從 `bc_multi_456` 出發，先在 4×4 上跑短的（分鐘級）確認不會崩，再考慮 6×6。
4. 對照組是 **BC 自己**（同一個 checkpoint 未微調），用 deterministic 與 best-of-32 兩個數字比。

**done 條件**

- [ ] `--init-from` 有測試（載入後的 policy 權重與來源 checkpoint 相同）
- [ ] 至少一組「BC vs BC+PPO」的對照數字落盤在 `logs/rl_probes/`
- [ ] 明確回答：**PPO 微調買到了什麼？** 買到 ⇒ RL 在這題有加值；沒買到 ⇒
      「這題不該用 RL」從假設升級成有證據的結論。**兩個方向都是交付物。**

**資源**：需要 GPU，**要搶號誌**。6×6 全長是小時級，**開跑前要本人授權**。

**已知陷阱**

- ⚠ **PPO 用 `V(s)` 算 advantage**。value head 已經在 BC 訓好（2026-09-12 量過
  `--value-coef 0.5` vs `0` 差 −0.0083，在雜訊內）⇒ 直接用帶 critic 的 checkpoint。
- ⚠ **6×6 的 seed 雜訊從沒量過**（用 BC 量約 16 分鐘：3 seed × 228s ＋ 評估）。
  沒量之前，6×6 的 0.0x 差異一律標「有動但未確證」。
- ⚠ **不要外推**。這條 track 外推錯過三次。

---

## Track B — 基礎設施與 solver 上線

**為什麼做**：服務已經能一鍵起，但**image 有 22.9 GB**，而且十種 solver 只上線四種。

**先讀**：[`../deployment-guide.md`](../deployment-guide.md)。

**要做的事**

1. **把 image 瘦下來**。基底是 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`（約 10 GB，含 CUDA toolkit），
   `uv sync` 之後又裝一份 `torch 2.4.1+cu121`。**app 容器根本不用 GPU**（GPU 在 ollama 那邊）
   ⇒ 換成 slim 基底應該能大幅縮小。
   ⚠ **未驗證系統相依**：gradio／matplotlib／opencv 可能需要 `libgl1`、`libglib2.0-0` 之類。
   **時間盒：兩次建置失敗就退回可動版本並把原因寫下來。**
2. **實跑驗證 `python start.py --dev`**——目前**只驗過 `--status` 這條路徑**，完整啟動沒跑過。
3. **把六種啟發式 solver 掛上 API**：在 `src/core/solvers/registry.py` 加 `SolverEntry`
   （啟發式需要 `attempts` 參數，見 `src/app/schemas/solver.py`）。

**done 條件**

- [ ] image 大小前後對照（`docker images` 實際輸出），或「試過但退回」的理由
- [ ] `python start.py --dev` 完整啟動 ＋ `/api/echo/health` 200 的實際輸出
- [ ] 十種 solver 都能從 `/api/solver/solve` 叫到，並貼出每一種的回應
- [ ] `deployment-guide.md` 與兩份 README 的數字同步更新

**資源**：不吃 GPU。Docker build 主要吃網路與 IO，**不用搶號誌**。

**已知陷阱**

- ⚠ **正式 image 會把原始碼烤進去**，`docker compose restart` 不會更新程式碼，要 `--build`。
- ⚠ `.env` 不進版控、每個 worktree 一份，**所以它會各自過期**。視覺端點的模型與 prompt
  必須配對，否則會安靜地回 200 ＋ 空盤面。

---

## Track C — 把視覺評估集變難

**為什麼做**：合成 held-out 的四層指標**全部飽和在 1.000**，它已經分辨不出兩種做法的差別
⇒ 視覺這條線**現在沒有可用的尺**。本人已定案**不做真實截圖標註**，所以要從合成資料下手。

**先讀**：[`../handover-vlm-parser.md`](../handover-vlm-parser.md)。

**要做的事**

1. 讓產生器能做出更難的圖：**視覺雜訊、多種渲染風格、模擬截圖失真（縮放／壓縮／裁切）、更大盤面**。
2. 重新評分現有模型，**證明新評估集真的有鑑別力**（指標必須掉下 1.000，且不同設定要分得開）。
3. 如果模型在新集上掉很多，**先判斷是模型不行還是圖根本不合理**——後者要修產生器。

**done 條件**

- [ ] 新評估集的產生方式可重現（seed ＋ 指令落盤）
- [ ] 同一個模型在舊集與新集上的分數對照表
- [ ] 明確結論：新集**有沒有**鑑別力（不同設定分數要分得開才算有）

**資源**：需要 GPU（VLM 推論），**要搶號誌**。

**已知陷阱**

- ⚠ **牆 F1 會被無牆題灌水**（無牆題預測 0 道就白拿 1.0）⇒ 看 `mean_wall_f1_walled_only`。
- ⚠ 評估**不要用 batch 1 逐筆生成**，會慢到比訓練還貴。

---

## 建議的開工順序

**B 可以立刻開始**（不吃 GPU、不等別人）。**A 與 C 都要 GPU ⇒ 錯開**，
或先讓 A 做不吃資源的部分（實作 `--init-from` ＋ 測試），等 C 的評估跑完再搶號誌。
