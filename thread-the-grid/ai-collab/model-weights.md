# 模型權重：放在哪、怎麼發佈的、怎麼取得

> 2026-09-19 收尾時寫。**兩個模型都不在版控裡**（一個 9.1 GB、一個無法逐位重生），改由外部託管：
>
> | 模型 | 公開位置（2026-09-19 公開）|
> |---|---|
> | RL policy `bc_multi_456_e6`（14 MB）| GitHub Release [`thread-the-grid-models-v1`](https://github.com/Hero0963/ml-workshop/releases/tag/thread-the-grid-models-v1) |
> | VLM `threadgrid-qwen35-4b-p4c:f16`（9.1 GB）| Hugging Face [`Hero0963/threadgrid-qwen35-4b-p4c-gguf`](https://huggingface.co/Hero0963/threadgrid-qwen35-4b-p4c-gguf) |
>
> 沒下載權重的 clone 服務照樣起得來，只是讀圖與 RL solver 分別回 503。
> 本檔回答：**為什麼放這兩個地方**（§3）、**怎麼發佈與驗收的**（§4）、**使用者怎麼裝**（§5）、**裝好怎麼驗**（§6）。
> 查證日期 2026-09-19；外部規格的出處附在各段。
> 專案 2026-09-19 改名 `thread-the-grid`，模型標籤與發佈的檔名是 `threadgrid-…`（本機匯出檔沿用舊檔名，位元組相同）。

---

## 1. 一句話

| 模型 | 大小 | 建議放哪 | 為什麼 |
|---|---|---|---|
| **RL policy**（`bc_multi_456_e6`）| **14 MB**（1 個檔）| **本 repo 的 GitHub Release** | 小、跟程式碼同一個地方、同一個版本號；Release 每檔上限 2 GiB、不限頻寬 |
| **VLM**（`threadgrid-qwen35-4b-p4c:f16`）| **9.1 GB**（2 個 GGUF）| **Hugging Face model repo** | 超過 Release 的 2 GiB 單檔上限；HF 是模型的標準發佈處，有 model card、授權欄、版本 |

**白話**：程式碼放 GitHub，大模型放專門放模型的地方，小模型跟著程式碼的 Release 走。
使用者只要多做「下載一個檔」和「下載兩個檔再匯入」兩件事。

---

## 2. 兩個模型的身分證（2026-09-19 實測）

### 2.1 RL policy

| 欄位 | 值 |
|---|---|
| 服務時讀的路徑 | `models/rl_a2/bc_multi_456_e6/checkpoints/model_final.zip`（`src/core/rl/solver_service.py` 的 `RUN_ID_BY_SIZE`，三個尺寸共用）|
| 大小 | 14,095,177 bytes |
| SHA-256 | `653efaa5ab0195b573d0c8c45ad592cca41e5c5f8d827d3c2037604182130a28` |
| 格式 | Stable-Baselines3 的 `.zip`（`MaskablePPO` 同架構，由行為克隆訓練）|
| 怎麼來的 | `uv run python -m src.core.rl.train_behaviour_cloning --goal goal3_multi --run-id bc_multi_456_e6 --epochs 6`（GPU 約 5.5 分鐘，資料集 `seed20300000_n20000_456`）|
| 成績 | 見 [`reports/2026-09-19_project-wrap-up.md`](reports/2026-09-19_project-wrap-up.md) §3 |

⚠ **「一行指令自己訓」對陌生人不成立**（部署指南以前這樣寫，已更正）：訓練讀的資料集 `datasets/` 不進版控，
而出題器用**牆鐘逾時**中止隨機回溯，重生出來是**另一包題目**（`train_config.py` 的 `MULTI_SIZE_DATASET` 註解）
⇒ 自己重訓得到的是「同配方的另一個模型」，數字會接近但不會逐位相同。**要重現已發表的數字，只能拿同一個檔。**

### 2.2 VLM

| 欄位 | 值 |
|---|---|
| Ollama 標籤 | `threadgrid-qwen35-4b-p4c:f16`（`.env` 的 `OLLAMA_MODEL_NAME`，要配 `VISION_PROMPT_VARIANT=finetune`）。2026-09-19 前叫 `zip-qwen35-4b-p4c:f16`，本機兩個標籤都在、指向同一份模型（`ollama cp`，ID `af36b1508cf7`）|
| 檔案 | 文字塔 **8,424,393,344 bytes** ＋ 視覺投影 **672,423,040 bytes**。本機匯出檔在 `models/gguf/zip-qwen35-4b-p4c-{text,mmproj}-f16.gguf`；發佈時叫 `threadgrid-qwen35-4b-p4c-{text,mmproj}-f16.gguf`（hardlink，位元組相同）|
| 匯出檔 SHA-256（要發佈的就是這兩個）| 文字塔 `28ec51a414244f306fdd6078325789e94da20968308695e49f94af9db3cdb773`、mmproj `4c2081c33cf7258f9cbc31ed06d522b35080c2e6c1dd6ceffc0b07d0b74a5e18` |
| Ollama 裡的 blob digest | model `sha256:fcbb7d29466bca36384336ccd6b60f644f2d17432bff6896b73493c10adb4e3b`、projector `sha256:4c2081c3…`（與匯出檔相同）|
| ⚠ 為什麼文字塔兩個雜湊不同 | **Ollama 匯入時重排了張量順序**。2026-09-19 用 [`gguf_compare.py`](reports/artifacts/wrap-up-acceptance/gguf_compare.py) 逐張量比對：metadata **35/35**、張量資料 **426/426 逐位元組相同**，只有排列順序不同；mmproj 23/23、298/298、順序也相同。⇒ **匯出檔就是服務中的模型**，驗證權重要比張量，不能比檔案雜湊 |
| Ollama manifest | **只有兩層**（model ＋ projector），**沒有 template 層**；`ollama show --modelfile` 顯示的 `TEMPLATE {{ .Prompt }}` 是預設值——完整的 prompt 由 app 自己渲染（`src/core/vl_models/`）|
| base | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)，**Apache-2.0**（衍生權重可散布，須附授權聲明）|
| 怎麼來的 | Colab L4 上 bf16 LoRA（975 步、1.56 小時）→ `src/core/vl_models/merge_lora.py` 併回 base → llama.cpp 轉 GGUF → `ollama create`。完整過程：[`reports/2026-08-29_vl-p4d-export-and-integration.md`](reports/2026-08-29_vl-p4d-export-and-integration.md) |
| 驗證過的 runtime | Ollama **0.32.13**（`docker-compose.ollama.yml` 已釘這一版）|

---

## 3. 提案：為什麼是這兩個地方

| 選項 | RL（14 MB）| VLM（9.1 GB）| 判斷 |
|---|---|---|---|
| **GitHub Release** | ✅ 每檔 < 2 GiB、不限頻寬 | ❌ 8.42 GB 超過單檔上限，要切檔 | **RL 採用** |
| **Hugging Face model repo** | 可以，但為 14 MB 多開一個平台不划算 | ✅ 模型的標準發佈處；可放 model card、授權、兩個 GGUF、Modelfile、LoRA adapter | **VLM 採用** |
| Ollama 官方 registry（`ollama push`）| — | 使用者一行 `ollama pull` 最方便，而且 push 的是**同一份 manifest** | **不做**（本人 2026-09-19 定案，理由見 §4.3）|
| `ollama run hf.co/<user>/<repo>` 直拉 | — | ⚠ **不採用**：官方文件沒保證 mmproj 會一起抓，且 template 會從 GGUF metadata 另外挑一個 ⇒ 和我們驗證過的那個模型**不等價**，而且錯的方式是安靜的（200 回空盤）| 不採用 |
| Git LFS 進版控 | — | ❌ 每個 clone 都得拉大檔；本 repo 紅線是「大檔不進版控」 | 不採用 |
| 讓使用者自己訓 | ⚠ 資料集不能逐位重生（§2.1）| ❌ 要付費 Colab L4、合併、轉檔，一整套 | 只當後備 |

出處（查證 2026-09-19）：
GitHub Release 每檔 < 2 GiB、每個 release 最多 1000 個檔、不限總量與頻寬——[GitHub Docs: About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)；
`ollama run hf.co/...` 的 template 與量化規則——[Hugging Face Docs: Use Ollama with any GGUF Model](https://huggingface.co/docs/hub/en/ollama)；
`ollama cp` ＋ `ollama push`——[Ollama Docs: Importing a model](https://docs.ollama.com/import)。
搜尋關鍵字：`ollama run hf.co vision model mmproj GGUF`、`GitHub release asset size limit`。

---

## 4. 給作者：怎麼發佈的（2026-09-19 已完成）

流程是「先非公開 → 下載回來驗 → 本人看過才公開」，兩邊都照這個順序走。

### 4.1 RL → GitHub Release（✅ 2026-09-19 公開）

```powershell
# 資產名稱加前綴（這是 monorepo，release 是整個 repo 共用的），先建成 draft
gh release create thread-the-grid-models-v1 <暫存資料夾>\threadgrid-rl-bc_multi_456_e6.zip `
  --draft --target main --title "thread-the-grid model weights v1" --notes-file <notes.md>

# 本人看過之後才做：公開（這一步才會建立 tag；tag 指向 main 的 80e91c5）
gh release edit thread-the-grid-models-v1 --draft=false
```

### 4.2 VLM → Hugging Face（✅ 2026-09-19 公開）

上傳的資料夾是 `thread-the-grid/models/hf-release/threadgrid-qwen35-4b-p4c-gguf/`（不進版控；大檔是 hardlink，不佔兩份空間）。
**model card 的原稿就是這個資料夾裡的 `README.md`**，要改 model card 改這份再上傳：

| 上傳到 HF 的檔 | 本機來源 |
|---|---|
| `threadgrid-qwen35-4b-p4c-text-f16.gguf` | `models/gguf/zip-qwen35-4b-p4c-text-f16.gguf` |
| `threadgrid-qwen35-4b-p4c-mmproj-f16.gguf` | `models/gguf/zip-qwen35-4b-p4c-mmproj-f16.gguf` |
| `Modelfile` | 已建，內容見下 |
| `README.md`（model card）| 已寫：base 與授權（Apache-2.0）、訓練資料是**自產合成圖**、用法（**prompt 必須逐字照訓練時的，裡面仍寫著 Zip**）、驗證數字、限制 |
| `LICENSE` | 從 Qwen 官方 repo 下載的 Apache-2.0 全文（含 Qwen 的著作權聲明）|
| `adapter/adapter_model.safetensors`（155,126,928 bytes，SHA-256 `92331a9e…a889`）＋ `adapter_config.json` | `models/colab_finetune/p4c_qwen35_4b_zip_checkpoints/checkpoint-975/`（給想自己合併或續訓的人；本人定案一起上傳）|
| `examples/input-6x6-red-walls.png`、`examples/solution-6x6-red-walls.png` | model card 的 Worked example 用圖（`illustrations/ui/acceptance-vlm-input-6x6.png` 與 [`vlm-walkthrough/solution.png`](reports/artifacts/vlm-walkthrough/)）|

`Modelfile`（**只有兩行**，路徑相對於 Modelfile 所在目錄；這樣建出來的 manifest 才會和驗證過的那個相同）：

```
FROM ./threadgrid-qwen35-4b-p4c-text-f16.gguf
FROM ./threadgrid-qwen35-4b-p4c-mmproj-f16.gguf
```

```powershell
cd thread-the-grid
uv run hf auth login        # 在自己的終端機做，token 不要貼進對話（見下方教訓）
uv run hf repo create Hero0963/threadgrid-qwen35-4b-p4c-gguf --private
uv run hf upload Hero0963/threadgrid-qwen35-4b-p4c-gguf models/hf-release/threadgrid-qwen35-4b-p4c-gguf .
# 本人看過之後才公開。huggingface_hub 0.36 的 CLI 沒有改可見度的指令，用 Python API：
uv run python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('Hero0963/threadgrid-qwen35-4b-p4c-gguf', private=False)"
uv run hf auth logout       # 發佈完就登出，並到 huggingface.co/settings/tokens 刪掉這把 token
```

- 一律 `uv run hf`：直接打 `hf` 可能跑到別的 venv 的舊版（2026-09-19 就抓到舊資料夾 `.venv` 的 `hf`）。
- 免費帳號的不公開空間 100 GB、公開空間 best-effort（[HF Docs: Storage limits](https://huggingface.co/docs/hub/storage-limits)），9.3 GB 不收費。
- **教訓**：2026-09-19 這把 token 被貼進了對話，因此發佈後立刻登出並作廢。下次在自己的終端機登入，
  並用一個用途一把、權限最小的 token（[HF Docs: User access tokens](https://huggingface.co/docs/hub/security-tokens)）。

### 4.3 Ollama registry 鏡像：本人定案不做（2026-09-19）

只省使用者一行指令（`ollama pull` 取代 `hf download` ＋ `ollama create`），卻要再開 ollama.com 帳號、登記容器公鑰、
維持兩處同步，而且流程未實測。HF 已足夠。

### 4.4 發佈前後的驗收紀錄（2026-09-19）

| 檢查 | 結果 |
|---|---|
| 上傳到 private repo | 9.25 GB、4 分 48 秒（平均約 222 Mbps）；匿名存取回 401 |
| 從 HF 下載回來核對 | 7 個檔 SHA-256 全與原檔相同（兩個 GGUF 與 adapter 等於 §2.2 的值）|
| 用下載檔 `ollama create` 成另一個標籤 | manifest 的 config 與兩層 layer digest 與服務中的 `:f16` **完全相同**（`2291240b…`／`4c2081c3…`／`fcbb7d29…`），Ollama 回報 `using existing layer` ⇒ 張量重排是決定性的，§5.2 的預期 digest 已實測成立 |
| `vision_check.py`（同收尾的尺）| 全新 6 張（seed 919000000）**6/6**、held-out 4 張 **4/4**，版面／牆／解標準答案盤三項全對：[`hf-download-vision-*.json`](reports/artifacts/wrap-up-round-2/) |
| 公開後匿名存取 | HF 頁面、API、README 200；三個大檔 range 讀取 206 且檔頭正確；GitHub 附件 200、14,095,177 bytes、SHA-256 `653efaa5…0a28` 相符 |
| model card 渲染 | 用 Chrome 開公開頁面截圖逐段看過，Worked example 的兩張圖正常顯示 |

---

## 5. 給使用者：拿到權重之後怎麼裝

### 5.1 RL

```bash
cd thread-the-grid
mkdir -p models/rl_a2/bc_multi_456_e6/checkpoints
curl -L -o models/rl_a2/bc_multi_456_e6/checkpoints/model_final.zip \
   https://github.com/Hero0963/ml-workshop/releases/download/thread-the-grid-models-v1/threadgrid-rl-bc_multi_456_e6.zip
sha256sum models/rl_a2/bc_multi_456_e6/checkpoints/model_final.zip   # 必須等於 §2.1 的值
```

用 `curl` 是因為它不用登入（2026-09-19 匿名下載實測，SHA-256 相符）；已登入 `gh` 的話，
`gh release download thread-the-grid-models-v1 --repo Hero0963/ml-workshop -p threadgrid-rl-bc_multi_456_e6.zip` 也行。

**不用重啟**：`models/` 是唯讀掛進 app 容器的，檔案一出現，下一個 RL 請求就會載入（2026-09-19 在全新 clone 上實測）。

### 5.2 VLM

```bash
cd thread-the-grid
uv run hf download Hero0963/threadgrid-qwen35-4b-p4c-gguf --local-dir models/vlm   # 不用登入
# ollama 容器把這個 checkout 的 ./models 唯讀掛在 /models，所以直接在容器裡匯入：
docker exec threadgrid_ollama_server ollama create threadgrid-qwen35-4b-p4c:f16 -f /models/vlm/Modelfile
docker exec threadgrid_ollama_server ollama show threadgrid-qwen35-4b-p4c:f16   # 應看到 vision 與 Projector
```

匯入後的 model blob 應該是 `sha256:fcbb7d29…`（Ollama 0.32.13 上 2026-09-19 實測：從 HF 下載的檔匯入，三個 digest 與服務中的模型完全相同，見 §4.4）。
換了 Ollama 版本而不相符也不代表壞了：用 `gguf_compare.py` 比張量，或直接跑 §6 的 `vision_check.py`，那才是權威驗收。

`.env` 的預設值本來就是 `OLLAMA_MODEL_NAME=threadgrid-qwen35-4b-p4c:f16`、`VISION_PROMPT_VARIANT=finetune`，不用改。
⚠ ollama 容器的 `./models` 是「**當初起它的那個 checkout**」的；多個 checkout 時，把檔放進那一個（`docker inspect threadgrid_ollama_server` 看 Mounts）。
⚠ **需要 NVIDIA GPU**（ollama 容器預約 GPU）；沒有的話 `start.py` 會說明並照常起其他服務。CPU 推論未測。

---

## 6. 裝好之後怎麼驗

```bash
python start.py      # 會說 RL 權重找到了沒、Ollama 裡有沒有 .env 指定的那個視覺模型
```

完整驗收（與 2026-09-19 收尾時同一把尺）：[`reports/artifacts/wrap-up-acceptance/`](reports/artifacts/wrap-up-acceptance/)
的 `acceptance.py`（全部 solver，答案用獨立裁判判）與 `vision_check.py`（版面、牆、路徑能不能解**標準答案那張盤**）。
測資可以自己出：`uv run python -m src.core.vl_models.dataset_builder --count 6 --name mytest --seed 919000000`。
收尾時的結果：全新合成 6/6、held-out 4/4 全對。

---

## 7. 不下載權重時的後備（例如沒有 GPU、或 HF 暫時連不上）

| 功能 | 後備 | 代價 |
|---|---|---|
| 讀圖 | `.env` 改 `OLLAMA_MODEL_NAME=qwen3.5:4b-q8_0`、`VISION_PROMPT_VARIANT=sized`，再 `docker exec threadgrid_ollama_server ollama pull qwen3.5:4b-q8_0` | 未微調：牆 F1 約 0.44、真實截圖端到端 2/6（2026-08-15 實測）|
| RL | 照 §2.1 自己生資料集＋訓練（資料集生成見 [`handover-rl-solver.md`](handover-rl-solver.md) §4）| 另一包題目、另一個模型；GPU 約 5 分鐘 |

---

## 8. 安全與未來

- **SB3 的 `.zip` 會用 pickle 反序列化部分內容 ⇒ 只載自己信任的來源，並核對 SHA-256。** GGUF 與 safetensors 是純資料格式。
- 未來可做（都**沒做**）：`start.py --fetch-models` 一鍵下載與驗證；VLM 轉 Q8_0 讓下載減半（**要重跑 held-out 驗收才能換**）；
  RL checkpoint 轉 safetensors ＋ JSON 設定，拿掉 pickle。
