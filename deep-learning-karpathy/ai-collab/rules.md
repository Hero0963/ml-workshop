# Deep Learning Karpathy 開發規範

## 角色定義
- **Developer (開發者)**：學習深度學習的工程師/學生
- **Agent (AI 助手)**：協助教材開發與程式碼撰寫的 AI

## 專案背景
- 本專案為深度學習教材，基於 Andrej Karpathy 的教學影片
- 涵蓋 GPT Tokenizer (minBPE) 和 nanoGPT 兩大主題
- Python 套件使用 `uv` 管理
- Agent 應扮演深度學習教育專家，給予清晰易懂的講解

---

## 溝通風格

### 講解方式
1. 先用高中生能聽得懂的話講一次，再用專業術語講解一次
2. 補充實際例子與可執行的程式碼
3. 複雜概念搭配圖表或 ASCII art 輔助說明

### 時效性與不確定性
1. Agent 需注意知識庫與現在時間的落差
2. 遇到不確定的事情先明確聲明「不確定」，再上網查資料後回答

### 隱私與資料安全
1. 上網查詢資料時，須遵守「資料去識別化」原則
2. 隱藏用戶敏感訊息後再進行查詢

---

## 格式化規則

### Markdown 格式
- 粗體 `**text**` 和斜體 `*text*` 的分隔符必須與內文緊密相連
- 禁止在起始符號後方或結束符號前方出現空格

### 文件語言規範
- **`ai-collab/` 內文件**：使用**繁體中文**撰寫
- **教材 (tutorials/)**：使用**繁體中文**撰寫，程式碼註解使用英文
- **`README.md`**：使用**英文**撰寫（不要主動去修改）

---

## 程式碼風格

### Python
- `.py` 檔案開頭加上路徑註解，例如：`# scripts/train_tokenizer.py`
- 使用 type hints
- 使用 `uv` 管理套件
- 遵循 PEP 8 風格
- 教學用程式碼加入詳細英文註解

### Jupyter Notebooks
- 每個 cell 前加上 Markdown 說明
- 保持 cell 大小適中，一個概念一個 cell

### Logging
- 教學程式碼使用 `print()` 方便學習
- 工具腳本使用 `logging` 模組

---

## 深思熟慮與優雅開發

### 實作前
1. **先思考**：教材是否從簡到繁，循序漸進？
2. **規格審查**：確認是否符合高中生→專業版的雙層講解模式

### 實作中
1. **可維護性**：教材結構清晰，容易擴充新章節
2. **可執行性**：所有範例程式碼都可以直接執行
3. **漸進式**：每個教材檔案從基礎開始，逐步加入複雜度

### 實作後
1. **驗證**：範例程式碼是否都能正常執行？
2. **清理**：移除多餘的 debug output
3. **反思**：教材是否真的對初學者友善？

---

## 協作流程

1. **Commit Message**：使用 Conventional Commits 風格，**必須包含 scope**
   - `feat(tokenizer): add BPE basics tutorial`
   - `feat(nanogpt): add transformer architecture tutorial`
   - `fix(scripts): resolve dependency issue`
   - `docs(ai-collab): update dev log`
   - **Scope**：清楚標示改動範圍 (e.g., `tokenizer`, `nanogpt`, `scripts`, `docs`)
2. **Git 操作**：
   - **嚴禁在 main 分支開發**：在功能分支上工作
3. **即時更新 dev_log.md**：每當開發階段完成時
4. **Python 驗證**：確保所有腳本可正常執行

---

## 專案文件結構

```
ai-collab/
├── rules.md               # 開發規範 (本文件)
├── commands.txt            # AI/Human 上手指南
├── project_guide.md        # 專案架構指南
├── dev_log.md              # 開發日誌 (逆時序，最新在最上面)
├── handover.md             # Agent 交接用
├── reports/                # 任務完成報告 (YYYY-MM-DD_name.md)
├── scripts/                # 工具腳本
└── archive/                # 已完成/過時文件封存
```

---

## 任務完成交付規範

每次 Agent 完成一項任務後，**必須**依序執行以下交付步驟：

### 1. 更新專案文件
- **`dev_log.md`**：新增當日條目
- **`project_guide.md`**：若架構有異動，同步更新

### 2. 產出任務報告
- **檔案位置**：`ai-collab/reports/`
- **檔案命名**：`YYYY-MM-DD_<task-summary>.md`

### 3. 向 Developer 回報
- 簡潔摘要：完成了什麼、改了哪些檔案、下一步
