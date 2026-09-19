# GPT-6 等級的 computer-use agent 會怎麼解 Zip？（survey，2026-09-19）

> 本人問：「GPT-6 computer use 可以用小畫家畫圖、可以玩 2048，以後這種等級的 agent 叫它解 zip-puzzle 會怎麼做？」
> 這份先**查證前提**，再用查得到的實證**推演**。⚠ **沒有找到任何人讓 GPT-6 等級的 agent 玩 LinkedIn Zip 的一手紀錄**，
> 所以 §5 是推演，不是觀察。查證日期 2026-09-19；時效性內容，半年後請重查。
> 主報告：[`2026-09-19_project-wrap-up.md`](2026-09-19_project-wrap-up.md)。

---

## 0. 一句話

**它多半會把我們這個專案「現場重做一遍」**：看截圖讀出盤面、寫一個搜尋程式解、再用滑鼠把路徑拖出來，
失敗就看畫面回饋重來。能不能成功取決於兩件事——**有沒有把牆讀對**（這正是我們的 VLM 花最多力氣解掉的問題），
以及**有沒有一個會說「錯了」的驗證器**讓它迭代。差別是：它每題大概要花分鐘級時間和 API 費用，
我們的管線在本機 16 GB 顯卡上約 5 秒一張、而且每一段都能單獨量。

---

## 1. GPT-6 是什麼（一手：OpenAI 公告）

- 正式名稱 **GPT-6 Astra**，**2026-09-03** 發布；分階段開放給 ChatGPT Plus／Pro／Business／Enterprise 與 API（$10／$50 每百萬 input／output token）。
- 官方宣稱電腦操作（computer use）目前最強：

| 評測 | GPT-6 Astra | GPT-5.6 Sol | Claude Opus 5 |
|---|---|---|---|
| OSWorld 2.0（v2026.08.08，offline，partial score）| **72.6%** | 65.7% | 70.2% |
| ScreenSpot-Pro（no tools）| **92.7%** | 76.9% | — |
| Agents' Last Exam | **59.3%** | 53.6% | 55.5% |

  OSWorld 延遲模擬中每題約 **40 分鐘**（Sol 約 75 分鐘）。官方示範：在 KiCad 佈 PCB、Excel、遊戲開發、填 1040 報稅表、前端 QA。
  同時更新了 Codex harness，Mind2Web 上完成任務快 1.9 倍。
- 出處：[OpenAI, GPT-6 Astra](https://openai.com/index/gpt-6-astra/)（WebFetch 被 403 擋，改用 Chrome headless 讀取 zh-TW 版頁面）。

---

## 2. 查證本人提到的兩件事

| 說法 | 查證結果 |
|---|---|
| 「可以用小畫家畫圖」| **有，但出處是社群示範，不是 OpenAI 官方**：發布首日就有使用者分享讓 Astra 打開 Microsoft Paint 畫肖像、把照片重畫一遍的影片（二手整理：[Happycapy](https://happycapy.ai/blog/gpt-6-astra-wants-your-mouse)、[magiccreator](https://magiccreator.ai/astra)）。**官方公告頁沒有提到小畫家。** |
| 「可以玩 2048」| **查無一手出處**。搜得到的「2048」是**用 Astra 做出來的** 2048 類遊戲（CityMaker，4×4 城市街區合併建築），不是 Astra 去**玩** 2048。本報告不把它當成事實。 |

搜尋關鍵字：`GPT-6 computer use OpenAI`、`OpenAI GPT-6 release announcement 2026`、`GPT-6 Astra Microsoft Paint drawing 2048 game computer use demo`、
`"Astra" OpenAI "2048" game computer use`、`GPT-6 Astra LinkedIn Zip Queens puzzle computer use plays`。

---

## 3. 這一級的 agent 實際怎麼「玩」東西

**白話**：它不是直接讀遊戲的程式，而是跟人一樣——**看截圖、想、動滑鼠鍵盤**，再看下一張截圖。
差別是它可以一邊玩一邊寫程式、跑程式。

**術語**：computer-use agent 是一個迴圈：`截圖 → 模型推理 → 動作（點擊／拖曳／打字／執行指令）→ 新截圖`。
Anthropic 的 computer use 工具就明列 `left_click_drag` 這類動作（[Claude Docs: Computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)）；OpenAI 的對應是 Codex harness。

**兩個有細節的案例**：

1. **Portal**（Tom's Hardware 報導的愛好者實驗）：用 **MCP ＋ 改過的 SourcePauseTool**，**模型思考時遊戲暫停**，
   模型拿到截圖與角色座標、送出一串輸入後遊戲才繼續。約 24 小時、**3,336 次工具呼叫**、API 帳面成本 **$571.18**。
   同站也報導它在 Minecraft 被 Creeper 炸死後花了幾小時種馬鈴薯。讀者質疑 Portal 攻略早就在訓練資料裡
   （[Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/openais-gpt-6-astra-model-autonomously-completes-portal-in-24-hours-feat-cost-just-usd571-in-tokens)）。
2. **Baba Is You**（Quesma，2026-09-14）：154 關、文字介面、6 小時、各用原生 harness。Astra 解 80 關、Claude Fable 5.1 解 33 關。
   原文：「It solved 7 of 70 levels on the first attempt with no undo」——**絕大多數是靠試錯＋undo 解開的**。
   原文也說 Astra 的推理在 Codex log 裡是加密的、只看得到指令，並自問是否訓練過攻略
   （[Quesma](https://quesma.com/blog/gpt-6-astra-solves-puzzles/)）。

---

## 4. 通用模型解「約束謎題」的實證

Zip 屬於鉛筆謎題（constraint satisfaction）。最相關的公開量測是 **Pencil Puzzle Bench**
（Justin Waugh，[arXiv:2603.02119](https://arxiv.org/abs/2603.02119)，2026-03-02；[repo](https://github.com/approximatelabs/pencil-puzzle-bench)）：

- 62,231 題／94 種，精選 300 題／20 種；**每一步都能被規則引擎檢查**（例如「兩個塗黑格相鄰」）。
- **單發作答 vs agentic（多輪、有驗證器回饋）**：Claude Opus 4.6 **0.3% → 30.0%**、GPT-5.2@xhigh **20.2% → 56.0%**。
  agentic 中位數 29 輪、17 分鐘，最長 1,221 輪、14.3 小時。
- ⚠ abstract 沒列出 20 種題型是否包含 Numbrix／Hidato 這類「哈密頓路徑」題，**不宣稱它含 Zip 的同類題**。

**這對本專案的意思**：通用模型「一次想完」解約束謎題仍然很弱；**有驗證器的迭代**把成功率拉高一個量級。
這和我們 RL 的量測是同一件事——策略 deterministic 只有 0.54，**配上可驗證的 best-of-32 到 0.95**。

---

## 5. 推演：叫它解 Zip，它會怎麼做

### 5.1 三種可能的解法

| | 做法 | 優點 | 會卡在哪 |
|---|---|---|---|
| **A. 看了就畫** | 看截圖、在腦中推出整條路、直接用滑鼠拖出來 | 最像人 | 要一次連對整條哈密頓路徑——正是 Pencil Puzzle Bench 單發模式的弱點；拖曳要逐格精準 |
| **B. 看了寫程式** | 把截圖轉成格子資料 → 寫 DFS／回溯（Zip 的 solver 約數十行）→ 程式算出路徑 → 換算成螢幕座標拖曳 | **最可能**：這一級的 agent 能寫也能跑程式；搜尋保證正確 | **讀錯盤面就全錯**，而且錯得很安靜 |
| **C. 試錯** | 邊拖邊看遊戲的反應，錯了退回（Zip 本身允許拖回去）| 遊戲本身就是驗證器 | 慢、貴；Quesma 的 Baba Is You 顯示這一級 agent 大多這樣解 |

**最可能的實際樣子是 B ＋ C**：先寫程式解，拖出來之後看畫面確認，不對就修正讀到的盤面再解一次。

### 5.2 對照：這其實就是本專案的管線

| 步驟 | 通用 agent（推演）| 本專案（實測）|
|---|---|---|
| 讀盤面 | 通用多模態模型看截圖 | 微調 Qwen3.5-4B：合成 held-out **200/200**、收尾驗收全新 6 張＋held-out 4 張 **10/10** |
| 解 | 當場寫的回溯程式 | DFS／A\*／**CP-SAT**（精確）＋ RL 與五種啟發式（對照）|
| 驗證 | 看遊戲畫面 | `verify.is_solution`（獨立裁判）＋ `solvable` 旗標 |
| 成本 | 分鐘級、API 費用（OSWorld 每題約 40 分鐘；Portal $571）| 本機約 **5 秒**／張（第一張載模型 58 秒）|

**最可能失敗的地方和我們踩過的一樣：牆。** 我們未微調的 Qwen3.5-4B 在六張真實截圖上逐格準確率 0.947、號碼召回 0.917，
但**牆 F1 只有 0.438**，端到端 2/6（[VLM 報告](2026-08-29_vl-p4d-export-and-integration.md)）。
GPT-6 等級的模型視覺強很多（ScreenSpot-Pro 92.7%），但**它讀牆準不準沒有人量過**——這是推演裡最大的未知數。

### 5.3 對本專案的意義（future work 候選，**都沒做**）

1. **把 Zip 做成 computer-use agent 的評測環境。** 本專案已經有三樣評測最缺的東西：
   **無限量的新題**（出題器；Quesma 與 Tom's Hardware 讀者都擔心「攻略在訓練資料裡」，新生成的題沒有這個問題）、
   **精確的裁判**（`verify.is_solution`）、**難度旋鈕**（盤面大小、牆數、數字密度）。
   Svelte 編輯器本身就是一個可以讓 agent 用滑鼠操作的畫布。
2. **拿通用模型當 VLM 的對照組**：同一批合成圖＋真實截圖，比「專用 4B 微調」vs「通用大模型」在牆上的 F1 與成本。
3. **把「驗證器迴圈」變成 RL 的密集獎勵**：Pencil Puzzle Bench 的 repo 已經示範用逐步驗證器接 GRPO；
   我們的 env 也能逐步檢查（mask 就是）——這接到 [RL 報告](2026-09-19_rl-where-next.md) §4.5。

---

## 6. 限制

- 沒有任何一手紀錄顯示 GPT-6 等級 agent 解過 LinkedIn Zip；§5 是推演。
- 「小畫家」來自社群示範的二手整理；「2048」查無一手出處。
- Quesma 與 Portal 都有「訓練資料可能含攻略」的疑慮，而且 Astra 的推理過程在 log 裡看不到——**不能據此判斷它是「推理」還是「寫程式」解的**。
- Pencil Puzzle Bench 的數字是 2026-03 的模型（GPT-5.2、Opus 4.6），不是 GPT-6 Astra。

## 7. 出處（查證 2026-09-19）

- OpenAI, *GPT-6 Astra: A new generation of intelligence* — <https://openai.com/index/gpt-6-astra/>
- CNBC, *OpenAI announces rollout of GPT-6 Astra model*（2026-09-03）— <https://www.cnbc.com/2026/09/03/open-ai-astra-gpt-6-cyber.html>
- Tom's Hardware, Portal 實驗 — <https://www.tomshardware.com/tech-industry/artificial-intelligence/openais-gpt-6-astra-model-autonomously-completes-portal-in-24-hours-feat-cost-just-usd571-in-tokens>
- Quesma, *GPT-6 Astra solves puzzles*（2026-09-14）— <https://quesma.com/blog/gpt-6-astra-solves-puzzles/>
- Waugh, *Pencil Puzzle Bench*（2026-03-02）— <https://arxiv.org/abs/2603.02119>、<https://github.com/approximatelabs/pencil-puzzle-bench>
- 小畫家示範的二手整理 — <https://happycapy.ai/blog/gpt-6-astra-wants-your-mouse>、<https://magiccreator.ai/astra>
- Claude computer use 工具文件 — <https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool>
