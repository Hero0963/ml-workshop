# Roadmap — board-game-rl

> **新 session 第一站**：現況、下一步、已定案不要再重開的決策。
> 做完一件事就更新本檔（現況＋下一步），細節寫進 [`dev_log.md`](dev_log.md)。
> Last Updated: 2026-09-24

## 現況（2026-09-24）

- **Stage 1（井字遊戲）完成**：Alpha-Beta（完美解）、Q-Learning（不敗）、DQN（對 Random 當後手約 2.5% 敗率）、Random，都能在 Gradio 對弈。
  各自的方法與數字見 [`project_guide.md`](project_guide.md) 與 [`reports/`](reports/)。
- 2026-04-11 做完 DQN 之後暫停約 5.5 個月；**2026-09-24 重新建立基線**：`uv sync --locked` 成功，原本 28 個測試全過。
- **正在做：M1 純 MCTS**，計畫書 [`plans/2026-09-24_m1-pure-mcts.md`](plans/2026-09-24_m1-pure-mcts.md)。
  為什麼先做它、被否決的選項，見 [`notes/2026-09-24-review-and-m1-decisions.md`](notes/2026-09-24-review-and-m1-decisions.md)。
  2026-09-24：規則介面、MCTS agent、API 與 Gradio 接線都完成，**本人已在 Gradio 跟 MCTS 下過棋**；`pytest` 69 passed。
  剩 S3 模擬次數掃描、S4 根節點統計、S6 教材與文件；接續方式見 [`handover.md`](handover.md)「下次開工」。
- **Future work：同一套 API、兩種前端**（仿 thread-the-grid 的 Gradio＋Svelte），見 [`handover.md`](handover.md)「Future Work」。

## 下一步

1. **M1 純 MCTS 收尾（井字遊戲）** — 已可對弈；剩 S3 量「模擬次數 → 棋力」曲線、S4 根節點統計、S6 教材。
2. **M2 AlphaZero-lite（井字遊戲）** — 策略／價值雙頭網路 ＋ PUCT ＋ 自我對弈；井字遊戲用 CPU 就夠。
3. **M3 Connect Four** — 第一個查表放不下的遊戲：驗證 `games/`／`agents/` 分層真的能換遊戲，也是 MCTS／AlphaZero 開始有優勢的地方。

## 已定案（不要再重開）

| 日期 | 決策 | 理由／出處 |
|---|---|---|
| 2026-03-28 | Q-Learning 用 Hybrid 對手（Random 開局 → Alpha-Beta 收尾）訓練 | 只跟 Alpha-Beta 練只會走到約 165 個狀態；見 [`dev_log.md`](dev_log.md) 2026-03-28 |
| 2026-04-11 | DQN 不再為井字遊戲調優 | 練習目的已達成；可用的改進手法記在 [`handover.md`](handover.md)「已知限制」 |
| 2026-09-24 | 順序固定為 **純 MCTS → AlphaZero → 換遊戲**，一次只改一個變數 | [`notes/2026-09-24-review-and-m1-decisions.md`](notes/2026-09-24-review-and-m1-decisions.md) §2 |
| 2026-09-24 | 雙前端（Gradio＋Svelte、同一套 API）先列為 future work，不排進 M1 | [`handover.md`](handover.md)「Future Work」；評估在 notes §6 |
| 2026-09-24 | 文件分工：`docs/` 放原理教材、`ai-collab/notes/` 放討論與取捨、`reports/` 放實驗、`dev_log.md` 放流水帳 | [`notes/README.md`](notes/README.md) |
