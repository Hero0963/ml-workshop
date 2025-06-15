# Hierarchical Navigable Small Worlds (HNSW) 筆記

## 緣起
在向量資料庫（Vector Database）中，我們常需進行大量高維向量的最近鄰搜尋（Nearest Neighbor Search），距離函數多採用 L2 距離或餘弦相似度（Cosine Similarity）。傳統的精確搜尋在大規模資料上會因線性掃描（O(N)）而效能不足，故引入近似最近鄰（Approximate Nearest Neighbor，ANN）算法以換取可接受的準確率和更快的查詢速度。

## HNSW 原理概述
- **多層結構（Hierarchical）**：將資料點組織於多層圖（Layered Graph），越高層節點越少，連結越稀疏；底層節點最多、連結最密集。  
- **可導覽小世界圖（Navigable Small World）**：透過小世界圖的分層連結實現快速跳躍式搜尋，兼具全局瀏覽性與地方聚類性。  
- **近似最近鄰（ANN）**：允許在可控的召回率（Recall）和延遲（Latency）之間進行權衡，通常能達到 90%+ 的召回率。

## 核心流程
1. **節點分層（Random Level Assignment）**  
   - 每個新節點 \(u\) 依照指數分布隨機決定其最大層數 \(L_u\)：  
     
   
   \[ P(L_u \ge k) = p^k,\quad (0<p<1) \]  
   - 層級越高節點越稀少；所有節點至少位於層 0。

2. **鄰居列表（Neighbors List）**  
   - 參數 \(M\)（最大度）決定每層最多連結的鄰居數。  
   - 節點 \(u\) 在每層 \(l \le L_u\) 維護其鄰居列表 \(\mathcal{N}_l(u)\)（大小上限為 \(M\)）。

3. **索引構建（Insertion / Construction）**  
   - 使用參數 `ef_construction` 控制在插入階段的候選列表規模。  
   - 流程：  
   
   - 於最高層 \(L_u\) 起始，從全局入口節點執行貪心搜尋（Greedy Search）定位最接近 \(u\) 的位置。  
   - 逐層向下至層 0，於每層：  
     1. 執行多路候選搜索（Multi-path Search），收集候選集合 \(\mathcal{C}_l\)。  
     2. 從 \(\mathcal{C}_l\) 選出前 \(M\) 個最近節點作為 \(\mathcal{N}_l(u)\)。  
     3. 對於候選中的各老節點 \(v\)，將 \(u\) 插入 \(\mathcal{N}_l(v)\)，如超過 \(M\) 則踢除最遠節點。  

4. **最近鄰查詢（Search）**  
   - 參數 `ef_search` 控制候選池（Candidate Pool）大小，上限常設為 \(4M\)。  
   - 流程：  
   
   - 從最高層全局入口節點執行貪心搜尋至定位 Query 節點或最近位置。  
   - 下降至下一層，以當前定位作為入口，重複貪心搜尋（Greedy Search）。  
   - 在層 0 開始動態 kNN 搜索（Multi-path Best-First Search）：  
     - 初始化候選隊列（Priority Queue）和已訪問集合（Visited Set）。  
     - 重複從隊列中彈出距離最小節點並擴展其鄰居，直到彈出次數達 `ef_search` 或最遠候選距離不再改進。  
     - 最終返回前 \(K\) 個最近鄰。
   
   - **無狀態（Stateless）**：每次查詢均從靜態索引出發，使用全新臨時結構，不會修改原圖。

5. **節點刪除（Deletion）**  
   - 直接物理刪除需 \(O(N	imes M)\) 時間，實務多採**標記刪除（Tombstone）**：  
     1. 查詢時跳過已標記節點；  
     2. 積累至一定比例後批次重建索引清除。  
   - 若需高精度可在刪除時觸發本地重連（Local Repair），於每層為受影響節點補充新的鄰居。

## 參數說明與調優
| 參數 | 含義 | 建議範圍／取值 |
|-----|------|--------------|
| \(M\) | 每層最大鄰居數 | 8、16、32、64 |
| ef_construction | 構建階段候選列表大小 | \(2M \sim 4M\) |
| ef_search | 查詢階段候選池大小 | \(2M \sim 4M\)（可根據召回率需求調整）|
| 距離函數 | L2 距離或 Cosine 相似度 | 視資料特性選擇 |

- **召回率 vs. 延遲**：  
  - \(ef\_search = 2M\) 常能達 90%+ 召回；  
  - 增加至 \(4M\) 可突破 95%+；  
  - 進一步提升需考量延遲增長情況。

## 複雜度分析
- **插入**：\(O(L_u 	imes (ef\_construction \log ef\_construction + M))\)  
- **查詢**：\(O(ef\_search \log ef\_search)\)（層間貪心定位疊加多路搜索）

---
*以上筆記整合 HNSW 核心概念、算法流程與工程實踐，助於快速理解與應用。*

## 參考連結
1. [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/pdf/1603.09320)
2. [Pinecone HNSW explain](https://www.pinecone.io/learn/series/faiss/hnsw/)

