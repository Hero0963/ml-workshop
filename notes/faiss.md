# Facebook AI Similarity Search (FAISS) 筆記

## 緣起
現代向量檢索（Vector Search）場景普遍涉及大量高維資料，傳統線性比對 O(N) 無法應付大規模需求，FAISS 作為主流 ANN（Approximate Nearest Neighbor）庫，支援多種索引與量化技術，廣泛應用於圖像、語意、文件等相似度檢索場景。

## 主要量化（Quantization）技術
### 1. Scalar Quantization
- **概念**：直接將向量元素量化為較低精度（如 FP16、int8），以壓縮儲存空間與加速運算。
- **操作**：將 float32 → float16，空間減半，對查詢精度影響較小。

### 2. Product Quantization (PQ)
- **核心流程**：
    1. 將 d 維向量 \( v \) 拆分為 m 個子向量 \( (w_1, ..., w_m) \)，每個維度 \( d' = d/m \)。
    2. 對每個子空間用 K-means 訓練 codebook，取得 k 個 centroids。
    3. 向量僅儲存每段「最接近的 centroid index」（即 PQ code），大幅壓縮記憶體與查詢成本。
    4. 查詢時以碼本還原近似距離，實現高速 ANN。
   
#### 查詢距離計算：Asymmetric Distance Computation (ADC)
- 查詢向量 \(q\) 也分割成 \(m\) 段，對每段用 codebook 找最近的 centroid（產生查詢 PQ code）。
- 資料庫每筆 \(x\) 都有自己的 PQ code（每段的 index）。
- 計算 \(q\) 與 \(x\) 的近似距離，不需重建原始向量，  
  只需把查詢每段 centroid 與資料點每段 centroid 之間的距離「查表」加總即可。
- 這種查詢法稱為**Asymmetric Distance Computation (ADC)**，查詢效率極高。
- **工程實作**：  
    - 訓練（`index.train(xb)`）需用大量原始資料。
    - 新資料只需用既有 codebook 量化，不必重訓。
- **常用場景**：圖像搜尋、語意檢索等大規模 embedding 查詢。

### 3. Additive Quantization (AQ)
- **核心流程**：
    1. 用多組 codebook \( (C_1, ..., C_m) \)，每組可選 1 個 centroid。
    2. 向量近似為多個 centroids 的和：\( v \approx c_1 + c_2 + ... + c_m \)。
    3. 屬於多層殘差編碼（multi-layer residual），精度較 PQ 高，但編碼與查詢成本較大。
- **係數說明**：傳統 AQ 每個 centroid 係數僅為 0 或 1（選或不選），如加權則屬於其他進階量化。

---

## FAISS 索引類型與查詢模式
- **IndexFlatL2**：暴力全量比對（精確查詢），僅適合小資料集。
- **IndexIVFFlat / IndexIVFPQ**：倒排分群 + PQ 壓縮，適合大規模資料。
- **IndexHNSWFlat**：支援 HNSW 圖索引，兼具高查準率與低延遲。

---

## 橫向對比：其他 ANN 演算法/向量資料庫  
| 演算法        | 核心思想                    | 優點                | 缺點/限制                  | 代表庫         |
|---------------|----------------------------|---------------------|----------------------------|----------------|
| PQ/AQ         | 量化壓縮 + 分群查詢         | 省空間、查詢快      | 須訓練 codebook，查詢近似  | FAISS, Annoy   |
| HNSW          | 分層小世界圖搜尋            | 高查準、快、可擴展  | 佔空間，插入較慢           | FAISS, NMSLIB, Qdrant, Milvus |
| Annoy         | 多顆隨機樹切割              | 查詢快、實作簡單    | 精度較低、佔空間           | Spotify Annoy  |
| LSH           | 雜湊投影                   | 查詢快、參數少      | 精度波動大                 | scikit-learn   |

- **主流向量資料庫**：
    - **Milvus、Qdrant、Weaviate**：多支援 HNSW / IVF / PQ / Flat，少部份支援 AQ。
    - **Pinecone**：主打 HNSW（自家優化版），支援 metadata filter。
    - **pgvector (PostgreSQL 插件)**：主要 Flat/HNSW，不含 PQ/AQ 量化。

---

## 縱向對比：各索引/量化方式優劣
| 方法      | 空間效率 | 查詢速度 | 查詢精度 | 適用規模  |
|-----------|----------|----------|----------|-----------|
| Flat      | 低       | 慢       | 100%     | 小型資料  |
| IVF       | 高       | 快       | 90~99%   | 中/大型   |
| PQ        | 超高     | 極快     | 85~98%   | 超大規模  |
| AQ        | 高       | 快       | 90~99%   | 超大規模  |
| HNSW      | 中       | 快       | 95~100%  | 大規模    |

---


## 參考資料

1. **FAISS 官方資源**
   - [FAISS 官網](https://faiss.ai/)
   - [THE FAISS LIBRARY（官方論文, arXiv 2024）](https://arxiv.org/pdf/2401.08281)



---