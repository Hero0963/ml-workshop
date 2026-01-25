# LeetCode 1039 延伸討論 (Extensions)

## 緣起
紀錄與朋友討論解法時產生的疑問，答疑與釐清。

## 前置作業
- **題目**: [1039. Minimum Score Triangulation of Polygon](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/)
- **解法**: 可用 `dp`，並從中得到切分方式

## 補充：
- **貪心算法會失效的例子**:
  - 求最大時： `v = [14,3,12,7,4,6,1,9,11]`
  - 求最小時： `v = [1,4,2,5,3]`

圖解在 [illustrations/lc_1039](./illustrations/lc_1039)

## 關於反例的構造思考
- **簡化例子** (求最大時)： `v = [9, 12, 7, 4, 6, 1]`
- **構造重點**：讓 `第 3, 4 大` 的權重很接近，但是都比 `第 1, 2 大` 小很多。