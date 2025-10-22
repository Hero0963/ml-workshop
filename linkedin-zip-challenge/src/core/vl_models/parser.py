# src/core/vl_models/parser.py

import base64
import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any


from pydantic_ai import Agent, BinaryContent
from pydantic_ai.providers.openai import OpenAIProvider


# 導入我們*確實*需要用來「後處理」的函式
try:
    from src.core.utils import parse_puzzle_layout
except ImportError as e:
    print("=" * 50)
    print(f"錯誤：無法導入 'parse_puzzle_layout' from 'src.core.utils'！ {e}")
    print("請確保 'src/core/utils.py' 檔案存在。")
    print("我們需要這個函式來進行後處理。")
    print("=" * 50)
    exit(1)


# --- 2. 定義 MLLM 的「輸出模型」 (新策略) ---
# 這是一個更簡單的模型，只要求 MLLM 輸出原始資料。


class WallPair(BaseModel):
    """
    使用 [cell1, cell2] 的格式，這對 MLLM 來說比 tuple 更容易生成。
    範例: [[0, 1], [1, 1]]
    """

    cell1: list[int] = Field(description="牆壁一側格子的 [row, col] 座標。")
    cell2: list[int] = Field(description="牆壁另一側相鄰格子的 [row, col] 座標。")


class SimplePuzzleOutput(BaseModel):
    """
    AI 只需要解析並回傳這個結構。
    """

    layout: list[list[str]] = Field(
        description="一個 2D 陣列，代表網格佈局。"
        "數字格子用 '01', '02' 等字串表示，空格子用 '  ' 表示。"
    )
    walls: list[WallPair] = Field(
        description="一個牆壁座標對的列表。"
        "如果 (0,1) 和 (1,1) 之間有牆，應回傳 {'cell1': [0, 1], 'cell2': [1, 1]}。"
    )


# --- 3. 硬編碼 Few-shot 範例 (新策略) ---
# 我們不再動態生成，而是直接寫死 'conftest.py' 中的標準答案，
# 這 100% 避免了 'KeyError'。

EXAMPLE_1_JSON = """
{
  "layout": [
    ["  ", "  ", "  ", "  ", "  ", "  "],
    ["  ", "01", "  ", "  ", "02", "  "],
    ["  ", "  ", "03", "04", "  ", "  "],
    ["  ", "  ", "06", "05", "  ", "  "],
    ["  ", "08", "  ", "  ", "07", "  "],
    ["  ", "  ", "  ", "  ", "  ", "  "]
  ],
  "walls": [
    {"cell1": [1, 0], "cell2": [1, 1]},
    {"cell1": [0, 1], "cell2": [1, 1]},
    {"cell1": [2, 1], "cell2": [2, 2]},
    {"cell1": [1, 3], "cell2": [2, 3]},
    {"cell1": [1, 4], "cell2": [2, 4]},
    {"cell1": [3, 3], "cell2": [3, 4]},
    {"cell1": [3, 1], "cell2": [4, 1]},
    {"cell1": [3, 2], "cell2": [4, 2]},
    {"cell1": [4, 4], "cell2": [4, 5]},
    {"cell1": [4, 4], "cell2": [5, 4]}
  ]
}
"""

EXAMPLE_2_JSON = """
{
  "layout": [
    ["  ", "  ", "05", "08", "  ", "  "],
    ["  ", "12", "  ", "  ", "09", "  "],
    ["04", "  ", "  ", "  ", "  ", "01"],
    ["  ", "  ", "06", "07", "  ", "  "],
    ["  ", "11", "  ", "  ", "10", "  "],
    ["03", "  ", "  ", "  ", "  ", "02"]
  ],
  "walls": []
}
"""

EXAMPLE_3_JSON = """
{
  "layout": [
    ["12", "  ", "11", "  ", "09", "  "],
    ["02", "  ", "01", "  ", "10", "  "],
    ["  ", "  ", "  ", "  ", "  ", "  "],
    ["  ", "  ", "  ", "  ", "  ", "  "],
    ["  ", "08", "  ", "07", "  ", "06"],
    ["  ", "03", "  ", "04", "  ", "05"]
  ],
  "walls": [
    {"cell1": [2, 1], "cell2": [3, 1]},
    {"cell1": [2, 2], "cell2": [3, 2]},
    {"cell1": [2, 3], "cell2": [3, 3]},
    {"cell1": [2, 4], "cell2": [3, 4]}
  ]
}
"""

# --- 4. 輔助函式 ---


def load_image_as_binary_content(image_path: Path) -> BinaryContent:
    """
    載入本地圖片檔案，並將其轉換為 pydantic-ai 所需的 BinaryContent 格式。
    """
    if not image_path.exists():
        raise FileNotFoundError(f"找不到圖片檔案： {image_path}")
    print(f"正在載入圖片： {image_path}...")
    with open(image_path, "rb") as f:
        content = f.read()
    encoded_data = base64.b64encode(content).decode("utf-8")
    media_type = f"image/{image_path.suffix.lstrip('.')}"
    return BinaryContent(data=encoded_data, media_type=media_type)


def post_process_data(mllm_output: SimplePuzzleOutput) -> dict[str, Any]:
    """
    後處理函式 ("我們後續再自己解")。
    將 MLLM 的簡單輸出轉換為 conftest/solver 所需的最終格式。
    """
    print("--- 正在執行 Python 後處理 ---")

    # 1. 使用 'src/core/utils.py' 中*已經存在*的函式
    final_puzzle_data = parse_puzzle_layout(mllm_output.layout)

    # 2. 轉換 MLLM 輸出的 'walls'
    wall_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for wall_pair in mllm_output.walls:
        cell1 = tuple(wall_pair.cell1)  # e.g., [0, 1] -> (0, 1)
        cell2 = tuple(wall_pair.cell2)  # e.g., [1, 1] -> (1, 1)

        # 使用 conftest.py 中的標準化格式
        standardized_wall = tuple(sorted((cell1, cell2)))
        wall_set.add(standardized_wall)

    # 3. 將 'walls' 加入字典
    final_puzzle_data["walls"] = wall_set

    print("--- 後處理完成 ---")
    return final_puzzle_data


# --- 5. 主要的解析函式 ---


def parse_puzzle_image_to_solver_format(
    target_image_path: Path, illustrations_dir: Path
) -> dict[str, Any] | None:
    """
    使用 pydantic-ai 和 Ollama (Qwen-VL) 進行多模態 Few-shot 解析。
    """
    model_name = os.environ.get("OLLAMA_MODEL_NAME", "qwen2.5vl:7b")
    provider_url = os.environ.get(
        "OLLAMA_PROVIDER_URL", "http://ollama_server:11434/v1"
    )
    print("--- 開始解析任務 (新策略) ---")
    print(f"模型 (Model): {model_name}")
    print(f"Ollama URL: {provider_url}")
    print(f"目標圖片 (Target): {target_image_path.name}")

    # 1. 準備圖片列表 (3 範例 + 1 目標)
    content_list: list[BinaryContent] = []
    try:
        content_list.append(
            load_image_as_binary_content(illustrations_dir / "puzzle_01.png")
        )
        content_list.append(
            load_image_as_binary_content(illustrations_dir / "puzzle_02.png")
        )
        content_list.append(
            load_image_as_binary_content(illustrations_dir / "puzzle_03.png")
        )
        content_list.append(load_image_as_binary_content(target_image_path))
    except FileNotFoundError as e:
        print(e)
        print(f"請確保所有範例圖片都在 '{illustrations_dir}' 資料夾中。")
        return None

    # 2. 準備 Few-shot 提示 (Prompt)
    prompt = f"""
    你是一個精確的 LinkedIn Zip 遊戲解析器。
    我將會提供 4 張圖片。前 3 張是範例，第 4 張是你要執行的任務。
    你的工作是嚴格依照範例的 JSON 格式，解析第 4 張圖片。
    座標系統：左上角為 (0, 0)，格式為 (row, col)。

    --- 範例 1 (Image 1) ---
    {EXAMPLE_1_JSON}

    --- 範例 2 (Image 2) ---
    {EXAMPLE_2_JSON}

    --- 範例 3 (Image 3) ---
    {EXAMPLE_3_JSON}

    --- 任務 (Image 4) ---
    現在，請分析第 4 張圖片 (即列表中的最後一張圖片)，並提供與上述範例 *完全相同* 格式的 JSON 輸出。
    """

    # 3. 初始化 Provider 和 Agent
    try:
        # 修正： Provider 移除 'model' 參數
        provider = OpenAIProvider(
            base_url=provider_url,
            api_key="ollama",
        )

        # 修正： 'model' 參數移到 Agent 的建構子中
        agent = Agent(
            provider=provider,
            output_type=SimplePuzzleOutput,
            model=model_name,  # <-- 參數移到這裡了
        )
    except Exception as e:
        print(f"初始化 OpenAIProvider 失敗： {e}")
        return None

    # 4. 執行 MLLM 解析
    print(f"正在將 {len(content_list)} 張圖片和 Few-shot 提示發送到 Ollama...")
    try:
        # 步驟 A: MLLM 回傳 SimplePuzzleOutput
        mllm_output: SimplePuzzleOutput = agent.run_sync(
            prompt=prompt, content=content_list
        )
        print("--- MLLM 解析成功！ ---")

        # 步驟 B: 我們自己後處理 ("後續再自己解")
        final_data = post_process_data(mllm_output)

        return final_data

    except Exception as e:
        print("--- 解析失敗 ---")
        print("執行 pydantic-ai agent 時發生錯誤：")
        print(f"{type(e).__name__}: {e}")
        print("\n請檢查：")
        print(f"1. Ollama 服務是否正在 {provider_url} 運行？")
        print(f"2. 名為 '{model_name}' 的模型是否已在 Ollama 中下載？")
        return None


# --- 6. 執行腳本 ---
if __name__ == "__main__":
    # 我們從 'src/core/vl_models/parser.py' 執行
    SCRIPT_DIR = Path(__file__).parent.resolve()
    # 專案根目錄 (往上 3 層: vl_models -> core -> src -> ROOT)
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"
    print(f"插圖資料夾路徑設定為: {ILLUSTRATIONS_DIR}")

    TARGET_IMAGE_PATH = ILLUSTRATIONS_DIR / "puzzle_02.png"

    if not TARGET_IMAGE_PATH.exists():
        print(f"錯誤：找不到目標圖片 {TARGET_IMAGE_PATH}")
        print("請檢查 'illustrations' 資料夾是否存在於專案根目錄。")
    else:
        # 執行主函式
        solver_ready_data = parse_puzzle_image_to_solver_format(
            target_image_path=TARGET_IMAGE_PATH, illustrations_dir=ILLUSTRATIONS_DIR
        )

        if solver_ready_data:
            print("\n--- 最終輸出 (Solver-Ready Dictionary) ---")

            # 字典不能用 .model_dump_json()，我們用 pprint 或 json.dumps
            # 為了能印出 set()，我們自訂一個轉換器
            def custom_serializer(obj):
                if isinstance(obj, set):
                    return sorted(list(obj))  # 轉換 set 為 sorted list
                if isinstance(obj, tuple):
                    return list(obj)  # 轉換 tuple 為 list
                return str(obj)

            print(json.dumps(solver_ready_data, indent=2, default=custom_serializer))

            # 驗證
            if TARGET_IMAGE_PATH.name == "puzzle_02.png":
                print("\n--- 驗證 (與 conftest.py 比較) ---")
                print(f"解析出的 1 號位置: {solver_ready_data['number_map'].get(1)}")
                # 假設 parse_puzzle_layout 返回的 key 是 'number_map'
                print(f"解析出的牆壁數量: {len(solver_ready_data['walls'])}")
                print("標準答案牆壁數量: 0")
        else:
            print("\n任務未完成。")
