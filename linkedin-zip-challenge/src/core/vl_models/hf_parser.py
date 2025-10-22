# src/core/vl_models/hf_parser.py
import torch
import re
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any

# --- 1. 導入 HF Transformers 依賴 ---
try:
    # 修正：使用 Qwen3-VL 專用的類別
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("=" * 50)
    print("錯誤：缺少 'transformers' 或 'torch' 套件。")
    print("請先執行：")
    print("uv add --upgrade transformers torch accelerate sentencepiece")
    print("=" * 50)
    exit(1)

# --- 2. 導入我們自己的後處理函式 ---
try:
    from src.core.utils import parse_puzzle_layout
except ImportError as e:
    print("=" * 50)
    print(f"錯誤：無法導入 'parse_puzzle_layout' from 'src.core.utils'！ {e}")
    print("請確保 'src.core/utils.py' 檔案存在。")
    print("=" * 50)
    exit(1)


# --- 3. Pydantic 模型 (不變) ---
class WallPair(BaseModel):
    cell1: list[int] = Field(description="牆壁一側格子的 [row, col] 座標。")
    cell2: list[int] = Field(description="牆壁另一側相鄰格子的 [row, col] 座標。")


class SimplePuzzleOutput(BaseModel):
    layout: list[list[str]] = Field(
        description="一個 2D 陣列，代表網格佈局。"
        "數字格子用 '01', '02' 等字串表示，空格子用 '  ' 表示。"
    )
    walls: list[WallPair] = Field(description="一個牆壁座標對的列表。")


# --- 4. Few-shot 範例 JSON (不變) ---
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

# --- 5. 輔助函式 (不變) ---


def extract_json_block(text: str) -> str | None:
    """從模型輸出的 markdown 中提取 JSON 區塊"""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 降級：如果找不到 markdown 區塊，就直接找 { ... }
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index : end_index + 1]

    print(f"錯誤：在回應中找不到 JSON 區塊。\n回應：{text}")
    return None


def post_process_data(mllm_output: SimplePuzzleOutput) -> dict[str, Any]:
    """
    將 MLLM 的輸出轉換為求解器所需的最終 Python 字典格式。

    1. 使用 `parse_puzzle_layout` 處理佈局，自動生成 grid, num_map, 和 blocked_cells。
    2. 處理 MLLM 輸出的 walls 列表，並將其添加到 puzzle 字典中。
    """
    print("--- 正在執行 Python 後處理 ---")

    # 步驟 1: 從 layout 解析基礎 puzzle 結構
    # 這一步會正確地從 'xx' 產生 'blocked_cells'
    final_puzzle_data = parse_puzzle_layout(mllm_output.layout)

    # 步驟 2: 處理 MLLM 輸出的 walls
    wall_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    if mllm_output.walls:
        for wall_pair in mllm_output.walls:
            # Pydantic 已經驗證過 cell1 和 cell2 是 list[int]
            cell1 = tuple(wall_pair.cell1)
            cell2 = tuple(wall_pair.cell2)
            # 標準化牆壁順序 (cell1 < cell2)，以確保唯一性
            standardized_wall = tuple(sorted((cell1, cell2)))
            wall_set.add(standardized_wall)

    # 步驟 3: 將處理好的 walls 集合更新到最終的字典中
    final_puzzle_data["walls"] = wall_set

    print("--- 後處理完成 ---")
    return final_puzzle_data


# --- 6. 載入函式 (修正) ---


def load_model_and_processor(model_name: str):
    """
    載入 Hugging Face 模型和處理器 (Processor)。
    """
    print(f"正在從 Hugging Face 載入 '{model_name}'...")
    print("這可能需要幾分鐘，並且會下載數 GB 的模型檔案...")

    # 1. 修正：使用 AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 2. 修正：使用 Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",  # 自動使用 GPU
        trust_remote_code=True,
        # 4B 模型使用 bfloat16 應足夠裝入 16GB VRAM
        dtype=torch.bfloat16,
    ).eval()

    print("--- 模型和處理器載入完成 ---")
    return model, processor


# --- 7. 主要的解析函式 (HF 版本 - 修正) ---


def parse_puzzle_image_hf(
    model,
    processor,  # 修正：傳入 processor
    target_image_path: Path,
    illustrations_dir: Path,
) -> dict[str, Any] | None:
    """
    使用 HF Transformers (標準流程) 執行 Few-shot 推理。
    """
    print("--- 開始 HF 解析任務 ---")
    print(f"目標圖片 (Target): {target_image_path.name}")

    # 1. 準備 HF Chat Template 格式的 messages
    messages = [
        # 範例 1
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(illustrations_dir / "puzzle_01.png")},
                {"type": "text", "text": "--- 範例 1 (Image 1) ---"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": EXAMPLE_1_JSON}]},
        # 範例 2
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(illustrations_dir / "puzzle_02.png")},
                {"type": "text", "text": "--- 範例 2 (Image 2) ---"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": EXAMPLE_2_JSON}]},
        # 範例 3
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(illustrations_dir / "puzzle_03.png")},
                {"type": "text", "text": "--- 範例 3 (Image 3) ---"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": EXAMPLE_3_JSON}]},
        # 任務
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(target_image_path)},
                {
                    "type": "text",
                    "text": "--- 任務 (Image 4) ---\n"
                    "現在，請分析第 4 張圖片 (即列表中的最後一張圖片)，"
                    "並提供與上述範例 *完全相同* 格式的 JSON 輸出。",
                },
            ],
        },
    ]

    # 2. 轉換為 Tokenizer 格式 (使用你範例 中的 pipeline)
    print("正在將 4 張圖片和提示套用 Chat Template...")
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 3. 執行模型推理
        print("正在執行 model.generate... (這應該會快很多)")
        generated_ids = model.generate(
            **inputs, max_new_tokens=2048
        )  # 增加 tokens 以容納 JSON

        # 4. 從輸出中移除輸入 token
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 5. 解碼
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        response = output_text[0]  # 我們只有一個 prompt，所以取第一個
        print("--- MLLM 推理成功！ ---")

        # 6. 提取 JSON
        json_str = extract_json_block(response)
        if not json_str:
            print(f"錯誤：無法從模型回應中提取 JSON。\n原始回應：{response}")
            return None

        mllm_json_data = json.loads(json_str)

        # 7. 用 Pydantic 驗證
        mllm_output = SimplePuzzleOutput(**mllm_json_data)

        # 8. 後處理
        final_data = post_process_data(mllm_output)
        return final_data

    except Exception as e:
        print("--- 推理或解析失敗 ---")
        print("執行 HF pipeline 時發生錯誤：")
        print(f"{type(e).__name__}: {e}")
        return None


# --- 8. 執行腳本 ---
if __name__ == "__main__":
    # 1. 設定模型和路徑

    # === 關鍵修改：更換為 4B-Thinking 模型 ===
    HF_MODEL_NAME = "Qwen/Qwen3-VL-4B-Thinking"
    # ========================================

    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"

    print(f"插圖資料夾路徑設定為: {ILLUSTRATIONS_DIR}")
    TARGET_IMAGE_PATH = ILLUSTRATIONS_DIR / "puzzle_02.png"

    if not TARGET_IMAGE_PATH.exists():
        print(f"錯誤：找不到目標圖片 {TARGET_IMAGE_PATH}")
    else:
        try:
            # 2. 載入模型 (使用修正後的函式)
            model, processor = load_model_and_processor(HF_MODEL_NAME)

            # 3. 執行解析 (使用修正後的函式)
            solver_ready_data = parse_puzzle_image_hf(
                model=model,
                processor=processor,
                target_image_path=TARGET_IMAGE_PATH,
                illustrations_dir=ILLUSTRATIONS_DIR,
            )

            if solver_ready_data:
                print("\n--- 最終輸出 (Solver-Ready Dictionary) ---")

                def custom_serializer(obj):
                    """自定義 JSON 序列化函式，用於印出 set 和 tuple"""
                    if isinstance(obj, set):
                        return sorted(list(obj))
                    if isinstance(obj, tuple):
                        return list(obj)
                    return str(obj)

                print(
                    json.dumps(solver_ready_data, indent=2, default=custom_serializer)
                )

            else:
                print("\n任務未完成。")

        except Exception as e:
            print(f"執行主腳本時發生嚴重錯誤： {e}")
            if "CUDA out of memory" in str(e):
                print("!!! 錯誤： GPU 記憶體不足！ (理論上 4B 模型不應發生) !!!")
