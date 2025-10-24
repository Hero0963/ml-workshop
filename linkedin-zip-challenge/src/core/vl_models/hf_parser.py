# src/core/vl_models/hf_parser.py
import torch
import re
import json
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Any

# --- 1. 導入 HF Transformers 依賴 ---
try:
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


# --- 3. Pydantic 模型 (已加入 thinking_process) ---
class WallPair(BaseModel):
    cell1: list[int] = Field(description="牆壁一側格子的 [row, col] 座標。 கூ")
    cell2: list[int] = Field(description="牆壁另一側相鄰格子的 [row, col] 座標 கூ")


class SimplePuzzleOutput(BaseModel):
    thinking_process: str = Field(
        description="The model's step-by-step reasoning process."
    )
    layout: list[list[str]] = Field(
        description="一個 2D 陣列，代表網格佈局。"
        "數字格子用 '01', '02' 等字串表示，空格子用 '  ' 表示 கூ"
    )
    walls: list[WallPair] = Field(description="一個牆壁座標對的列表 கூ")


# --- 4. Few-shot 範例 JSON (已加入 thinking_process) ---
EXAMPLE_1_JSON = """
{
  "thinking_process": "The user provided the JSON for this example.",
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
  "thinking_process": "The user provided the JSON for this example.",
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
  "thinking_process": "The user provided the JSON for this example.",
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
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index : end_index + 1]
    print(f"錯誤：在回應中找不到 JSON 區塊。\n回應：{text}")
    return None


def post_process_data(mllm_output: SimplePuzzleOutput) -> dict[str, Any]:
    """將 MLLM 的輸出轉換為求解器所需的最終 Python 字典格式。"""
    print("--- 正在執行 Python 後處理 ---")
    final_puzzle_data = parse_puzzle_layout(mllm_output.layout)
    wall_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    if mllm_output.walls:
        for wall_pair in mllm_output.walls:
            cell1 = tuple(wall_pair.cell1)
            cell2 = tuple(wall_pair.cell2)
            standardized_wall = tuple(sorted((cell1, cell2)))
            wall_set.add(standardized_wall)
    final_puzzle_data["walls"] = wall_set
    print("--- 後處理完成 ---")
    return final_puzzle_data


# --- 6. 載入函式 (不變) ---


def load_model_and_processor(model_name: str):
    """
    載入 Hugging Face 模型和處理器 (Processor)。
    """
    print(f"正在從 Hugging Face 載入 '{model_name}'...")
    print("這可能需要幾分鐘，並且會下載數 GB 的模型檔案...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).eval()
    print("--- 模型和處理器載入完成 ---")
    return model, processor


# --- 7. 主要的解析函式 (重構為新策略) ---


def build_puzzle_prompt() -> str:
    """建立一個包含 Few-shot 範例的純文字提示。"""
    return f"""
You are an expert Zip puzzle analyzer. Your task is to analyze the provided image and respond with ONLY a single JSON object in a markdown code block.

The JSON object must have three keys: "thinking_process", "layout", and "walls".
- "thinking_process": Describe your step-by-step analysis of the image.
- "layout": A 2D array of strings representing the grid.
- "walls": A list of objects, where each object represents a wall between two cells.

Here are some examples of the required output format.

--- EXAMPLE 1 ---
```json
{EXAMPLE_1_JSON.strip()}
```

--- EXAMPLE 2 ---
```json
{EXAMPLE_2_JSON.strip()}
```

--- EXAMPLE 3 ---
```json
{EXAMPLE_3_JSON.strip()}
```

--- TASK ---
Now, analyze the new image provided. First, describe your step-by-step analysis in the `thinking_process` field. Then, based on your analysis, fill in the `layout` and `walls` fields. Provide the final output as a single JSON object in the exact same format as the examples.
"""


def parse_puzzle_image_hf(
    model,
    processor,
    target_image_path: Path,
) -> dict[str, Any] | None:
    """(新策略) 使用純文字 Few-shot 提示和單張目標圖片進行推理。"""
    print("--- 開始 HF 解析任務 (新策略) ---")
    print(f"目標圖片 (Target): {target_image_path.name}")

    # 1. 建立純文字提示
    prompt = build_puzzle_prompt()

    # 2. 準備只包含單張圖片和提示的 messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(target_image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    print("正在將單張圖片和提示套用 Chat Template...")
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        print("正在執行 model.generate...")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        response = output_text[0]
        print(f"--- MLLM 推理成功！---\n原始回應:\n{response}")

        json_str = extract_json_block(response)
        if not json_str:
            return None

        mllm_json_data = json.loads(json_str)
        mllm_output = SimplePuzzleOutput(**mllm_json_data)
        final_data = post_process_data(mllm_output)
        return final_data

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        print(f"--- 推理或解析失敗 ---\n執行 HF pipeline 時發生錯誤： {e}")
        return None
    except Exception as e:
        print(f"--- 推理或解析失敗 ---\n執行 HF pipeline 時發生嚴重錯誤： {e}")
        return None


# --- 8. 執行腳本 ---
if __name__ == "__main__":
    HF_MODEL_NAME = "Qwen/Qwen3-VL-4B-Thinking"
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"

    print(f"插圖資料夾路徑設定為: {ILLUSTRATIONS_DIR}")
    TARGET_IMAGE_PATH = ILLUSTRATIONS_DIR / "puzzle_04.png"

    if not TARGET_IMAGE_PATH.exists():
        print(f"錯誤：找不到目標圖片 {TARGET_IMAGE_PATH}")
    else:
        try:
            model, processor = load_model_and_processor(HF_MODEL_NAME)
            solver_ready_data = parse_puzzle_image_hf(
                model=model,
                processor=processor,
                target_image_path=TARGET_IMAGE_PATH,
            )

            if solver_ready_data:
                print("\n--- 最終輸出 (Solver-Ready Dictionary) ---")

                def custom_serializer(obj):
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
