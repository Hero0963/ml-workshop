# src/lm_course/chat.py
"""From a base model to a chat model (lesson 11): conversation format, loss masks for SFT,
a toy arithmetic task with a verifiable answer, and an engine that runs a calculator tool.

The special tokens and the masking rule follow nanochat's design (MIT licence): user text,
tool outputs and the <|assistant_start|> marker are context (mask 0); everything the assistant
writes, including <|assistant_end|> and tool calls, is trained on (mask 1).
"""

import ast
import operator
import random
import re
from dataclasses import dataclass

import torch

from lm_course.model import GPT
from lm_course.sampling import next_token_probs, sample_from
from lm_course.tokenizer import BPETokenizer
from lm_course.training import IGNORE_INDEX

SPECIAL_TOKENS = [
    "<|bos|>",  # starts every document / conversation
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",  # the assistant calls the calculator tool
    "<|python_end|>",
    "<|output_start|>",  # the tool's answer, inserted by the engine, never trained on
    "<|output_end|>",
]

Message = dict  # {"role": "user" | "assistant", "content": str | list[dict]}


def render_conversation(
    tokenizer: BPETokenizer, messages: list[Message], max_tokens: int | None = None
) -> tuple[list[int], list[int]]:
    """Token ids and a 0/1 mask of the same length (1 = the model is trained to produce it).

    Assistant content is either a string or a list of parts {"type": "text" | "python" |
    "python_output", "text": ...}.
    """
    special = tokenizer.special_id
    ids: list[int] = []
    mask: list[int] = []

    def add(tokens: list[int] | int, trained: int) -> None:
        tokens = [tokens] if isinstance(tokens, int) else tokens
        ids.extend(tokens)
        mask.extend([trained] * len(tokens))

    add(special("<|bos|>"), 0)
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"message {i} should come from the {expected}")
        content = message["content"]
        if expected == "user":
            add(special("<|user_start|>"), 0)
            add(tokenizer.encode(content), 0)
            add(special("<|user_end|>"), 0)
            continue
        add(special("<|assistant_start|>"), 0)
        parts = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        for part in parts:
            text_ids = tokenizer.encode(part["text"])
            if part["type"] == "text":
                add(text_ids, 1)
            elif part["type"] == "python":
                add(special("<|python_start|>"), 1)
                add(text_ids, 1)
                add(special("<|python_end|>"), 1)
            elif part["type"] == "python_output":
                add(special("<|output_start|>"), 0)
                add(text_ids, 0)
                add(special("<|output_end|>"), 0)
            else:
                raise ValueError(f"unknown part type {part['type']!r}")
        add(special("<|assistant_end|>"), 1)
    if max_tokens is not None:
        ids, mask = ids[:max_tokens], mask[:max_tokens]
    return ids, mask


def render_for_completion(
    tokenizer: BPETokenizer, messages: list[Message]
) -> list[int]:
    """The prompt the model sees at inference: the conversation so far + <|assistant_start|>."""
    ids, _ = render_conversation(tokenizer, messages)
    return ids + [tokenizer.special_id("<|assistant_start|>")]


def sft_batch(
    rendered: list[tuple[list[int], list[int]]],
    pad_id: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and targets for a batch of rendered conversations, right-padded.

    target[t] = ids[t + 1] if the mask of token t + 1 is 1, else -1 (ignored by the loss).
    """
    length = max(len(ids) for ids, _ in rendered) - 1
    inputs = torch.full((len(rendered), length), pad_id, dtype=torch.long)
    targets = torch.full((len(rendered), length), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, mask) in enumerate(rendered):
        n = len(ids) - 1
        inputs[row, :n] = torch.tensor(ids[:-1])
        target = torch.tensor(ids[1:])
        target[torch.tensor(mask[1:]) == 0] = IGNORE_INDEX
        targets[row, :n] = target
    return inputs.to(device), targets.to(device)


# ---------------------------------------------------------------------------------------------
# A task with a checkable answer: adding two numbers
# ---------------------------------------------------------------------------------------------

QUESTION_TEMPLATES = [
    "What is {a} + {b}?",
    "What is {a} plus {b}?",
    "Can you add {a} and {b}?",
    "Please compute {a} + {b}.",
    "How much is {a} plus {b}?",
]
# the answer style the user asks for (lesson 11 §2.4, lab 11 §3)
STYLE_SUFFIX = {
    "direct": "",
    "steps": " Think step by step.",
    "tool": " Use the calculator.",
}


@dataclass(frozen=True)
class AdditionProblem:
    a: int
    b: int

    @property
    def answer(self) -> int:
        return self.a + self.b

    def question(self, style: str = "direct", rng: random.Random | None = None) -> str:
        template = (rng or random.Random(self.a * 7919 + self.b)).choice(
            QUESTION_TEMPLATES
        )
        return template.format(a=self.a, b=self.b) + STYLE_SUFFIX[style]

    def steps(self) -> str:
        """Column addition written out digit by digit, least significant digit first:
        "7 + 5 = 12, write 2 carry 1. 4 + 8 + 1 = 13, write 3 carry 1. ..."."""
        width = max(len(str(self.a)), len(str(self.b)))
        a_digits = str(self.a).zfill(width)[::-1]
        b_digits = str(self.b).zfill(width)[::-1]
        parts, carry = [], 0
        for i, (x, y) in enumerate(zip(a_digits, b_digits)):
            total = int(x) + int(y) + carry
            lhs = f"{x} + {y}" if i == 0 else f"{x} + {y} + {carry}"
            parts.append(f"{lhs} = {total}, write {total % 10} carry {total // 10}.")
            carry = total // 10
        if carry:
            parts.append(f"Write the carry {carry}.")
        return " ".join(parts)

    def reply(self, style: str = "direct") -> str | list[dict]:
        if style == "direct":
            return f"{self.a} + {self.b} = {self.answer}."
        if style == "steps":
            return f"{self.steps()} So {self.a} + {self.b} = {self.answer}."
        if style == "tool":
            return [
                {"type": "python", "text": f"{self.a}+{self.b}"},
                {"type": "python_output", "text": str(self.answer)},
                {"type": "text", "text": f"{self.a} + {self.b} = {self.answer}."},
            ]
        raise ValueError(f"unknown style {style!r}")

    def conversation(
        self, style: str = "direct", rng: random.Random | None = None
    ) -> list[Message]:
        return [
            {"role": "user", "content": self.question(style, rng)},
            {"role": "assistant", "content": self.reply(style)},
        ]


def addition_problems(
    num: int, max_digits: int, seed: int, exclude: set[tuple[int, int]] | None = None
) -> list[AdditionProblem]:
    """Distinct problems with operands of up to ``max_digits`` digits."""
    rng = random.Random(seed)
    exclude = exclude or set()
    seen: set[tuple[int, int]] = set()
    problems = []
    limit = 10**max_digits
    while len(problems) < num:
        a, b = rng.randrange(limit), rng.randrange(limit)
        if (a, b) in seen or (a, b) in exclude:
            continue
        seen.add((a, b))
        problems.append(AdditionProblem(a, b))
    return problems


def extract_answer(text: str) -> int | None:
    """The last integer in a reply ("... So 19 + 23 = 42." -> 42)."""
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else None


def addition_reward(problem: AdditionProblem, reply: str) -> float:
    return 1.0 if extract_answer(reply) == problem.answer else 0.0


# ---------------------------------------------------------------------------------------------
# Story requests, so the model keeps its pretraining skill
# ---------------------------------------------------------------------------------------------

STORY_TOPICS = [
    "dog", "cat", "ball", "tree", "bird", "park", "cake", "boat", "flower", "friend",
    "toy", "garden", "rain", "sun", "fish", "car", "moon", "box", "hat", "bear",
]  # fmt: skip
STORY_TEMPLATES = ["Tell me a story about a {t}.", "Write a short story with a {t}.",
                   "Can you tell me a story about a {t}?"]  # fmt: skip


def story_conversations(
    stories: list[str], max_words: int = 120, seed: int = 0
) -> list[list[Message]]:
    """User asks for a story about a topic word; the reply is a TinyStories story that
    contains that word (short stories only, to keep sequences short)."""
    rng = random.Random(seed)
    conversations = []
    for story in stories:
        if len(story.split()) > max_words:
            continue
        words = set(re.findall(r"[a-z]+", story.lower()))
        topics = [t for t in STORY_TOPICS if t in words]
        if not topics:
            continue
        request = rng.choice(STORY_TEMPLATES).format(t=rng.choice(topics))
        conversations.append(
            [
                {"role": "user", "content": request},
                {"role": "assistant", "content": story},
            ]
        )
    return conversations


# ---------------------------------------------------------------------------------------------
# A calculator tool and a small inference engine that runs it
# ---------------------------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> str:
    """Evaluate integer arithmetic (+ - * // % and parentheses) without ``eval``."""

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    try:
        return str(walk(ast.parse(expression.strip(), mode="eval")))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return "error"


@dataclass
class ChatReply:
    tokens: list[
        int
    ]  # everything generated after <|assistant_start|>, including tool output
    text: str  # the visible reply (text parts only)
    tool_calls: list[tuple[str, str]]  # (expression, result)


class ChatEngine:
    """Generates one assistant turn with a KV cache and runs the calculator when the model
    writes <|python_start|> ... <|python_end|>: the result is forced into the sequence as
    <|output_start|> result <|output_end|>, and generation continues from there."""

    def __init__(self, model: GPT, tokenizer: BPETokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        s = tokenizer.special_id
        self.end = s("<|assistant_end|>")
        self.python_start, self.python_end = s("<|python_start|>"), s("<|python_end|>")
        self.output_start, self.output_end = s("<|output_start|>"), s("<|output_end|>")

    @torch.no_grad()
    def reply(
        self,
        messages: list[Message],
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        use_tools: bool = True,
        generator: torch.Generator | None = None,
    ) -> ChatReply:
        self.model.eval()
        device = next(self.model.parameters()).device
        prompt = render_for_completion(self.tokenizer, messages)
        budget = min(max_tokens, self.model.config.context_length - len(prompt))
        cache = self.model.new_kv_cache(1, len(prompt) + budget)
        logits = self.model(torch.tensor([prompt], device=device), kv_cache=cache)[
            :, -1
        ]
        out: list[int] = []
        forced: list[int] = []
        code: list[int] | None = None
        tool_calls = []
        visible: list[int] = []
        while len(out) < budget:
            if forced:
                token = forced.pop(0)
            else:
                token = int(
                    sample_from(
                        next_token_probs(logits, temperature, top_k), generator
                    )[0]
                )
            out.append(token)
            if token == self.end:
                break
            if token == self.python_start:
                code = []
            elif token == self.python_end and code is not None:
                expression = self.tokenizer.decode(code)
                result = calculator(expression) if use_tools else "error"
                tool_calls.append((expression, result))
                forced = [
                    self.output_start,
                    *self.tokenizer.encode(result),
                    self.output_end,
                ]
                code = None
            elif code is not None:
                code.append(token)
            elif not self.tokenizer.is_special(token) and not self._inside_output(out):
                visible.append(token)
            if len(prompt) + len(out) >= cache.max_length:
                break
            logits = self.model(torch.tensor([[token]], device=device), kv_cache=cache)[
                :, -1
            ]
        return ChatReply(out, self.tokenizer.decode(visible), tool_calls)

    def _inside_output(self, out: list[int]) -> bool:
        last_start = max(
            (i for i, t in enumerate(out) if t == self.output_start), default=-1
        )
        last_end = max(
            (i for i, t in enumerate(out) if t == self.output_end), default=-1
        )
        return last_start > last_end
