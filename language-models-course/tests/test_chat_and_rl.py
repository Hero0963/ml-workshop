# tests/test_chat_and_rl.py
import math
import re

import pytest
import torch

from lm_course.chat import (
    HELD_OUT_TOPICS,
    SPECIAL_TOKENS,
    STORY_TOPICS,
    AdditionProblem,
    ChatEngine,
    addition_problems,
    addition_reward,
    calculator,
    extract_answer,
    mentions_topic,
    render_conversation,
    render_for_completion,
    sft_batch,
    story_request,
)
from lm_course.model import GPT, llama_style_config
from lm_course.rl import (
    completion_logprobs,
    dpo_loss,
    group_advantages,
    pass_at_k,
    policy_gradient_loss,
)
from lm_course.tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    text = ["What is 12 + 30? The answer is 42. Tell me a story about a dog."] * 10
    return BPETokenizer.train(text, 300, special_tokens=SPECIAL_TOKENS)


def test_only_assistant_tokens_are_trained(tok: BPETokenizer) -> None:
    conversation = AdditionProblem(12, 30).conversation()
    ids, mask = render_conversation(tok, conversation)
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    context = tok.decode([i for i, m in zip(ids, mask) if not m])
    assert trained == "12 + 30 = 42.<|assistant_end|>"
    assert context.startswith("<|bos|><|user_start|>") and context.endswith(
        "<|assistant_start|>"
    )


def test_tool_output_is_context_but_the_call_is_trained(tok: BPETokenizer) -> None:
    ids, mask = render_conversation(
        tok, AdditionProblem(12, 30).conversation(style="tool")
    )
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    assert (
        trained == "<|python_start|>12+30<|python_end|>12 + 30 = 42.<|assistant_end|>"
    )
    assert "<|output_start|>42<|output_end|>" in tok.decode(ids)


def test_roles_must_alternate(tok: BPETokenizer) -> None:
    with pytest.raises(ValueError):
        render_conversation(tok, [{"role": "assistant", "content": "hi"}])


def test_sft_targets_are_shifted_and_masked(tok: BPETokenizer) -> None:
    rendered = [
        render_conversation(tok, AdditionProblem(a, 1).conversation()) for a in (5, 123)
    ]
    x, y = sft_batch(rendered, pad_id=0)
    ids, mask = rendered[1]
    assert x.shape[1] == max(len(r[0]) for r in rendered) - 1
    for t in range(len(ids) - 1):
        assert x[1, t] == ids[t]
        assert y[1, t] == (ids[t + 1] if mask[t + 1] else -1)
    assert (y[0, len(rendered[0][0]) - 1 :] == -1).all()  # padding is ignored


def test_completion_prompt_ends_with_assistant_start(tok: BPETokenizer) -> None:
    prompt = render_for_completion(tok, [{"role": "user", "content": "What is 1 + 2?"}])
    assert prompt[-1] == tok.special_id("<|assistant_start|>")


def test_addition_problems_answers_and_rewards() -> None:
    problems = addition_problems(50, max_digits=2, seed=0)
    assert len({(p.a, p.b) for p in problems}) == 50
    assert all(0 <= p.a < 100 and 0 <= p.b < 100 for p in problems)
    held_out = addition_problems(20, 2, seed=1, exclude={(p.a, p.b) for p in problems})
    assert not {(p.a, p.b) for p in held_out} & {(p.a, p.b) for p in problems}
    p = AdditionProblem(19, 23)
    assert addition_reward(p, "19 + 23 = 42.") == 1.0
    assert addition_reward(p, "19 + 23 = 41.") == 0.0
    assert addition_reward(p, p.reply("steps")) == 1.0
    assert extract_answer("no digits") is None


@pytest.mark.parametrize("a, b", [(47, 85), (3, 83), (999, 1), (0, 0), (560, 72)])
def test_step_by_step_reply_is_correct_column_addition(a: int, b: int) -> None:
    p = AdditionProblem(a, b)
    steps = p.steps()
    written = [int(d) for d in re.findall(r"write (\d)", steps)]
    carry = re.search(r"Write the carry (\d)", steps)
    digits = written + ([int(carry.group(1))] if carry else [])
    assert int("".join(str(d) for d in reversed(digits))) == a + b
    assert p.question("steps").endswith("Think step by step.")
    assert p.question("tool").endswith("Use the calculator.")


def test_topic_reward() -> None:
    assert mentions_topic("cat", "The Cats played.") == 1.0
    assert mentions_topic("cat", "A category of things.") == 0.0
    assert not set(HELD_OUT_TOPICS) & set(STORY_TOPICS)
    assert story_request("kite")[0]["content"] == "Tell me a story about a kite."


def test_calculator_is_safe() -> None:
    assert calculator("12+30") == "42"
    assert calculator("(3 + 4) * 5 - 6 // 4") == "34"
    assert calculator("__import__('os')") == "error"
    assert calculator("1/0") == "error"


def test_engine_runs_the_tool_when_the_model_calls_it(tok: BPETokenizer) -> None:
    """A fake model that always wants to emit the next token of a fixed script."""
    s = tok.special_id
    # while the engine forces <|output_start|> 4 2 <|output_end|>, the model's own
    # predictions are ignored: the script holds placeholders for those steps
    forced = 2 + len(tok.encode("42"))
    script = [s("<|python_start|>"), *tok.encode("12+30"), s("<|python_end|>"), *[0] * forced,
              *tok.encode(" ok"), s("<|assistant_end|>")]  # fmt: skip
    config = llama_style_config(
        vocab_size=tok.vocab_size, context_length=64, n_layer=1, n_head=2, d_model=16
    )
    model = GPT(config)

    class Scripted(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner, self.step, self.config = model, 0, config

        def forward(self, idx: torch.Tensor, kv_cache=None) -> torch.Tensor:  # noqa: ANN001
            self.inner(idx, kv_cache=kv_cache)  # keep the cache bookkeeping real
            logits = torch.full((1, idx.shape[1], tok.vocab_size), -1e9)
            logits[0, -1, script[min(self.step, len(script) - 1)]] = 0
            self.step += 1
            return logits

        def new_kv_cache(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            return self.inner.new_kv_cache(*args, **kwargs)

    fake = Scripted()
    reply = ChatEngine(fake, tok).reply(
        [{"role": "user", "content": "What is 12 + 30?"}]
    )
    assert reply.tool_calls == [("12+30", "42")]
    assert "<|output_start|>42<|output_end|>" in tok.decode(reply.tokens)
    assert reply.text == " ok"


def test_group_advantages() -> None:
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(
        group_advantages(rewards), torch.tensor([0.5, -0.5, -0.5, 0.5])
    )
    assert (
        group_advantages(torch.ones(4)).abs().sum() == 0
    )  # all equal: no learning signal


def test_on_policy_loss_gradient_is_reinforce() -> None:
    logits = torch.randn(3, 5, requires_grad=True)
    logprobs = torch.log_softmax(logits, dim=-1)[:, :4]  # pretend 4 completion tokens
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1.0]])
    adv = torch.tensor([1.0, -0.5, 0.25])
    (g1,) = torch.autograd.grad(
        policy_gradient_loss(logprobs, adv, mask), logits, retain_graph=True
    )
    reinforce = -(logprobs * adv[:, None] * mask).sum() / mask.sum()
    (g2,) = torch.autograd.grad(reinforce, logits)
    torch.testing.assert_close(g1, g2)


def test_clipping_stops_the_update_past_the_trust_region() -> None:
    logprobs = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-1.0]])  # ratio = e > 1.2
    loss = policy_gradient_loss(
        logprobs, torch.tensor([1.0]), torch.ones(1, 1), old, clip_eps=0.2
    )
    (grad,) = torch.autograd.grad(loss, logprobs)
    assert grad.item() == 0.0


def test_completion_logprobs_match_a_manual_computation(tok: BPETokenizer) -> None:
    config = llama_style_config(
        vocab_size=tok.vocab_size, context_length=32, n_layer=1, n_head=2, d_model=16
    )
    model = GPT(config).eval()
    prompt, completions = [1, 2, 3], [[4, 5], [6, 7, 8]]
    logprobs, mask = completion_logprobs(model, prompt, completions, pad_id=0)
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    with torch.no_grad():
        full = torch.log_softmax(
            model(torch.tensor([prompt + completions[1]])), dim=-1
        )[0]
    expected = [full[2, 6], full[3, 7], full[4, 8]]
    torch.testing.assert_close(logprobs[1].detach(), torch.stack(expected))
    hot, _ = completion_logprobs(model, prompt, completions, pad_id=0, temperature=2.0)
    with torch.no_grad():
        full_hot = torch.log_softmax(
            model(torch.tensor([prompt + completions[1]])) / 2.0, dim=-1
        )[0]
    torch.testing.assert_close(hot[1, 0].detach(), full_hot[2, 6])


def test_dpo_loss() -> None:
    zeros = torch.zeros(4)
    assert dpo_loss(zeros, zeros, zeros, zeros).item() == pytest.approx(math.log(2))
    better = dpo_loss(torch.full((4,), 2.0), zeros, zeros, zeros, beta=1.0)
    assert better.item() == pytest.approx(-math.log(1 / (1 + math.exp(-2))))


def test_pass_at_k() -> None:
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - (7 * 6) / (10 * 9))
    assert pass_at_k(10, 3.0, 2) == pass_at_k(10, 3, 2)  # a sum of 0/1 rewards
