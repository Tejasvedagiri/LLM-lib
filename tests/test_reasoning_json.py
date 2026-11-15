import pytest

from llm_lib import ThinkFilter, extract_json, has_json, strip_reasoning


def run(chunks):
    f = ThinkFilter()
    out = [p for c in chunks for p in f.feed(c)] + f.flush()
    content = "".join(t for k, t in out if k == "content")
    reasoning = "".join(t for k, t in out if k == "reasoning")
    return content, reasoning


def test_think_block_split_across_chunks_is_reasoning():
    content, reasoning = run(["<thi", "nk>pondering", " deeply</th", "ink>\n\nHello", " there"])
    assert content == "Hello there"
    assert reasoning == "pondering deeply"


def test_thinking_tag_variant_and_case():
    content, reasoning = run(["<THINKING>x</thinking>answer"])
    assert (content, reasoning) == ("answer", "x")


def test_text_that_only_looks_like_a_tag_start_is_kept():
    content, _ = run(["a <", "b and <th", "e end"])
    assert content == "a <b and <the end"


def test_unclosed_think_stays_reasoning():
    content, reasoning = run(["<think>never finished"])
    assert content == "" and reasoning == "never finished"


def test_strip_reasoning():
    assert strip_reasoning("<think>a</think> hi ") == "hi"
    assert strip_reasoning("ok <think>cut off") == "ok"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n["good","bad"]\n```', ["good", "bad"]),
        ('Sure! Here: {"x": [1, 2]} hope it helps', {"x": [1, 2]}),
        ('[{"a": 1}, {"b": 2}]', [{"a": 1}, {"b": 2}]),
        ("no json here", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_has_json():
    assert has_json('reply: {"ok": true}')
    assert not has_json("nope")
    assert not has_json("42")
