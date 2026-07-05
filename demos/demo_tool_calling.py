"""
Tool Calling Demo for HVN (Issue #24)
=====================================

Demonstrates the tool-calling protocol:
  1. ToolRegistry — register tools with JSON schemas
  2. ToolCallParser — parse [TOOL_CALL] blocks from model output
  3. ToolExecutor — execute tools and format [TOOL_RESULT] blocks
  4. ConstrainedJSONDecoder — logit masking for JSON-constrained decoding
  5. ToolCallingSession — multi-turn agentic loop
  6. generate_with_tools — model generation with constrained decoding

This demo does NOT require a trained model — it tests the tool-calling
infrastructure (parsing, execution, constrained decoding) using mock data.

Usage:
    python demo_tool_calling.py
"""

import torch
# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---
from hvn.tools import (
    ToolRegistry,
    ToolCallParser,
    ToolExecutor,
    ConstrainedJSONDecoder,
    ToolCallingSession,
    create_demo_registry,
    generate_tool_training_data,
    calculator,
    web_search,
    string_reverse,
    word_count,
    TOOL_TOKENS,
    SPECIAL_TOKEN_STRINGS,
)
from train import CharTokenizer


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_tool_registry():
    """Test 1: ToolRegistry — register and look up tools."""
    section("Test 1: ToolRegistry")
    registry = create_demo_registry()

    print(f"Registered tools: {registry.list_names()}")
    assert registry.exists("calculator")
    assert registry.exists("web_search")
    assert registry.exists("string_reverse")
    assert registry.exists("word_count")
    assert not registry.exists("nonexistent_tool")

    # Test format_for_prompt
    prompt_text = registry.format_for_prompt()
    print(f"\nFormatted prompt:\n{prompt_text}")

    # Test validate_args
    assert registry.validate_args("calculator", {"expression": "2+2"})
    assert not registry.validate_args("calculator", {})  # missing required arg
    assert not registry.validate_args("nonexistent", {})

    print("\n✓ ToolRegistry tests passed")


def test_tool_call_parser():
    """Test 2: ToolCallParser — parse [TOOL_CALL] blocks."""
    section("Test 2: ToolCallParser")

    # Single tool call
    text1 = """Some text before.
[TOOL_CALL]
{"name": "calculator", "arguments": {"expression": "2+2"}}
[/TOOL_CALL]
Some text after."""

    calls = ToolCallParser.parse_tool_calls(text1)
    assert len(calls) == 1
    assert calls[0]["name"] == "calculator"
    assert calls[0]["arguments"]["expression"] == "2+2"
    print(f"Parsed call: {calls[0]}")

    # Multiple tool calls
    text2 = """[TOOL_CALL]
{"name": "calculator", "arguments": {"expression": "3*5"}}
[/TOOL_CALL]
[TOOL_RESULT]
15
[/TOOL_RESULT]
[TOOL_CALL]
{"name": "web_search", "arguments": {"query": "python"}}
[/TOOL_CALL]"""

    calls2 = ToolCallParser.parse_tool_calls(text2)
    assert len(calls2) == 2
    assert calls2[0]["name"] == "calculator"
    assert calls2[1]["name"] == "web_search"
    print(f"Parsed {len(calls2)} calls: {[c['name'] for c in calls2]}")

    # Parse tool results
    results = ToolCallParser.parse_tool_results(text2)
    assert len(results) == 1
    assert results[0] == "15"
    print(f"Parsed {len(results)} results: {results}")

    # Strip tool blocks
    stripped = ToolCallParser.strip_tool_blocks(text2)
    assert "[TOOL_CALL]" not in stripped
    assert "[TOOL_RESULT]" not in stripped
    print(f"Stripped text: '{stripped}'")

    # has_tool_call
    assert ToolCallParser.has_tool_call(text2)
    assert not ToolCallParser.has_tool_call("just regular text")

    # has_tool_call_start (partial generation)
    partial = "Some text [TOOL_CALL]\n{\"name\":"
    assert ToolCallParser.has_tool_call_start(partial)
    assert not ToolCallParser.has_tool_call_start(text2)  # all closed

    # Parse thinking blocks
    text3 = "[THINK]\nI need to calculate something.\n[/THINK]\nLet me use the calculator."
    thinking = ToolCallParser.parse_thinking(text3)
    assert len(thinking) == 1
    assert "calculate" in thinking[0]
    print(f"Parsed thinking: {thinking}")

    # Malformed JSON (should be skipped)
    text4 = "[TOOL_CALL]\n{invalid json}\n[/TOOL_CALL]"
    calls4 = ToolCallParser.parse_tool_calls(text4)
    assert len(calls4) == 0
    print(f"Malformed JSON correctly skipped: {len(calls4)} calls")

    print("\n✓ ToolCallParser tests passed")


def test_tool_executor():
    """Test 3: ToolExecutor — execute tools and format results."""
    section("Test 3: ToolExecutor")
    registry = create_demo_registry()
    executor = ToolExecutor(registry)

    # Calculator
    result = executor.execute({"name": "calculator", "arguments": {"expression": "2+2"}})
    print(f"calculator(2+2) → {result}")
    assert "4" in result

    # Web search
    result = executor.execute({"name": "web_search", "arguments": {"query": "python"}})
    print(f"web_search(python) → {result}")
    assert "Python" in result

    # String reverse
    result = executor.execute({"name": "string_reverse", "arguments": {"text": "hello"}})
    print(f"string_reverse(hello) → {result}")
    assert "olleh" in result

    # Word count
    result = executor.execute({"name": "word_count", "arguments": {"text": "the quick brown fox"}})
    print(f"word_count(the quick brown fox) → {result}")
    assert "4" in result

    # Nonexistent tool
    result = executor.execute({"name": "nonexistent", "arguments": {}})
    print(f"nonexistent tool → {result}")
    assert "not found" in result

    # Invalid args (missing required parameter)
    result = executor.execute({"name": "calculator", "arguments": {}})
    print(f"missing args → {result}")
    assert "invalid arguments" in result

    # Execute multiple
    results = executor.execute_all([
        {"name": "calculator", "arguments": {"expression": "10+20"}},
        {"name": "string_reverse", "arguments": {"text": "abc"}},
    ])
    print(f"\nExecute all:\n{results}")
    assert "30" in results
    assert "cba" in results

    print("\n✓ ToolExecutor tests passed")


def test_constrained_json_decoder():
    """Test 4: ConstrainedJSONDecoder — logit masking for JSON."""
    section("Test 4: ConstrainedJSONDecoder")

    # Build a simple char tokenizer with tool tokens — include chars NOT valid in JSON
    text = '{"name": "calculator", "arguments": {"expression": "2+2"}}~`@#$%^&*|<>?'
    chars = sorted(set(text + "0123456789abcdefghijklmnopqrstuvwxyz(){}[]:,. \n\t~`@#$%^&*|<>?"))
    id_to_token = {i: ch for i, ch in enumerate(chars)}
    token_to_id = {ch: i for i, ch in enumerate(chars)}

    # Add special tokens
    for st in SPECIAL_TOKEN_STRINGS:
        new_id = len(id_to_token)
        id_to_token[new_id] = st
        token_to_id[st] = new_id

    vocab_size = len(id_to_token)
    decoder = ConstrainedJSONDecoder(vocab_size=vocab_size, id_to_token=id_to_token)

    # Before activation: no mask
    assert not decoder.active
    mask = decoder.get_allowed_ids()
    assert mask is None

    # After activation: mask should be non-None
    decoder.activate()
    assert decoder.active
    mask = decoder.get_allowed_ids()
    assert mask is not None
    print(f"Vocab size: {vocab_size}")
    print(f"Allowed tokens when active: {mask.sum().item()} / {vocab_size}")
    assert mask.sum().item() > 0  # some tokens should be allowed
    assert mask.sum().item() < vocab_size  # not all tokens allowed (non-JSON chars excluded)

    # Deactivate
    decoder.deactivate()
    assert not decoder.active

    print("\n✓ ConstrainedJSONDecoder tests passed")


def test_tool_calling_session():
    """Test 5: ToolCallingSession — multi-turn agentic loop (mock model)."""
    section("Test 5: ToolCallingSession (with mock model)")

    # Create a simple tokenizer that handles special tokens
    class MockTokenizer:
        def __init__(self):
            # Build vocab from common chars + special tokens
            base = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?_-/()[]{}@#$%^&*+=\\\"'\n\t"
            self.char_to_id = {}
            for ch in sorted(set(base)):
                self.char_to_id[ch] = len(self.char_to_id)
            # Add special tokens
            for st in SPECIAL_TOKEN_STRINGS:
                self.char_to_id[st] = len(self.char_to_id)
            self.id_to_char = {v: k for k, v in self.char_to_id.items()}
            self.vocab_size = len(self.char_to_id)

        def encode(self, text):
            ids = []
            i = 0
            while i < len(text):
                matched = False
                for st in sorted(SPECIAL_TOKEN_STRINGS, key=len, reverse=True):
                    if text[i:i+len(st)] == st and st in self.char_to_id:
                        ids.append(self.char_to_id[st])
                        i += len(st)
                        matched = True
                        break
                if matched:
                    continue
                ch = text[i]
                if ch in self.char_to_id:
                    ids.append(self.char_to_id[ch])
                i += 1
            return ids

        def decode(self, ids):
            return "".join(self.id_to_char.get(i, "") for i in ids)

    mock_tokenizer = MockTokenizer()

    # Create a mock model that emits a tool call on first turn, then an answer
    class MockModel:
        """Mock model that emits a [TOOL_CALL] on first turn, then an answer."""
        def __init__(self, tokenizer):
            self.call_count = 0
            self.vocab_size = tokenizer.vocab_size
            self.tokenizer = tokenizer

        @torch.no_grad()
        def generate_incremental(self, prompt_ids, max_new_tokens=200, temperature=0.8, top_k=10, device=None):
            self.call_count += 1
            if self.call_count == 1:
                # First call: emit a tool call
                response = '[TOOL_CALL]\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n[/TOOL_CALL]'
            else:
                # Second call: emit the final answer
                response = "2+2 = 4"
            response_ids = self.tokenizer.encode(response)
            return list(prompt_ids) + response_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mock_model = MockModel(mock_tokenizer)
    registry = create_demo_registry()
    executor = ToolExecutor(registry)

    session = ToolCallingSession(
        model=mock_model,
        tokenizer=mock_tokenizer,
        executor=executor,
        device=device,
        max_turns=3,
        temperature=0.7,
        top_k=10,
        max_tokens_per_turn=200,
    )

    result = session.run("What is 2+2?")
    print(f"Answer: {result['answer']}")
    print(f"Tool calls: {len(result['tool_calls'])}")
    print(f"Tool results: {len(result['tool_results'])}")
    print(f"Turns: {result['n_turns']}")

    # The mock model should have made at least 1 tool call
    assert len(result["tool_calls"]) >= 1
    assert result["tool_calls"][0]["name"] == "calculator"

    print("\n✓ ToolCallingSession tests passed")


def test_training_data_generation():
    """Test 6: generate_tool_training_data — synthetic training examples."""
    section("Test 6: Training Data Generation")
    examples = generate_tool_training_data(num_examples=10)
    print(f"Generated {len(examples)} examples")

    assert len(examples) == 10
    for ex in examples:
        assert "prompt" in ex
        assert "response" in ex
        assert "[TOOL_CALL]" in ex["response"]
        assert "[/TOOL_CALL]" in ex["response"]
        assert "[TOOL_RESULT]" in ex["response"]
        assert "[/TOOL_RESULT]" in ex["response"]

    # Print first example
    print(f"\n--- Example 1 ---")
    print(f"Prompt: {examples[0]['prompt'][:200]}...")
    print(f"Response: {examples[0]['response'][:200]}...")

    print("\n✓ Training data generation tests passed")


def test_special_tokens_in_tokenizer():
    """Test 7: CharTokenizer with special tool tokens."""
    section("Test 7: CharTokenizer with Special Tool Tokens")
    text = "Hello world 123"
    tokenizer = CharTokenizer(text, add_tool_tokens=True)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens}")

    # Check special tokens are in vocab
    assert "[TOOL_CALL]" in tokenizer.char_to_id
    assert "[/TOOL_CALL]" in tokenizer.char_to_id
    assert "[TOOL_RESULT]" in tokenizer.char_to_id
    assert "[/TOOL_RESULT]" in tokenizer.char_to_id
    assert "[THINK]" in tokenizer.char_to_id
    assert "[/THINK]" in tokenizer.char_to_id

    # Encode text with special tokens
    text_with_tools = "Hello [TOOL_CALL]{\"name\": \"calc\"}[/TOOL_CALL] world"
    ids = tokenizer.encode(text_with_tools)
    print(f"Encoded: {ids}")

    # Decode back
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
    assert "[TOOL_CALL]" in decoded
    assert "[/TOOL_CALL]" in decoded

    # Test backward compatibility (no tool tokens)
    tokenizer_plain = CharTokenizer(text, add_tool_tokens=False)
    assert "[TOOL_CALL]" not in tokenizer_plain.char_to_id
    assert tokenizer_plain.vocab_size < tokenizer.vocab_size

    print("\n✓ CharTokenizer special token tests passed")


def test_model_generate_with_tools():
    """Test 8: Model.generate_with_tools — constrained decoding support."""
    section("Test 8: Model.generate_with_tools")
    from hvn.core import HamiltonianVortexNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build a small model
    text = "Hello world [TOOL_CALL]{\"name\":\"calc\"}[/TOOL_CALL] test 123"
    tokenizer = CharTokenizer(text, add_tool_tokens=True)
    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_layers=1,
        n_heads=2,
        chunk_size=16,
    ).to(device)
    model.eval()

    # Test generate_with_tools with stop tokens
    stop_ids = [tokenizer.char_to_id.get("[/TOOL_CALL]", -1)]
    stop_ids = [s for s in stop_ids if s >= 0]

    prompt = "Hello "
    prompt_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}' → {prompt_ids}")
    print(f"Stop token IDs: {stop_ids}")

    # Generate with stop tokens
    generated = model.generate_with_tools(
        prompt_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=10,
        device=device,
        stop_token_ids=stop_ids,
    )
    output = tokenizer.decode(generated)
    print(f"Generated (with stop): {output}")
    assert len(generated) > len(prompt_ids)  # should have generated something

    # Test with allowed_token_ids (constrained decoding)
    # Only allow alphanumeric + JSON chars
    allowed_chars = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]":,. \n\t')
    allowed_ids = [tokenizer.char_to_id[c] for c in allowed_chars if c in tokenizer.char_to_id]
    print(f"Allowed token IDs: {len(allowed_ids)} tokens")

    generated_constrained = model.generate_with_tools(
        prompt_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=10,
        device=device,
        allowed_token_ids=allowed_ids,
    )
    output_constrained = tokenizer.decode(generated_constrained)
    print(f"Generated (constrained): {output_constrained}")

    # Test without constraints (should behave like generate_incremental)
    generated_free = model.generate_with_tools(
        prompt_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=10,
        device=device,
    )
    output_free = tokenizer.decode(generated_free)
    print(f"Generated (free): {output_free}")

    print("\n✓ Model.generate_with_tools tests passed")


def main():
    print("=" * 60)
    print("  HVN Tool Calling Demo (Issue #24)")
    print("  Structured Output and Function Invocation")
    print("=" * 60)

    # Run all tests
    test_tool_registry()
    test_tool_call_parser()
    test_tool_executor()
    test_constrained_json_decoder()
    test_tool_calling_session()
    test_training_data_generation()
    test_special_tokens_in_tokenizer()
    test_model_generate_with_tools()

    section("All Tests Passed!")
    print("""
Summary:
  ✓ ToolRegistry — register tools with JSON schemas
  ✓ ToolCallParser — parse [TOOL_CALL] blocks from model output
  ✓ ToolExecutor — execute tools and format [TOOL_RESULT] blocks
  ✓ ConstrainedJSONDecoder — logit masking for JSON-constrained decoding
  ✓ ToolCallingSession — multi-turn agentic loop
  ✓ Training data generation — synthetic tool-calling examples
  ✓ CharTokenizer — special tool tokens support
  ✓ Model.generate_with_tools — constrained decoding support

Files created/modified:
  ✓ hvn/tools.py (new) — ToolRegistry, ToolCallParser, ToolExecutor, etc.
  ✓ hvn/tokenizer.py — special tokens in BPE trainer
  ✓ hvn/core/model.py — generate_with_tools() method
  ✓ train.py — CharTokenizer special tokens, --tool_training flag
  ✓ generate.py — --tools flag for tool-augmented generation
  ✓ demo_tool_calling.py (new) — this demo

Preserves:
  ✓ O(1) inference — no KV cache, no architectural change
  ✓ Tool calling is a prompting/training feature, not an architectural change
""")


if __name__ == "__main__":
    main()