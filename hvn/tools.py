"""
Tool Calling Protocol for HVN
=============================

Provides structured tool-call support for the Hamiltonian Vortex Network.

This module implements:
  - ``ToolRegistry``  — register tools with JSON schemas; look up by name.
  - ``ToolCallParser`` — parse [TOOL_CALL]...[/TOOL_CALL] blocks from model output.
  - ``ToolExecutor``   — execute registered tools and format [TOOL_RESULT] blocks.
  - ``ConstrainedJSONDecoder`` — logit-mask helper for JSON-constrained decoding.
  - ``ToolCallingSession`` — multi-turn agentic loop: generate → parse → execute → feed back.

Tool calling is a **prompting/training feature**, not an architectural change.
The HVN model itself is unchanged — O(1) inference and no KV cache are preserved.

Special tokens (added to the tokenizer):
  - ``[TOOL_CALL]``   — marks the start of a tool-call JSON block
  - ``[/TOOL_CALL]``  — marks the end of a tool-call JSON block
  - ``[TOOL_RESULT]`` — marks the start of a tool-result block
  - ``[/TOOL_RESULT]``— marks the end of a tool-result block
  - ``[THINK]``       — marks the start of a reasoning block
  - ``[/THINK]``      — marks the end of a reasoning block

Usage (standalone):
    from hvn.tools import ToolRegistry, ToolCallParser, ToolExecutor, ToolCallingSession

    registry = ToolRegistry()
    registry.register("calculator", calculator_fn, schema={...})
    executor = ToolExecutor(registry)
    session = ToolCallingSession(model, tokenizer, executor, device)
    result = session.run("What is 2+2?", max_turns=5)
"""

from __future__ import annotations

import json
import re
import math
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Special tool-calling tokens
# ---------------------------------------------------------------------------

TOOL_TOKENS: Dict[str, str] = {
    "tool_call_start": "[TOOL_CALL]",
    "tool_call_end": "[/TOOL_CALL]",
    "tool_result_start": "[TOOL_RESULT]",
    "tool_result_end": "[/TOOL_RESULT]",
    "think_start": "[THINK]",
    "think_end": "[/THINK]",
}

# All special token strings (for tokenizer registration)
SPECIAL_TOKEN_STRINGS: List[str] = list(TOOL_TOKENS.values())


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry of callable tools with JSON schemas.

    Each tool has:
      - A unique name (string).
      - A callable that accepts keyword arguments matching the schema.
      - A JSON schema dict describing the expected parameters.

    The schema is used for:
      1. Injecting tool descriptions into prompts during training/inference.
      2. Validating parsed tool-call arguments.
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable,
        schema: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        """Register a tool.

        Parameters
        ----------
        name : str
            Unique tool name (e.g. "calculator", "web_search").
        func : callable
            The function to execute when this tool is called.
        schema : dict, optional
            JSON-schema-style dict describing parameters.
            Example::

                {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
        description : str
            Human-readable description of what the tool does.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = {
            "func": func,
            "schema": schema or {},
            "description": description,
        }

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def exists(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_names(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def format_for_prompt(self) -> str:
        """Format all registered tools as a text block for prompt injection.

        This is injected into the prompt context so the model learns
        when and how to call tools.

        Example output::

            Available tools:
            1. calculator: A simple calculator
               Schema: {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
            2. web_search: Search the web
               Schema: {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        """
        if not self._tools:
            return "No tools available."
        lines = ["Available tools:"]
        for i, (name, info) in enumerate(self._tools.items(), 1):
            desc = info["description"] or "No description"
            schema_json = json.dumps(info["schema"], separators=(",", ":"))
            lines.append(f"  {i}. {name}: {desc}")
            lines.append(f"     Schema: {schema_json}")
        return "\n".join(lines)

    def validate_args(self, name: str, args: Dict[str, Any]) -> bool:
        """Basic validation of arguments against the tool's schema.

        Checks that all required parameters are present.  Does not do
        full JSON schema validation (type checking) — that's left to the
        tool function itself.
        """
        tool = self._tools.get(name)
        if tool is None:
            return False
        schema = tool.get("schema", {})
        required = schema.get("required", [])
        for param in required:
            if param not in args:
                return False
        return True


# ---------------------------------------------------------------------------
# Tool Call Parser
# ---------------------------------------------------------------------------

class ToolCallParser:
    """Parse [TOOL_CALL]...[/TOOL_CALL] blocks from model output text.

    The model is expected to emit tool calls as::

        [TOOL_CALL]
        {"name": "calculator", "arguments": {"expression": "2+2"}}
        [/TOOL_CALL]

    Multiple tool calls can appear in a single generation.  Text outside
    [TOOL_CALL] blocks is treated as normal model output (the answer or
    reasoning).

    This parser also extracts [THINK]...[/THINK] reasoning blocks.
    """

    # Regex patterns
    TOOL_CALL_RE = re.compile(
        r"\[TOOL_CALL\]\s*(.*?)\s*\[/TOOL_CALL\]",
        re.DOTALL,
    )
    TOOL_RESULT_RE = re.compile(
        r"\[TOOL_RESULT\]\s*(.*?)\s*\[/TOOL_RESULT\]",
        re.DOTALL,
    )
    THINK_RE = re.compile(
        r"\[THINK\]\s*(.*?)\s*\[/THINK\]",
        re.DOTALL,
    )

    @classmethod
    def parse_tool_calls(cls, text: str) -> List[Dict[str, Any]]:
        """Extract all tool-call blocks from text.

        Returns
        -------
        calls : list[dict]
            Each dict has keys:
              - ``name``: tool name
              - ``arguments``: dict of arguments
              - ``raw``: the raw JSON string
            Returns empty list if no tool calls found or JSON is invalid.
        """
        calls = []
        for match in cls.TOOL_CALL_RE.finditer(text):
            raw_json = match.group(1).strip()
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict) and "name" in parsed:
                    calls.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                        "raw": raw_json,
                    })
            except json.JSONDecodeError:
                # Skip malformed tool calls
                continue
        return calls

    @classmethod
    def parse_tool_results(cls, text: str) -> List[str]:
        """Extract all [TOOL_RESULT] blocks from text."""
        results = []
        for match in cls.TOOL_RESULT_RE.finditer(text):
            results.append(match.group(1).strip())
        return results

    @classmethod
    def parse_thinking(cls, text: str) -> List[str]:
        """Extract all [THINK]...[/THINK] reasoning blocks."""
        return [m.group(1).strip() for m in cls.THINK_RE.finditer(text)]

    @classmethod
    def strip_tool_blocks(cls, text: str) -> str:
        """Remove all [TOOL_CALL], [TOOL_RESULT], and [THINK] blocks.

        Returns the remaining text (the model's direct answer to the user).
        """
        text = cls.TOOL_CALL_RE.sub("", text)
        text = cls.TOOL_RESULT_RE.sub("", text)
        text = cls.THINK_RE.sub("", text)
        return text.strip()

    @classmethod
    def has_tool_call(cls, text: str) -> bool:
        """Check if the text contains at least one [TOOL_CALL] block."""
        return bool(cls.TOOL_CALL_RE.search(text))

    @classmethod
    def has_tool_call_start(cls, text: str) -> bool:
        """Check if the text contains an unclosed [TOOL_CALL] (partial generation).

        This is used during generation to detect when the model has started
        emitting a tool call but hasn't closed it yet — at that point we
        switch to JSON-constrained decoding.
        """
        # Count occurrences of start vs end
        starts = text.count(TOOL_TOKENS["tool_call_start"])
        ends = text.count(TOOL_TOKENS["tool_call_end"])
        return starts > ends


# ---------------------------------------------------------------------------
# Tool Executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Execute tool calls and format results.

    Given a parsed tool call (name + arguments), this class:
      1. Looks up the tool in the registry.
      2. Validates the arguments.
      3. Calls the tool function.
      4. Formats the result as a [TOOL_RESULT] block.
    """

    def __init__(self, registry: ToolRegistry, max_result_len: int = 500):
        self.registry = registry
        self.max_result_len = max_result_len

    def execute(self, tool_call: Dict[str, Any]) -> str:
        """Execute a single tool call and return a [TOOL_RESULT] block.

        Parameters
        ----------
        tool_call : dict
            Parsed tool call with keys ``name`` and ``arguments``.

        Returns
        -------
        result_block : str
            A formatted [TOOL_RESULT]...[/TOOL_RESULT] block containing
            either the tool's return value or an error message.
        """
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        # Check if tool exists
        if not self.registry.exists(name):
            error = f"Error: tool '{name}' not found. Available: {self.registry.list_names()}"
            return self._format_result(error)

        # Validate arguments
        if not self.registry.validate_args(name, args):
            error = f"Error: invalid arguments for '{name}'. Required: {self.registry.get(name)['schema'].get('required', [])}"
            return self._format_result(error)

        # Execute
        try:
            tool_info = self.registry.get(name)
            result = tool_info["func"](**args)
            # Serialize result to string
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, separators=(",", ":"))
            else:
                result_str = str(result)
            # Truncate if too long
            if len(result_str) > self.max_result_len:
                result_str = result_str[:self.max_result_len] + "...[truncated]"
            return self._format_result(result_str)
        except Exception as e:
            error = f"Error executing '{name}': {type(e).__name__}: {e}"
            return self._format_result(error)

    def execute_all(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Execute multiple tool calls and return concatenated results."""
        if not tool_calls:
            return ""
        result_blocks = [self.execute(tc) for tc in tool_calls]
        return "\n".join(result_blocks)

    def _format_result(self, result_str: str) -> str:
        """Format a result string as a [TOOL_RESULT] block."""
        return f"{TOOL_TOKENS['tool_result_start']}\n{result_str}\n{TOOL_TOKENS['tool_result_end']}"


# ---------------------------------------------------------------------------
# Constrained JSON Decoding (logit masking)
# ---------------------------------------------------------------------------

class ConstrainedJSONDecoder:
    """Logit mask helper for JSON-constrained decoding.

    When the model emits ``[TOOL_CALL]``, the subsequent tokens should form
    valid JSON.  This class provides a logit mask that restricts the
    vocabulary to only characters valid in the current JSON context.

    This is a character-level constraint (works with CharTokenizer).
    For BPE tokenizers, it provides a best-effort mask at the token level.

    The constraint is a simple state machine:
      - ``START``: only allow ``{`` or whitespace
      - ``KEY``: allow alphanumeric, ``"``, ``_``
      - ``COLON``: allow ``:`` or whitespace
      - ``VALUE_START``: allow ``"``, digits, ``{``, ``[``, ``t`` (true), ``f`` (false), ``n`` (null)
      - ``STRING_VALUE``: allow printable chars, ``\\"``, ``\\\\``
      - ``NUMBER_VALUE``: allow digits, ``.``, ``-``, ``+``, ``e``, ``E``
      - ``AFTER_VALUE``: allow ``,`` or ``}`` or whitespace
      - ``END``: only allow ``[/TOOL_CALL]`` prefix characters

    For simplicity, we use a per-character allowlist approach that works
    with character-level tokenizers.  For BPE, we fall back to no constraint
    (the model is expected to learn JSON format from training).
    """

    # Characters valid in JSON
    JSON_CHARS = set('{"}[]:,\n\r\t 0123456789.-+eE')
    JSON_STRING_CHARS = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ' .,;:!?_-/()[]{}@#$%^&*+=\'"\\\n\r\t'
    )
    JSON_KEYWORD_CHARS = set('truefalsnul')  # true, false, null

    def __init__(self, vocab_size: int, id_to_token: Optional[Dict[int, str]] = None):
        """
        Parameters
        ----------
        vocab_size : int
            Size of the model's vocabulary.
        id_to_token : dict, optional
            Mapping from token ID to token string (for character-level constraint).
            If None, constraint is disabled (BPE fallback).
        """
        self.vocab_size = vocab_size
        self.id_to_token = id_to_token
        self._active = False
        self._depth = 0  # JSON nesting depth
        self._in_string = False

    @property
    def active(self) -> bool:
        """Whether JSON-constrained decoding is currently active."""
        return self._active

    def activate(self):
        """Start JSON-constrained decoding (after [TOOL_CALL] is emitted)."""
        self._active = True
        self._depth = 0
        self._in_string = False

    def deactivate(self):
        """Stop JSON-constrained decoding (after [/TOOL_CALL] is emitted)."""
        self._active = False
        self._depth = 0
        self._in_string = False

    def get_allowed_ids(self) -> Optional[torch.Tensor]:
        """Return a boolean mask of allowed token IDs, or None if constraint inactive.

        Returns
        -------
        mask : torch.Tensor or None
            Boolean tensor of shape [vocab_size], True = allowed.
            None if constraint is not active or not applicable.
        """
        import torch

        if not self._active or self.id_to_token is None:
            return None

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        for token_id in range(self.vocab_size):
            token_str = self.id_to_token.get(token_id, "")
            if not token_str:
                continue
            # Check if this token's characters are valid in current JSON context
            if self._is_valid_token(token_str):
                mask[token_id] = True

        # Always allow [/TOOL_CALL] end token characters
        end_token = TOOL_TOKENS["tool_call_end"]
        for ch in end_token:
            for token_id, token_str in self.id_to_token.items():
                if token_str == ch:
                    mask[token_id] = True

        return mask

    def update_state(self, emitted_token: str):
        """Update the JSON parser state based on the last emitted token.

        Parameters
        ----------
        emitted_token : str
            The token string that was just emitted by the model.
        """
        if not self._active:
            return

        # Check for [/TOOL_CALL] — deactivate constraint
        if TOOL_TOKENS["tool_call_end"] in emitted_token:
            self.deactivate()
            return

        for ch in emitted_token:
            if ch == '"':
                self._in_string = not self._in_string
            elif not self._in_string:
                if ch == '{' or ch == '[':
                    self._depth += 1
                elif ch == '}' or ch == ']':
                    self._depth = max(0, self._depth - 1)

    def _is_valid_token(self, token_str: str) -> bool:
        """Check if a token string is valid in the current JSON context."""
        if self._in_string:
            # Inside a string: allow most printable chars
            for ch in token_str:
                if ord(ch) < 32 and ch not in '\n\r\t':
                    return False
            return True
        else:
            # Outside string: only allow JSON structural characters and keywords
            # Valid chars: { } [ ] : , " digits letters (for true/false/null)
            # whitespace, . - + e E (for numbers)
            valid_outside = set('{}[]:,"\n\r\t 0123456789.-+eEtruefalsnul')
            for ch in token_str:
                if ch not in valid_outside:
                    return False
            return True


# ---------------------------------------------------------------------------
# Tool Calling Session (multi-turn agentic loop)
# ---------------------------------------------------------------------------

class ToolCallingSession:
    """Multi-turn tool-calling session for HVN.

    Implements the agentic loop:
      1. Format the user's prompt with tool schemas.
      2. Generate text with the model.
      3. Check for [TOOL_CALL] blocks in the output.
      4. If tool calls found: execute them, inject [TOOL_RESULT], continue.
      5. If no tool calls: return the final answer.

    This preserves O(1) inference — each generation step uses the
    Hamiltonian state cache, and tool results are simply appended to
    the context as new tokens.

    Parameters
    ----------
    model : HamiltonianVortexNetwork
        The trained HVN model.
    tokenizer : CharTokenizer or BPETokenizer
        Tokenizer with special tool tokens added.
    executor : ToolExecutor
        Tool executor with registered tools.
    device : torch.device
        Device to run on.
    max_turns : int
        Maximum number of tool-calling turns before giving up.
    """

    def __init__(
        self,
        model,
        tokenizer,
        executor: ToolExecutor,
        device,
        max_turns: int = 5,
        temperature: float = 0.7,
        top_k: int = 10,
        max_tokens_per_turn: int = 200,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor
        self.device = device
        self.max_turns = max_turns
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens_per_turn = max_tokens_per_turn

    def format_prompt(self, user_query: str) -> str:
        """Format a user query with tool schemas for the model.

        The prompt structure is::

            [THINK]
            You have access to the following tools:
            <tool schemas>
            [/THINK]

            User: <query>

        The model is expected to respond with either a direct answer or
        [TOOL_CALL] blocks.
        """
        tool_desc = self.executor.registry.format_for_prompt()
        prompt = (
            f"{TOOL_TOKENS['think_start']}\n"
            f"You have access to the following tools:\n"
            f"{tool_desc}\n"
            f"Use [TOOL_CALL]json[/TOOL_CALL] to call a tool.\n"
            f"{TOOL_TOKENS['think_end']}\n\n"
            f"User: {user_query}\n"
        )
        return prompt

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run the tool-calling agentic loop.

        Parameters
        ----------
        user_query : str
            The user's question or request.

        Returns
        -------
        result : dict
            Dictionary with keys:
              - ``answer``: the final text answer (with tool blocks stripped)
              - ``raw_output``: full model output including tool calls/results
              - ``tool_calls``: list of all tool calls made
              - ``tool_results``: list of all tool result strings
              - ``n_turns``: number of turns executed
              - ``thinking``: list of [THINK] blocks
        """
        prompt = self.format_prompt(user_query)
        all_tool_calls = []
        all_tool_results = []
        all_thinking = []
        full_output = prompt

        for turn in range(self.max_turns):
            # Generate from the current context
            generated = self._generate(full_output)
            full_output += generated

            # Extract thinking blocks
            thinking = ToolCallParser.parse_thinking(generated)
            all_thinking.extend(thinking)

            # Check for tool calls
            tool_calls = ToolCallParser.parse_tool_calls(generated)

            if not tool_calls:
                # No tool calls — this is the final answer
                answer = ToolCallParser.strip_tool_blocks(generated)
                return {
                    "answer": answer.strip(),
                    "raw_output": full_output,
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "n_turns": turn + 1,
                    "thinking": all_thinking,
                }

            # Execute tool calls
            all_tool_calls.extend(tool_calls)
            results = self.executor.execute_all(tool_calls)
            all_tool_results.append(results)
            full_output += "\n" + results + "\n"

        # Exceeded max turns
        answer = ToolCallParser.strip_tool_blocks(full_output)
        return {
            "answer": answer.strip() or "Reached maximum tool-calling turns without a final answer.",
            "raw_output": full_output,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "n_turns": self.max_turns,
            "thinking": all_thinking,
        }

    def _generate(self, context: str) -> str:
        """Generate text from a context string using the HVN model.

        Uses incremental generation (O(1) per token) with the Hamiltonian
        state cache.
        """
        # Encode the context
        ids = self.tokenizer.encode(context)
        if not ids:
            ids = [0]

        # Use incremental generation
        generated_ids = self.model.generate_incremental(
            ids,
            max_new_tokens=self.max_tokens_per_turn,
            temperature=self.temperature,
            top_k=self.top_k,
            device=self.device,
        )

        # Return only the new tokens (not the prompt)
        new_ids = generated_ids[len(ids):]
        return self.tokenizer.decode(new_ids)


# ---------------------------------------------------------------------------
# Built-in tools (for demo)
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """A simple safe calculator tool.

    Evaluates basic arithmetic expressions safely (no exec/eval of arbitrary code).
    Supports +, -, *, /, **, (), and math functions.

    Parameters
    ----------
    expression : str
        A mathematical expression to evaluate (e.g. "2+2", "sqrt(16)", "3*4+1").
    """
    # Safe evaluation: only allow math functions and basic operators
    allowed_names = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    allowed_names.update({
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
    })

    # Remove any potentially dangerous characters
    # Only allow: digits, operators, parentheses, dots, letters, spaces
    safe_chars = set("0123456789+-*/().,%^ \t\nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    if not all(c in safe_chars for c in expression):
        return "Error: expression contains invalid characters"

    # Replace ^ with ** for exponentiation
    expression = expression.replace("^", "**")

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def web_search(query: str) -> str:
    """A mock web search tool (for demo purposes).

    In a real deployment, this would call an actual search API.
    For the demo, it returns a canned response based on the query.

    Parameters
    ----------
    query : str
        The search query.
    """
    # Mock results — in production, replace with actual API call
    mock_results = {
        "python": "Python is a high-level programming language. Created by Guido van Rossum in 1991.",
        "hvn": "The Hamiltonian Vortex Network is a novel architecture combining fluid dynamics and quantum mechanics.",
        "weather": "Weather: 22°C, partly cloudy, humidity 45%.",
        "default": f"Search results for '{query}': No relevant results found.",
    }
    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return result
    return mock_results["default"]


def string_reverse(text: str) -> str:
    """Reverse a string.

    Parameters
    ----------
    text : str
        The text to reverse.
    """
    return text[::-1]


def word_count(text: str) -> str:
    """Count words in a text.

    Parameters
    ----------
    text : str
        The text to count words in.
    """
    count = len(text.split())
    return str(count)


def create_demo_registry() -> ToolRegistry:
    """Create a ToolRegistry with built-in demo tools.

    Returns
    -------
    registry : ToolRegistry
        Registry with calculator, web_search, string_reverse, and word_count tools.
    """
    registry = ToolRegistry()

    registry.register(
        name="calculator",
        func=calculator,
        description="A simple calculator. Evaluate mathematical expressions like '2+2', 'sqrt(16)', '3*4+1'.",
        schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
    )

    registry.register(
        name="web_search",
        func=web_search,
        description="Search the web for information. Returns a text summary of search results.",
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        },
    )

    registry.register(
        name="string_reverse",
        func=string_reverse,
        description="Reverse a string of text.",
        schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to reverse"
                }
            },
            "required": ["text"]
        },
    )

    registry.register(
        name="word_count",
        func=word_count,
        description="Count the number of words in a text.",
        schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in"
                }
            },
            "required": ["text"]
        },
    )

    return registry


# ---------------------------------------------------------------------------
# Tool-calling training data generator
# ---------------------------------------------------------------------------

def generate_tool_training_data(num_examples: int = 50) -> List[Dict[str, str]]:
    """Generate synthetic tool-calling training examples.

    Each example is a dict with:
      - ``prompt``: the input prompt (with tool schemas + user query)
      - ``response``: the expected model output (with [TOOL_CALL] blocks)

    This is used by train.py to teach the model the tool-calling format.
    The examples are template-based (not learned) — they demonstrate the
    correct format for tool calls.

    Parameters
    ----------
    num_examples : int
        Number of training examples to generate.

    Returns
    -------
    examples : list[dict]
        List of {"prompt": ..., "response": ...} dicts.
    """
    templates = [
        # Calculator examples
        {
            "query": "What is 2+2?",
            "tool": "calculator",
            "args": {"expression": "2+2"},
            "result": "4",
            "answer": "2+2 = 4",
        },
        {
            "query": "What is 15*3?",
            "tool": "calculator",
            "args": {"expression": "15*3"},
            "result": "45",
            "answer": "15*3 = 45",
        },
        {
            "query": "What is the square root of 144?",
            "tool": "calculator",
            "args": {"expression": "sqrt(144)"},
            "result": "12.0",
            "answer": "The square root of 144 is 12.",
        },
        {
            "query": "What is 100/7?",
            "tool": "calculator",
            "args": {"expression": "100/7"},
            "result": "14.285714285714286",
            "answer": "100/7 is approximately 14.29.",
        },
        {
            "query": "What is 2**10?",
            "tool": "calculator",
            "args": {"expression": "2**10"},
            "result": "1024",
            "answer": "2 to the power of 10 is 1024.",
        },
        # Web search examples
        {
            "query": "What is Python?",
            "tool": "web_search",
            "args": {"query": "python"},
            "result": "Python is a high-level programming language. Created by Guido van Rossum in 1991.",
            "answer": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        },
        {
            "query": "What is the weather today?",
            "tool": "web_search",
            "args": {"query": "weather"},
            "result": "Weather: 22°C, partly cloudy, humidity 45%.",
            "answer": "Today's weather is 22°C, partly cloudy, with 45% humidity.",
        },
        # String reverse examples
        {
            "query": "Reverse the string 'hello'",
            "tool": "string_reverse",
            "args": {"text": "hello"},
            "result": "olleh",
            "answer": "The reverse of 'hello' is 'olleh'.",
        },
        # Word count examples
        {
            "query": "How many words are in 'the quick brown fox'?",
            "tool": "word_count",
            "args": {"text": "the quick brown fox"},
            "result": "4",
            "answer": "There are 4 words in 'the quick brown fox'.",
        },
        # Multi-turn example (call tool, then answer)
        {
            "query": "What is 3*5 and what is 10+20?",
            "tool": "calculator",
            "args": {"expression": "3*5"},
            "result": "15",
            "answer": "3*5 = 15. Now let me calculate 10+20.",
            "second_tool": "calculator",
            "second_args": {"expression": "10+20"},
            "second_result": "30",
            "second_answer": "10+20 = 30. So 3*5 = 15 and 10+20 = 30.",
        },
    ]

    examples = []
    for i in range(num_examples):
        template = templates[i % len(templates)]
        tool_desc = f"Available tools: {template['tool']}"
        prompt = (
            f"{TOOL_TOKENS['think_start']}\n"
            f"You have access to the following tools:\n"
            f"  1. {template['tool']}\n"
            f"{TOOL_TOKENS['think_end']}\n\n"
            f"User: {template['query']}\n"
        )

        # Build response with tool call(s)
        call_json = json.dumps({
            "name": template["tool"],
            "arguments": template["args"],
        }, separators=(",", ":"))

        response_parts = [
            f"{TOOL_TOKENS['tool_call_start']}\n{call_json}\n{TOOL_TOKENS['tool_call_end']}",
            f"{TOOL_TOKENS['tool_result_start']}\n{template['result']}\n{TOOL_TOKENS['tool_result_end']}",
            template["answer"],
        ]

        # Multi-turn: add second tool call
        if "second_tool" in template:
            second_call_json = json.dumps({
                "name": template["second_tool"],
                "arguments": template["second_args"],
            }, separators=(",", ":"))
            response_parts.extend([
                f"{TOOL_TOKENS['tool_call_start']}\n{second_call_json}\n{TOOL_TOKENS['tool_call_end']}",
                f"{TOOL_TOKENS['tool_result_start']}\n{template['second_result']}\n{TOOL_TOKENS['tool_result_end']}",
                template["second_answer"],
            ])

        response = "\n".join(response_parts)
        examples.append({"prompt": prompt, "response": response})

    return examples


__all__ = [
    "TOOL_TOKENS",
    "SPECIAL_TOKEN_STRINGS",
    "ToolRegistry",
    "ToolCallParser",
    "ToolExecutor",
    "ConstrainedJSONDecoder",
    "ToolCallingSession",
    "calculator",
    "web_search",
    "string_reverse",
    "word_count",
    "create_demo_registry",
    "generate_tool_training_data",
]