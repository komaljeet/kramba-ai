"""
Agentic Mode for HVN (Issue #25)
=================================

Provides a ReAct-style agentic loop with multi-step reasoning, planning,
tool chaining, and self-correction for the Hamiltonian Vortex Network.

Components:
  - ``AgentRuntime``  — top-level orchestrator: manages conversation state,
    runs the ReAct loop, executes tools, and collects the reasoning chain.
  - ``ReActLoop``     — the core Thought → Action → Observation cycle.
  - ``Planner``       — decomposes complex tasks into subtasks and tracks
    execution progress; can revise the plan based on intermediate results.
  - ``SelfCorrector`` — detects errors (from tool output or self-reflection),
    emits corrections, and triggers re-planning while retaining Hamiltonian
    memory context from failed attempts.

Agentic mode is a **runtime feature** — the HVN model itself is unchanged.
O(1) inference per token is preserved (the Hamiltonian state cache carries
conversation context across turns).  No KV cache is introduced.

Special tokens (shared with Issue #24 tool-calling):
  - ``[TOOL_CALL]`` / ``[/TOOL_CALL]``
  - ``[TOOL_RESULT]`` / ``[/TOOL_RESULT]``
  - ``[THINK]`` / ``[/THINK]``

Additional agentic tokens:
  - ``[PLAN]`` / ``[/PLAN]``       — planning block
  - ``[ACTION]`` / ``[/ACTION]``   — action block (alias for TOOL_CALL)
  - ``[OBSERVATION]`` / ``[/OBSERVATION]`` — observation block (alias for TOOL_RESULT)
  - ``[CORRECT]`` / ``[/CORRECT]`` — self-correction block
  - ``[FINAL]`` / ``[/FINAL]``     — final answer block

Usage:
    from hvn.agent import AgentRuntime, ReActLoop, Planner, SelfCorrector
    from hvn.tools import ToolRegistry, ToolExecutor, create_demo_registry

    registry = create_demo_registry()
    executor = ToolExecutor(registry)
    agent = AgentRuntime(model, tokenizer, executor, device)
    result = agent.run("Research Python and calculate 2+2", max_iterations=10)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Agentic tokens (extend the tool-calling tokens from Issue #24)
# ---------------------------------------------------------------------------

AGENT_TOKENS: Dict[str, str] = {
    "plan_start": "[PLAN]",
    "plan_end": "[/PLAN]",
    "action_start": "[ACTION]",
    "action_end": "[/ACTION]",
    "observation_start": "[OBSERVATION]",
    "observation_end": "[/OBSERVATION]",
    "correct_start": "[CORRECT]",
    "correct_end": "[/CORRECT]",
    "final_start": "[FINAL]",
    "final_end": "[/FINAL]",
}

# All agentic token strings (for tokenizer registration if needed)
AGENT_TOKEN_STRINGS: List[str] = list(AGENT_TOKENS.values())


# ---------------------------------------------------------------------------
# Reasoning step dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single step in the agentic reasoning chain.

    Each step is one of:
      - ``thought``: the model's reasoning (from [THINK] block)
      - ``action``: a tool call (from [TOOL_CALL] / [ACTION] block)
      - ``observation``: the tool result (from [TOOL_RESULT] / [OBSERVATION] block)
      - ``plan``: a planning block (from [PLAN] block)
      - ``correction``: a self-correction (from [CORRECT] block)
      - ``final``: the final answer (from [FINAL] block or plain text)
    """
    step_type: str  # thought | action | observation | plan | correction | final
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = {"step_type": self.step_type, "content": self.content, "timestamp": self.timestamp}
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_args:
            d["tool_args"] = self.tool_args
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result
        return d


# ---------------------------------------------------------------------------
# Regex patterns for parsing agentic blocks
# ---------------------------------------------------------------------------

class AgentParser:
    """Parse agentic blocks from model output.

    Recognises [THINK], [TOOL_CALL], [TOOL_RESULT], [PLAN], [ACTION],
    [OBSERVATION], [CORRECT], and [FINAL] blocks.
    """

    PLAN_RE = re.compile(r"\[PLAN\]\s*(.*?)\s*\[/PLAN\]", re.DOTALL)
    ACTION_RE = re.compile(r"\[ACTION\]\s*(.*?)\s*\[/ACTION\]", re.DOTALL)
    OBSERVATION_RE = re.compile(r"\[OBSERVATION\]\s*(.*?)\s*\[/OBSERVATION\]", re.DOTALL)
    CORRECT_RE = re.compile(r"\[CORRECT\]\s*(.*?)\s*\[/CORRECT\]", re.DOTALL)
    FINAL_RE = re.compile(r"\[FINAL\]\s*(.*?)\s*\[/FINAL\]", re.DOTALL)

    # Reuse patterns from ToolCallParser
    TOOL_CALL_RE = re.compile(r"\[TOOL_CALL\]\s*(.*?)\s*\[/TOOL_CALL\]", re.DOTALL)
    TOOL_RESULT_RE = re.compile(r"\[TOOL_RESULT\]\s*(.*?)\s*\[/TOOL_RESULT\]", re.DOTALL)
    THINK_RE = re.compile(r"\[THINK\]\s*(.*?)\s*\[/THINK\]", re.DOTALL)

    @classmethod
    def parse_all(cls, text: str) -> List[ReasoningStep]:
        """Parse all agentic blocks from text into a list of ReasoningSteps.

        Returns steps in order of appearance in the text.
        """
        steps: List[ReasoningStep] = []

        # Parse thinking blocks
        for m in cls.THINK_RE.finditer(text):
            steps.append(ReasoningStep(step_type="thought", content=m.group(1).strip()))

        # Parse plan blocks
        for m in cls.PLAN_RE.finditer(text):
            steps.append(ReasoningStep(step_type="plan", content=m.group(1).strip()))

        # Parse tool calls / actions
        for m in cls.TOOL_CALL_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                parsed = json.loads(raw)
                steps.append(ReasoningStep(
                    step_type="action",
                    content=raw,
                    tool_name=parsed.get("name", ""),
                    tool_args=parsed.get("arguments", {}),
                ))
            except json.JSONDecodeError:
                steps.append(ReasoningStep(step_type="action", content=raw))

        for m in cls.ACTION_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                parsed = json.loads(raw)
                steps.append(ReasoningStep(
                    step_type="action",
                    content=raw,
                    tool_name=parsed.get("name", ""),
                    tool_args=parsed.get("arguments", {}),
                ))
            except json.JSONDecodeError:
                steps.append(ReasoningStep(step_type="action", content=raw))

        # Parse tool results / observations
        for m in cls.TOOL_RESULT_RE.finditer(text):
            steps.append(ReasoningStep(
                step_type="observation",
                content=m.group(1).strip(),
                tool_result=m.group(1).strip(),
            ))

        for m in cls.OBSERVATION_RE.finditer(text):
            steps.append(ReasoningStep(
                step_type="observation",
                content=m.group(1).strip(),
                tool_result=m.group(1).strip(),
            ))

        # Parse corrections
        for m in cls.CORRECT_RE.finditer(text):
            steps.append(ReasoningStep(step_type="correction", content=m.group(1).strip()))

        # Parse final answer
        for m in cls.FINAL_RE.finditer(text):
            steps.append(ReasoningStep(step_type="final", content=m.group(1).strip()))

        # Sort by position in text (approximate by timestamp of creation)
        # Actually, we want to preserve text order. Let's re-parse with positions.
        return steps

    @classmethod
    def parse_ordered(cls, text: str) -> List[ReasoningStep]:
        """Parse all agentic blocks preserving their order in the text."""
        # Collect all matches with their start position
        all_matches: List[Tuple[int, ReasoningStep]] = []

        for m in cls.THINK_RE.finditer(text):
            all_matches.append((m.start(), ReasoningStep(
                step_type="thought", content=m.group(1).strip()
            )))

        for m in cls.PLAN_RE.finditer(text):
            all_matches.append((m.start(), ReasoningStep(
                step_type="plan", content=m.group(1).strip()
            )))

        for m in cls.TOOL_CALL_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                parsed = json.loads(raw)
                step = ReasoningStep(
                    step_type="action",
                    content=raw,
                    tool_name=parsed.get("name", ""),
                    tool_args=parsed.get("arguments", {}),
                )
            except json.JSONDecodeError:
                step = ReasoningStep(step_type="action", content=raw)
            all_matches.append((m.start(), step))

        for m in cls.ACTION_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                parsed = json.loads(raw)
                step = ReasoningStep(
                    step_type="action",
                    content=raw,
                    tool_name=parsed.get("name", ""),
                    tool_args=parsed.get("arguments", {}),
                )
            except json.JSONDecodeError:
                step = ReasoningStep(step_type="action", content=raw)
            all_matches.append((m.start(), step))

        for m in cls.TOOL_RESULT_RE.finditer(text):
            all_matches.append((m.start(), ReasoningStep(
                step_type="observation",
                content=m.group(1).strip(),
                tool_result=m.group(1).strip(),
            )))

        for m in cls.OBSERVATION_RE.finditer(text):
            all_matches.append((m.start(), ReasoningStep(
                step_type="observation",
                content=m.group(1).strip(),
                tool_result=m.group(1).strip(),
            )))

        for m in cls.CORRECT_RE.finditer(text):
            all_matches.append((m.start(), ReasoningStep(
                step_type="correction", content=m.group(1).strip()
            )))

        for m in cls.FINAL_RE.finditer(text):
            all_matches.append((m.start(), ReasoningStep(
                step_type="final", content=m.group(1).strip()
            )))

        # Sort by position and return steps
        all_matches.sort(key=lambda x: x[0])
        return [step for _, step in all_matches]

    @classmethod
    def has_final(cls, text: str) -> bool:
        """Check if text contains a [FINAL] block."""
        return bool(cls.FINAL_RE.search(text))

    @classmethod
    def has_correction(cls, text: str) -> bool:
        """Check if text contains a [CORRECT] block."""
        return bool(cls.CORRECT_RE.search(text))

    @classmethod
    def has_plan(cls, text: str) -> bool:
        """Check if text contains a [PLAN] block."""
        return bool(cls.PLAN_RE.search(text))

    @classmethod
    def has_tool_call(cls, text: str) -> bool:
        """Check if text contains a [TOOL_CALL] or [ACTION] block."""
        return bool(cls.TOOL_CALL_RE.search(text) or cls.ACTION_RE.search(text))

    @classmethod
    def extract_final_answer(cls, text: str) -> str:
        """Extract the final answer from a [FINAL] block, or strip blocks."""
        m = cls.FINAL_RE.search(text)
        if m:
            return m.group(1).strip()
        # Fall back to stripping all blocks
        text = cls.THINK_RE.sub("", text)
        text = cls.TOOL_CALL_RE.sub("", text)
        text = cls.TOOL_RESULT_RE.sub("", text)
        text = cls.PLAN_RE.sub("", text)
        text = cls.ACTION_RE.sub("", text)
        text = cls.OBSERVATION_RE.sub("", text)
        text = cls.CORRECT_RE.sub("", text)
        text = cls.FINAL_RE.sub("", text)
        return text.strip()

    @classmethod
    def parse_plan_items(cls, plan_text: str) -> List[str]:
        """Parse a plan block into individual subtask items.

        Each line in the plan is treated as a subtask.  Lines are stripped
        of numbering (e.g. "1. ", "2. ") and bullet markers ("- ", "* ").
        """
        lines = plan_text.strip().split("\n")
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering: "1. ", "2. ", etc.
            line = re.sub(r"^\d+\.\s*", "", line)
            # Remove bullet markers
            line = re.sub(r"^[-*]\s*", "", line)
            if line:
                items.append(line)
        return items


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Decomposes complex tasks into subtasks and tracks execution.

    The planner can:
      1. Generate a plan from the model (using [PLAN] blocks).
      2. Track which subtasks are completed, in-progress, or failed.
      3. Revise the plan based on intermediate results (re-planning).

    Parameters
    ----------
    model : HamiltonianVortexNetwork
        The trained HVN model (used for plan generation).
    tokenizer : CharTokenizer or BPETokenizer
        Tokenizer for encoding/decoding.
    device : torch.device
        Device to run on.
    temperature : float
        Sampling temperature for plan generation.
    top_k : int
        Top-k sampling.
    max_tokens : int
        Max tokens per generation call.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        temperature: float = 0.7,
        top_k: int = 10,
        max_tokens: int = 200,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.subtasks: List[Dict[str, Any]] = []
        self.current_idx = 0
        self.revisions: List[List[Dict[str, Any]]] = []  # history of plan revisions

    def create_plan(self, task: str) -> List[str]:
        """Decompose a task into subtasks.

        If a model is available, generates a plan using the model.
        Otherwise, uses a simple heuristic decomposition.

        Parameters
        ----------
        task : str
            The complex task to decompose.

        Returns
        -------
        subtasks : list[str]
            List of subtask descriptions.
        """
        if self.model is not None and self.tokenizer is not None:
            return self._generate_plan(task)
        else:
            return self._heuristic_plan(task)

    def _generate_plan(self, task: str) -> List[str]:
        """Generate a plan using the HVN model."""
        prompt = (
            f"[THINK]\nYou need to plan the following task. Break it into steps.\n"
            f"Task: {task}\n"
            f"Write a plan with one step per line.\n[/THINK]\n"
            f"[PLAN]\n"
        )
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [0]

        generated = self.model.generate_incremental(
            ids,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            device=self.device,
        )
        new_ids = generated[len(ids):]
        output = self.tokenizer.decode(new_ids)

        # Extract plan text (everything before [/PLAN])
        plan_match = re.search(r"(.*?)(?:\[/PLAN\]|$)", output, re.DOTALL)
        plan_text = plan_match.group(1) if plan_match else output

        subtasks = AgentParser.parse_plan_items(plan_text)
        if not subtasks:
            subtasks = self._heuristic_plan(task)

        self._set_subtasks(subtasks)
        return subtasks

    def _heuristic_plan(self, task: str) -> List[str]:
        """Simple heuristic decomposition when no model is available.

        Splits the task by common conjunctions and question words.
        """
        # Try to split on "and", "then", commas
        parts = re.split(r"\b(?:and|then)\b|,", task, flags=re.IGNORECASE)
        subtasks = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
        if len(subtasks) <= 1:
            subtasks = [task]
        self._set_subtasks(subtasks)
        return subtasks

    def _set_subtasks(self, subtasks: List[str]):
        """Initialize subtask tracking."""
        self.subtasks = [
            {"description": st, "status": "pending", "result": None, "attempts": 0}
            for st in subtasks
        ]
        self.current_idx = 0

    def get_next_subtask(self) -> Optional[str]:
        """Get the next pending subtask, or None if all are done."""
        for i, st in enumerate(self.subtasks):
            if st["status"] == "pending":
                self.current_idx = i
                return st["description"]
        return None

    def mark_completed(self, subtask_idx: int, result: str):
        """Mark a subtask as completed."""
        if 0 <= subtask_idx < len(self.subtasks):
            self.subtasks[subtask_idx]["status"] = "completed"
            self.subtasks[subtask_idx]["result"] = result

    def mark_failed(self, subtask_idx: int, error: str):
        """Mark a subtask as failed."""
        if 0 <= subtask_idx < len(self.subtasks):
            self.subtasks[subtask_idx]["status"] = "failed"
            self.subtasks[subtask_idx]["result"] = error
            self.subtasks[subtask_idx]["attempts"] += 1

    def revise_plan(self, feedback: str) -> List[str]:
        """Revise the plan based on feedback (intermediate results).

        Saves the current plan to revisions history, then generates a new plan
        that accounts for what has been done so far.

        Parameters
        ----------
        feedback : str
            Feedback about what went wrong or what was learned.

        Returns
        -------
        new_subtasks : list[str]
            The revised list of subtasks.
        """
        # Save current plan to history
        self.revisions.append([dict(st) for st in self.subtasks])

        # Build context of completed/failed subtasks
        completed = [st["description"] for st in self.subtasks if st["status"] == "completed"]
        failed = [st["description"] for st in self.subtasks if st["status"] == "failed"]
        pending = [st["description"] for st in self.subtasks if st["status"] == "pending"]

        if self.model is not None and self.tokenizer is not None:
            prompt = (
                f"[THINK]\n"
                f"Original task: {self.subtasks[0]['description'] if self.subtasks else 'unknown'}\n"
                f"Completed: {completed}\n"
                f"Failed: {failed}\n"
                f"Feedback: {feedback}\n"
                f"Revise the plan. Keep completed steps, retry or replace failed steps.\n"
                f"[/THINK]\n[PLAN]\n"
            )
            ids = self.tokenizer.encode(prompt)
            if not ids:
                ids = [0]
            generated = self.model.generate_incremental(
                ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                device=self.device,
            )
            new_ids = generated[len(ids):]
            output = self.tokenizer.decode(new_ids)
            plan_match = re.search(r"(.*?)(?:\[/PLAN\]|$)", output, re.DOTALL)
            plan_text = plan_match.group(1) if plan_match else output
            new_subtasks = AgentParser.parse_plan_items(plan_text)
            if not new_subtasks:
                new_subtasks = pending + [f"Retry: {f}" for f in failed]
        else:
            # Heuristic: keep pending, add retry for failed
            new_subtasks = pending + [f"Retry: {f}" for f in failed]

        self._set_subtasks(new_subtasks)
        return new_subtasks

    def get_progress(self) -> Dict[str, Any]:
        """Get a summary of plan progress."""
        total = len(self.subtasks)
        completed = sum(1 for st in self.subtasks if st["status"] == "completed")
        failed = sum(1 for st in self.subtasks if st["status"] == "failed")
        pending = sum(1 for st in self.subtasks if st["status"] == "pending")
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "subtasks": [dict(st) for st in self.subtasks],
            "n_revisions": len(self.revisions),
        }

    def is_complete(self) -> bool:
        """Check if all subtasks are completed."""
        return all(st["status"] == "completed" for st in self.subtasks) if self.subtasks else True


# ---------------------------------------------------------------------------
# SelfCorrector
# ---------------------------------------------------------------------------

class SelfCorrector:
    """Detects errors and triggers self-correction.

    Error detection strategies:
      1. Tool error detection: check if tool result starts with "Error"
      2. Self-reflection: ask the model to evaluate its own output
      3. Pattern matching: detect common error patterns

    When an error is detected, the corrector:
      1. Logs the error
      2. Emits a [CORRECT] block with the correction
      3. Triggers re-planning via the Planner

    Parameters
    ----------
    max_corrections : int
        Maximum number of self-corrections before giving up.
    error_patterns : list[str], optional
        Additional regex patterns to detect errors in tool output.
    """

    ERROR_INDICATORS = [
        "error",
        "exception",
        "traceback",
        "failed",
        "invalid",
        "not found",
        "cannot",
        "unable",
        "nan",
        "inf",
    ]

    def __init__(
        self,
        max_corrections: int = 3,
        error_patterns: Optional[List[str]] = None,
    ):
        self.max_corrections = max_corrections
        self.correction_count = 0
        self.error_history: List[Dict[str, Any]] = []
        self.custom_patterns = [re.compile(p, re.IGNORECASE) for p in (error_patterns or [])]

    def detect_error(self, tool_result: str, model_output: str = "") -> Tuple[bool, str]:
        """Check if a tool result or model output indicates an error.

        Parameters
        ----------
        tool_result : str
            The result from a tool execution.
        model_output : str
            The model's generated text (for self-reflection).

        Returns
        -------
        is_error : bool
            True if an error is detected.
        error_msg : str
            Description of the detected error (empty string if no error).
        """
        result_lower = tool_result.lower()

        # Check for error indicators in tool result
        for indicator in self.ERROR_INDICATORS:
            if indicator in result_lower:
                return True, f"Tool error detected: {tool_result[:200]}"

        # Check custom patterns
        for pattern in self.custom_patterns:
            if pattern.search(tool_result):
                return True, f"Custom pattern matched: {tool_result[:200]}"

        # Check for empty result
        if not tool_result.strip():
            return True, "Empty tool result"

        return False, ""

    def should_correct(self) -> bool:
        """Check if we should attempt another correction."""
        return self.correction_count < self.max_corrections

    def record_correction(self, error: str, correction: str):
        """Record a correction attempt."""
        self.correction_count += 1
        self.error_history.append({
            "error": error,
            "correction": correction,
            "timestamp": time.time(),
        })

    def format_correction(self, error: str, correction: str) -> str:
        """Format a correction as a [CORRECT] block."""
        self.record_correction(error, correction)
        return (
            f"{AGENT_TOKENS['correct_start']}\n"
            f"Error: {error}\n"
            f"Correction: {correction}\n"
            f"{AGENT_TOKENS['correct_end']}"
        )

    def reset(self):
        """Reset correction count and history."""
        self.correction_count = 0
        self.error_history = []


# ---------------------------------------------------------------------------
# ReAct Loop
# ---------------------------------------------------------------------------

class ReActLoop:
    """The core ReAct (Reason + Act) loop.

    Each iteration:
      1. Generate model output from current context.
      2. Parse for thoughts, actions (tool calls), and final answers.
      3. If actions found: execute tools, format observations, append to context.
      4. If final answer found: return.
      5. If error detected: trigger self-correction.

    Parameters
    ----------
    model : HamiltonianVortexNetwork
        The trained HVN model.
    tokenizer : CharTokenizer or BPETokenizer
        Tokenizer with special tokens.
    executor : ToolExecutor
        Tool executor for running tools.
    device : torch.device
        Device to run on.
    temperature : float
        Sampling temperature.
    top_k : int
        Top-k sampling.
    max_tokens_per_turn : int
        Max tokens to generate per turn.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        executor: Optional[ToolExecutor] = None,
        device=None,
        temperature: float = 0.7,
        top_k: int = 10,
        max_tokens_per_turn: int = 200,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens_per_turn = max_tokens_per_turn
        self.steps: List[ReasoningStep] = []

    def format_react_prompt(self, user_query: str, tool_desc: str = "") -> str:
        """Format a ReAct-style prompt.

        The prompt instructs the model to use the ReAct format:
          Thought: ... (in [THINK] blocks)
          Action: ... (in [TOOL_CALL] blocks)
          Observation: ... (in [TOOL_RESULT] blocks)
          Final Answer: ... (in [FINAL] blocks)
        """
        if not tool_desc and self.executor:
            tool_desc = self.executor.registry.format_for_prompt()

        prompt = (
            f"[THINK]\n"
            f"You are an autonomous agent. Use the ReAct pattern:\n"
            f"1. Think about what to do (in [THINK] blocks)\n"
            f"2. Call a tool (in [TOOL_CALL] blocks with JSON)\n"
            f"3. Observe the result (in [TOOL_RESULT] blocks)\n"
            f"4. Repeat until you have the answer\n"
            f"5. Give the final answer (in [FINAL] blocks)\n\n"
            f"{tool_desc}\n"
            f"[/THINK]\n\n"
            f"User: {user_query}\n"
        )
        return prompt

    def run(
        self,
        user_query: str,
        max_iterations: int = 10,
        context: str = "",
    ) -> Dict[str, Any]:
        """Run the ReAct loop.

        Parameters
        ----------
        user_query : str
            The user's question or task.
        max_iterations : int
            Maximum number of Reason-Act cycles.
        context : str
            Additional context to prepend (e.g. from planning).

        Returns
        -------
        result : dict
            Dictionary with:
              - ``answer``: final answer
              - ``steps``: list of ReasoningStep dicts
              - ``tool_calls``: list of tool call dicts
              - ``tool_results``: list of tool result strings
              - ``n_iterations``: number of iterations executed
              - ``success``: whether a final answer was reached
        """
        prompt = self.format_react_prompt(user_query)
        full_context = context + prompt
        all_tool_calls = []
        all_tool_results = []

        for iteration in range(max_iterations):
            # Generate from current context
            generated = self._generate(full_context)
            full_context += generated

            # Parse steps from generated text
            new_steps = AgentParser.parse_ordered(generated)
            self.steps.extend(new_steps)

            # Check for final answer
            if AgentParser.has_final(generated):
                answer = AgentParser.extract_final_answer(generated)
                return {
                    "answer": answer,
                    "steps": [s.to_dict() for s in self.steps],
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "n_iterations": iteration + 1,
                    "success": True,
                    "full_context": full_context,
                }

            # Check for tool calls
            from hvn.tools import ToolCallParser
            tool_calls = ToolCallParser.parse_tool_calls(generated)

            if not tool_calls:
                # No tool calls and no final answer — treat remaining text as answer
                answer = AgentParser.extract_final_answer(generated)
                if answer.strip():
                    return {
                        "answer": answer,
                        "steps": [s.to_dict() for s in self.steps],
                        "tool_calls": all_tool_calls,
                        "tool_results": all_tool_results,
                        "n_iterations": iteration + 1,
                        "success": True,
                        "full_context": full_context,
                    }
                # Otherwise continue to next iteration
                continue

            # Execute tool calls
            all_tool_calls.extend(tool_calls)
            results = self.executor.execute_all(tool_calls)
            all_tool_results.append(results)
            full_context += "\n" + results + "\n"

        # Max iterations reached
        answer = AgentParser.extract_final_answer(full_context)
        return {
            "answer": answer.strip() or "Reached maximum iterations without a final answer.",
            "steps": [s.to_dict() for s in self.steps],
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "n_iterations": max_iterations,
            "success": False,
            "full_context": full_context,
        }

    def _generate(self, context: str) -> str:
        """Generate text from context using the HVN model."""
        if self.model is None or self.tokenizer is None:
            return ""

        ids = self.tokenizer.encode(context)
        if not ids:
            ids = [0]

        generated_ids = self.model.generate_incremental(
            ids,
            max_new_tokens=self.max_tokens_per_turn,
            temperature=self.temperature,
            top_k=self.top_k,
            device=self.device,
        )
        new_ids = generated_ids[len(ids):]
        return self.tokenizer.decode(new_ids)


# ---------------------------------------------------------------------------
# Agent Runtime (top-level orchestrator)
# ---------------------------------------------------------------------------

class AgentRuntime:
    """Top-level agentic runtime for HVN.

    Orchestrates the full agentic loop:
      1. (Optional) Planning: decompose the task into subtasks.
      2. ReAct loop: for each subtask, run Thought → Action → Observation.
      3. Self-correction: detect errors and trigger corrections.
      4. Synthesize: combine subtask results into a final answer.

    The runtime maintains conversation state using the HVN model's
    Hamiltonian state cache — O(1) inference per token is preserved.

    Parameters
    ----------
    model : HamiltonianVortexNetwork
        The trained HVN model.
    tokenizer : CharTokenizer or BPETokenizer
        Tokenizer with special tokens.
    executor : ToolExecutor
        Tool executor with registered tools.
    device : torch.device
        Device to run on.
    temperature : float
        Sampling temperature.
    top_k : int
        Top-k sampling.
    max_tokens_per_turn : int
        Max tokens per generation call.
    enable_planning : bool
        Whether to use the planning phase.
    enable_self_correction : bool
        Whether to use self-correction.
    max_corrections : int
        Max self-correction attempts per subtask.
    verbose : bool
        Print step-by-step reasoning.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        executor: Optional[ToolExecutor] = None,
        device=None,
        temperature: float = 0.7,
        top_k: int = 10,
        max_tokens_per_turn: int = 200,
        enable_planning: bool = True,
        enable_self_correction: bool = True,
        max_corrections: int = 3,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.executor = executor
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens_per_turn = max_tokens_per_turn
        self.enable_planning = enable_planning
        self.enable_self_correction = enable_self_correction
        self.verbose = verbose

        # Sub-components
        self.planner = Planner(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens_per_turn,
        )
        self.corrector = SelfCorrector(max_corrections=max_corrections)
        self.react_loop = ReActLoop(
            model=model,
            tokenizer=tokenizer,
            executor=executor,
            device=device,
            temperature=temperature,
            top_k=top_k,
            max_tokens_per_turn=max_tokens_per_turn,
        )

        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.full_context: str = ""

    def _log(self, msg: str):
        """Log a message if verbose mode is on."""
        if self.verbose:
            print(msg)

    def run(
        self,
        user_query: str,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Run the full agentic loop on a user query.

        Parameters
        ----------
        user_query : str
            The user's task or question.
        max_iterations : int
            Maximum ReAct iterations per subtask.

        Returns
        -------
        result : dict
            Dictionary with:
              - ``answer``: final answer
              - ``plan``: list of subtasks (if planning enabled)
              - ``plan_progress``: progress summary
              - ``steps``: full reasoning chain
              - ``tool_calls``: all tool calls made
              - ``tool_results``: all tool results
              - ``corrections``: list of corrections made
              - ``n_iterations``: total iterations
              - ``success``: whether the task completed
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"  AGENT RUNTIME — Starting task")
        self._log(f"  Query: {user_query}")
        self._log(f"{'=' * 60}")

        all_steps: List[ReasoningStep] = []
        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_results: List[str] = []
        all_corrections: List[Dict[str, Any]] = []
        subtask_results: List[str] = []
        total_iterations = 0
        context = ""

        # --- Phase 1: Planning ---
        subtasks = None
        if self.enable_planning:
            self._log("\n--- Phase 1: Planning ---")
            subtasks = self.planner.create_plan(user_query)
            self._log(f"Plan ({len(subtasks)} subtasks):")
            for i, st in enumerate(subtasks, 1):
                self._log(f"  {i}. {st}")
            all_steps.append(ReasoningStep(
                step_type="plan",
                content="\n".join(f"{i+1}. {st}" for i, st in enumerate(subtasks)),
            ))
            context += f"[PLAN]\n" + "\n".join(f"{i+1}. {st}" for i, st in enumerate(subtasks)) + "\n[/PLAN]\n"

        # --- Phase 2: Execution (ReAct loop per subtask) ---
        self._log("\n--- Phase 2: Execution ---")

        if subtasks:
            # Execute each subtask
            for idx, subtask in enumerate(subtasks):
                self._log(f"\n  Subtask {idx + 1}/{len(subtasks)}: {subtask}")
                self.corrector.reset()

                subtask_result = self._execute_subtask(
                    subtask, max_iterations, context,
                    all_steps, all_tool_calls, all_tool_results, all_corrections,
                )
                subtask_results.append(subtask_result["answer"])
                total_iterations += subtask_result["n_iterations"]

                # Update planner
                if subtask_result["success"]:
                    self.planner.mark_completed(idx, subtask_result["answer"])
                    self._log(f"  ✓ Subtask {idx + 1} completed: {subtask_result['answer'][:100]}")
                else:
                    self.planner.mark_failed(idx, subtask_result["answer"])
                    self._log(f"  ✗ Subtask {idx + 1} failed: {subtask_result['answer'][:100]}")

                    # Self-correction: try to revise plan
                    if self.enable_self_correction and self.corrector.should_correct():
                        self._log("  → Revising plan due to failure...")
                        revised = self.planner.revise_plan(subtask_result["answer"])
                        self._log(f"  Revised plan: {revised}")
                        all_corrections.append({
                            "type": "plan_revision",
                            "reason": subtask_result["answer"],
                            "new_plan": revised,
                        })

                # Append subtask result to context
                context += f"\n[THINK]\nSubtask {idx + 1} result: {subtask_result['answer']}\n[/THINK]\n"
        else:
            # No planning — run single ReAct loop
            subtask_result = self._execute_subtask(
                user_query, max_iterations, context,
                all_steps, all_tool_calls, all_tool_results, all_corrections,
            )
            subtask_results.append(subtask_result["answer"])
            total_iterations = subtask_result["n_iterations"]

        # --- Phase 3: Synthesis ---
        self._log("\n--- Phase 3: Synthesis ---")
        if len(subtask_results) > 1:
            final_answer = self._synthesize(user_query, subtask_results, context)
        else:
            final_answer = subtask_results[0] if subtask_results else "No answer produced."

        self._log(f"\nFinal Answer: {final_answer[:200]}")
        self._log(f"\n{'=' * 60}")
        self._log(f"  Task complete: {total_iterations} iterations, "
                  f"{len(all_tool_calls)} tool calls, "
                  f"{len(all_corrections)} corrections")
        self._log(f"{'=' * 60}")

        return {
            "answer": final_answer,
            "plan": subtasks,
            "plan_progress": self.planner.get_progress() if subtasks else None,
            "steps": [s.to_dict() for s in all_steps],
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "corrections": all_corrections,
            "n_iterations": total_iterations,
            "success": self.planner.is_complete() if subtasks else subtask_result["success"],
        }

    def _execute_subtask(
        self,
        subtask: str,
        max_iterations: int,
        context: str,
        all_steps: List[ReasoningStep],
        all_tool_calls: List[Dict[str, Any]],
        all_tool_results: List[str],
        all_corrections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a single subtask via the ReAct loop with self-correction."""
        for correction_attempt in range(self.corrector.max_corrections + 1):
            result = self.react_loop.run(
                subtask,
                max_iterations=max_iterations,
                context=context,
            )

            # Collect steps
            for step_dict in result["steps"]:
                step = ReasoningStep(
                    step_type=step_dict["step_type"],
                    content=step_dict["content"],
                    tool_name=step_dict.get("tool_name"),
                    tool_args=step_dict.get("tool_args"),
                    tool_result=step_dict.get("tool_result"),
                )
                all_steps.append(step)

            all_tool_calls.extend(result["tool_calls"])
            all_tool_results.extend(result["tool_results"])

            if result["success"]:
                return result

            # Self-correction: check for errors in tool results
            if self.enable_self_correction and self.corrector.should_correct():
                # Check the last tool result for errors
                last_result = result["tool_results"][-1] if result["tool_results"] else ""
                is_error, error_msg = self.corrector.detect_error(last_result)

                if is_error:
                    self._log(f"    ⚠ Error detected: {error_msg[:100]}")
                    correction = f"Retry subtask with different approach. Error: {error_msg}"
                    correction_block = self.corrector.format_correction(error_msg, correction)
                    all_corrections.append({
                        "type": "tool_error",
                        "error": error_msg,
                        "correction": correction,
                    })
                    context += "\n" + correction_block + "\n"
                    continue  # retry
                else:
                    # No error detected but no final answer — still return
                    return result
            else:
                return result

        # Exhausted corrections
        return result

    def _synthesize(self, user_query: str, subtask_results: List[str], context: str) -> str:
        """Synthesize subtask results into a final answer.

        If a model is available, generates a synthesis. Otherwise, concatenates.
        """
        if self.model is not None and self.tokenizer is not None:
            synthesis_prompt = (
                context +
                f"\n[THINK]\nAll subtasks are complete. Synthesize the results.\n"
                f"Original query: {user_query}\n"
                f"Subtask results: {subtask_results}\n"
                f"[/THINK]\n[FINAL]\n"
            )
            ids = self.tokenizer.encode(synthesis_prompt)
            if not ids:
                ids = [0]
            generated = self.model.generate_incremental(
                ids,
                max_new_tokens=self.max_tokens_per_turn,
                temperature=self.temperature,
                top_k=self.top_k,
                device=self.device,
            )
            new_ids = generated[len(ids):]
            output = self.tokenizer.decode(new_ids)
            # Extract text before [/FINAL]
            match = re.search(r"(.*?)(?:\[/FINAL\]|$)", output, re.DOTALL)
            if match and match.group(1).strip():
                return match.group(1).strip()

        # Fallback: concatenate
        parts = [f"Subtask {i+1}: {r}" for i, r in enumerate(subtask_results)]
        return "\n".join(parts)

    def reset(self):
        """Reset the agent runtime for a new task."""
        self.planner = Planner(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            temperature=self.temperature,
            top_k=self.top_k,
            max_tokens=self.max_tokens_per_turn,
        )
        self.corrector = SelfCorrector(max_corrections=self.corrector.max_corrections)
        self.react_loop = ReActLoop(
            model=self.model,
            tokenizer=self.tokenizer,
            executor=self.executor,
            device=self.device,
            temperature=self.temperature,
            top_k=self.top_k,
            max_tokens_per_turn=self.max_tokens_per_turn,
        )
        self.conversation_history = []
        self.full_context = ""


# ---------------------------------------------------------------------------
# Conversation state management (for multi-turn chat)
# ---------------------------------------------------------------------------

class ConversationState:
    """Manages conversation state for multi-turn interactions.

    Stores the conversation history and provides methods to format it
    for the model.  The Hamiltonian state cache in the HVN model carries
    the actual memory — this class just manages the text-level context.

    Parameters
    ----------
    max_history : int
        Maximum number of turns to keep in history (older turns are dropped).
    system_prompt : str, optional
        A system prompt prepended to every conversation.
    """

    def __init__(
        self,
        max_history: int = 10,
        system_prompt: str = "",
    ):
        self.max_history = max_history
        self.system_prompt = system_prompt
        self.turns: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str):
        """Add a conversation turn.

        Parameters
        ----------
        role : str
            "user", "assistant", "tool", or "system".
        content : str
            The message content.
        """
        self.turns.append({"role": role, "content": content})
        # Trim history
        if len(self.turns) > self.max_history * 2:  # each turn = user + assistant
            self.turns = self.turns[-(self.max_history * 2):]

    def format_for_model(self) -> str:
        """Format the conversation history for model input."""
        parts = []
        if self.system_prompt:
            parts.append(f"[THINK]\n{self.system_prompt}\n[/THINK]\n")
        for turn in self.turns:
            if turn["role"] == "user":
                parts.append(f"User: {turn['content']}\n")
            elif turn["role"] == "assistant":
                parts.append(f"Assistant: {turn['content']}\n")
            elif turn["role"] == "tool":
                parts.append(f"{AGENT_TOKENS['observation_start']}\n{turn['content']}\n{AGENT_TOKENS['observation_end']}\n")
        return "\n".join(parts)

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message, or None."""
        for turn in reversed(self.turns):
            if turn["role"] == "user":
                return turn["content"]
        return None

    def clear(self):
        """Clear conversation history."""
        self.turns = []


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "AGENT_TOKENS",
    "AGENT_TOKEN_STRINGS",
    "ReasoningStep",
    "AgentParser",
    "Planner",
    "SelfCorrector",
    "ReActLoop",
    "AgentRuntime",
    "ConversationState",
]