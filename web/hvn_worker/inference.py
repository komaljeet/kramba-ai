"""
Bridges Celery tasks to hvn.chat.ChatSession. Each call loads that
conversation's Hamiltonian state from disk (or starts fresh), runs one user
turn, and saves the state back. Persisting through StateCheckpoint (JSON on
the shared `checkpoints` volume - see docker-compose.yml/docker-prod.yml) is
the source of truth even if a different worker process picks up the next
turn, which is why this doesn't try to keep ChatSession objects in memory
across tasks. A Redis-backed hot-state cache (avoiding the disk round-trip
for active conversations) is a possible future optimization, not needed yet.
"""
import os

from hvn.chat import ChatSession
from hvn.tools import ToolCallParser

from .state import state_path as _state_file
from .worker import get_model


def _state_path(conversation_id) -> str:
    """Save-path variant of ``state.state_path`` — ensures the parent
    directory exists (this module writes the file), while the torch-free
    ``state.state_path`` used by the delete path only needs to locate it."""
    path = _state_file(conversation_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def run_turn(
    conversation_id,
    user_message: str,
    model_tag: str,
    checkpoint_path: str,
    system_prompt: str = "You are a helpful assistant.",
    thinking: bool = True,
    web_search: bool = False,
) -> str:
    """Runs one user turn against the given conversation's persisted
    Hamiltonian state and returns the assistant's reply text.

    ``thinking``/``web_search`` are the per-message toggles plumbed from the UI
    (Issue #56) — see ``stream_turn`` for their exact effect.
    """
    model, tokenizer, device = get_model(model_tag, checkpoint_path)
    state_path = _state_path(conversation_id)

    if os.path.exists(state_path):
        session = ChatSession.load(state_path, model, tokenizer, device=device)
    else:
        session = ChatSession(model, tokenizer, device=device, system_prompt=system_prompt)

    _configure_toggles(session, web_search=web_search)
    response = session.user_turn(user_message)
    session.save(state_path)
    return response if thinking else ToolCallParser.THINK_RE.sub("", response).strip()


def _configure_toggles(session, *, web_search: bool) -> None:
    """Apply the per-message web-search toggle to a ChatSession.

    The web-search toggle gates the ``web_search`` tool (hvn/tools.py, made a
    live backend in #41). ChatSession does not yet run a tool-calling loop, so
    for now this only records the requested state on the session; when the
    agentic/tool loop lands (#25) it reads ``session.web_search_enabled`` to
    decide whether the tool is offered. Threading the flag here means the toggle
    already reaches the worker (Issue #56 acceptance) rather than being wired
    later from scratch.
    """
    session.web_search_enabled = web_search


def stream_turn(
    conversation_id,
    user_message: str,
    model_tag: str,
    checkpoint_path: str,
    result: dict,
    system_prompt: str = "You are a helpful assistant.",
    thinking: bool = True,
    web_search: bool = False,
):
    """Generator variant of run_turn(): yields text deltas as they're
    generated (for Phase 4's WebSocket streaming - see api/tasks.py), then
    saves the conversation's Hamiltonian state once exhausted, same as
    run_turn().

    The caller MUST fully exhaust this generator. `result` is a plain dict
    the caller passes in and reads back afterwards - a generator's `return`
    value is only reachable via StopIteration.value when driving the
    iterator manually, which `for delta in stream_turn(...)` doesn't do, so
    an out-param is simpler than callers needing to catch StopIteration.

    Per-message toggles (Issue #56):
    - ``thinking`` (default True): when False, the model's [THINK]...[/THINK]
      chain-of-thought (#42) is stripped from the persisted ``result["content"]``.
      Raw deltas are still streamed verbatim (the UI decides whether to render a
      live think block), matching api/tasks.py's existing "stream raw, sanitize
      the persisted copy" split.
    - ``web_search`` (default False): gates the web_search tool via
      ``_configure_toggles`` (no-op until the tool loop lands — see there).
    """
    model, tokenizer, device = get_model(model_tag, checkpoint_path)
    state_path = _state_path(conversation_id)

    if os.path.exists(state_path):
        session = ChatSession.load(state_path, model, tokenizer, device=device)
    else:
        session = ChatSession(model, tokenizer, device=device, system_prompt=system_prompt)

    _configure_toggles(session, web_search=web_search)
    yield from session.user_turn_streaming(user_message)

    session.save(state_path)
    content = session.turns[-1]["content"]
    result["content"] = content if thinking else ToolCallParser.THINK_RE.sub("", content).strip()
