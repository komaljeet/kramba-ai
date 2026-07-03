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

from django.conf import settings

from .worker import get_model


def _state_path(conversation_id) -> str:
    conv_dir = os.path.join(settings.HVN_CHECKPOINT_DIR, "conversations")
    os.makedirs(conv_dir, exist_ok=True)
    return os.path.join(conv_dir, f"{conversation_id}.json")


def run_turn(
    conversation_id,
    user_message: str,
    model_tag: str,
    checkpoint_path: str,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Runs one user turn against the given conversation's persisted
    Hamiltonian state and returns the assistant's reply text."""
    model, tokenizer, device = get_model(model_tag, checkpoint_path)
    state_path = _state_path(conversation_id)

    if os.path.exists(state_path):
        session = ChatSession.load(state_path, model, tokenizer, device=device)
    else:
        session = ChatSession(model, tokenizer, device=device, system_prompt=system_prompt)

    response = session.user_turn(user_message)
    session.save(state_path)
    return response


def stream_turn(
    conversation_id,
    user_message: str,
    model_tag: str,
    checkpoint_path: str,
    result: dict,
    system_prompt: str = "You are a helpful assistant.",
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
    """
    model, tokenizer, device = get_model(model_tag, checkpoint_path)
    state_path = _state_path(conversation_id)

    if os.path.exists(state_path):
        session = ChatSession.load(state_path, model, tokenizer, device=device)
    else:
        session = ChatSession(model, tokenizer, device=device, system_prompt=system_prompt)

    yield from session.user_turn_streaming(user_message)

    session.save(state_path)
    result["content"] = session.turns[-1]["content"]
