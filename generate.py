"""
HVN Inference Script
====================

Load a trained HVN checkpoint and generate text.

Usage:
    python generate.py --prompt "The " --tokens 300
    python generate.py --checkpoint hvn_checkpoint.pt --prompt "Hamiltonian " --temp 0.5
"""

import argparse
import json
import time
import torch

from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer, generate_text
from hvn.tools import (
    ToolRegistry, ToolCallParser, ToolExecutor, ToolCallingSession,
    create_demo_registry, TOOL_TOKENS,
)
from hvn.agent import AgentRuntime, ConversationState
from hvn.chat import ChatSession, ConversationFormatter


def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    """Load a trained HVN model from checkpoint."""

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = ckpt["vocab_size"]
    d_model = ckpt["d_model"]
    num_layers = ckpt["num_layers"]
    n_heads = ckpt.get("n_heads", 4)  # backward compat
    chunk_size = ckpt.get("chunk_size", 64)  # backward compat

    # Rebuild model with saved hyperparameters
    model = HamiltonianVortexNetwork(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=n_heads,
        chunk_size=chunk_size,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer — try BPE first, fall back to CharTokenizer
    try:
        from hvn.tokenizer import BPETokenizer
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"✓ Loaded BPE tokenizer from {tokenizer_path}")
    except Exception:
        tokenizer = CharTokenizer.load(tokenizer_path)
        print(f"✓ Loaded char tokenizer from {tokenizer_path}")

    print(f"✓ Loaded model: vocab={vocab_size}, d_model={d_model}, layers={num_layers}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained HVN model")
    parser.add_argument("--checkpoint", type=str, default="hvn_checkpoint.pt")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer JSON (default: auto-detect from checkpoint name)")
    parser.add_argument("--prompt", type=str, default="The ", help="Generation prompt")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Multiple prompts separated by '||' for batched generation")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of sequences to generate in parallel (batched generation)")
    parser.add_argument("--tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--tools", action="store_true",
                        help="Enable tool-calling mode (Issue #24). Uses demo tools (calculator, web_search, etc.)")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Maximum tool-calling turns (only used with --tools)")
    parser.add_argument("--agent", action="store_true",
                        help="Enable agentic mode (Issue #25). Multi-step reasoning, planning, self-correction.")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum ReAct iterations per subtask (only used with --agent)")
    parser.add_argument("--no_planning", action="store_true",
                        help="Disable planning phase in agentic mode")
    parser.add_argument("--no_correction", action="store_true",
                        help="Disable self-correction in agentic mode")
    parser.add_argument("--chat", action="store_true",
                        help="Enable interactive chat mode (Issue #28). Multi-turn conversation with persistent Hamiltonian state.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt for chat mode (only used with --chat)")
    parser.add_argument("--save_chat", type=str, default=None,
                        help="Save conversation state to this path when exiting chat mode")
    parser.add_argument("--load_chat", type=str, default=None,
                        help="Load conversation state from this path to resume chat")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect tokenizer path
    tok_path = args.tokenizer or args.checkpoint.replace(".pt", "_tokenizer.json")

    # Load model + tokenizer
    model, tokenizer = load_model(args.checkpoint, tok_path, device)

    # --- Batched generation mode ---
    if not args.tools and (args.prompts or args.batch > 1):
        if args.prompts:
            prompt_strs = [p.strip() for p in args.prompts.split("||") if p.strip()]
        else:
            prompt_strs = [args.prompt] * args.batch

        print(f"\nBatched generation: {len(prompt_strs)} sequences")
        print("-" * 60)
        for i, p in enumerate(prompt_strs):
            print(f"  Seq {i}: \"{p}\"")
        print("-" * 60)

        prompt_ids_list = [tokenizer.encode(p) or [0] for p in prompt_strs]

        t0 = time.time()
        generated_list = model.generate_batched(
            prompt_ids_list,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        elapsed = time.time() - t0

        total_tokens = sum(len(g) - len(p) for g, p in zip(generated_list, prompt_ids_list))
        for i, (gen_ids, prompt_str) in enumerate(zip(generated_list, prompt_strs)):
            output = tokenizer.decode(gen_ids)
            print(f"\n--- Sequence {i} ---")
            print(output)

        print("-" * 60)
        print(f"Generated {total_tokens} tokens across {len(prompt_strs)} sequences "
              f"in {elapsed:.2f}s ({total_tokens/elapsed:.1f} tokens/sec)")
        print(f"Method: batched incremental O(1) per-token ({len(prompt_strs)} parallel sequences)")

    # --- Chat mode (Issue #28) ---
    if args.chat:
        print("\n" + "=" * 60)
        print("CHAT MODE (Issue #28 — Conversation Memory)")
        print("=" * 60)

        if args.load_chat:
            try:
                session = ChatSession.load(
                    args.load_chat, model, tokenizer, device=device,
                    max_new_tokens=args.tokens,
                    temperature=args.temp,
                    top_k=args.top_k,
                )
                print(f"✓ Loaded conversation from {args.load_chat}")
                print(f"  Turns restored: {len(session.get_history())}")
            except Exception as e:
                print(f"✗ Load failed: {e}, starting fresh session")
                session = ChatSession(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.tokens,
                    temperature=args.temp,
                    top_k=args.top_k,
                )
        else:
            session = ChatSession(
                model=model,
                tokenizer=tokenizer,
                device=device,
                system_prompt=args.system_prompt,
                max_new_tokens=args.tokens,
                temperature=args.temp,
                top_k=args.top_k,
            )

        print(f"System: {args.system_prompt}")
        print(f"Max tokens/turn: {args.tokens}, Temperature: {args.temp}, Top-k: {args.top_k}")
        print("=" * 60)

        # If a prompt is provided (not default), send it as the first user message
        if args.prompt and args.prompt != "The ":
            print(f"You> {args.prompt}")
            print("Assistant> ", end="", flush=True)
            response = session.user_turn(args.prompt)
            print(response)

        # Enter interactive loop
        session.interactive()

        # Save on exit if requested
        if args.save_chat:
            session.save(args.save_chat)
            print(f"✓ Conversation saved to {args.save_chat}")

    # --- Agentic mode (Issue #25) ---
    elif args.agent:
        print("\n" + "=" * 60)
        print("AGENTIC MODE (Issue #25)")
        print("=" * 60)

        # Create demo tool registry and executor
        registry = create_demo_registry()
        executor = ToolExecutor(registry)

        print(f"Registered tools: {registry.list_names()}")
        print(f"Planning: {'enabled' if not args.no_planning else 'disabled'}")
        print(f"Self-correction: {'enabled' if not args.no_correction else 'disabled'}")
        print(f"Max iterations: {args.max_iterations}")
        print(f"Prompt: \"{args.prompt}\"")
        print("-" * 60)

        # Create agent runtime
        agent = AgentRuntime(
            model=model,
            tokenizer=tokenizer,
            executor=executor,
            device=device,
            temperature=args.temp,
            top_k=args.top_k,
            max_tokens_per_turn=args.tokens,
            enable_planning=not args.no_planning,
            enable_self_correction=not args.no_correction,
            verbose=True,
        )

        result = agent.run(args.prompt, max_iterations=args.max_iterations)

        print("\n" + "=" * 60)
        print("AGENT RESULT")
        print("=" * 60)

        if result["plan"]:
            print(f"\n--- Plan ({len(result['plan'])} subtasks) ---")
            for i, st in enumerate(result["plan"], 1):
                print(f"  {i}. {st}")

            if result["plan_progress"]:
                pp = result["plan_progress"]
                print(f"\n--- Plan Progress ---")
                print(f"  Completed: {pp['completed']}/{pp['total']}")
                print(f"  Failed: {pp['failed']}/{pp['total']}")
                print(f"  Revisions: {pp['n_revisions']}")

        print(f"\n--- Tool Calls ({len(result['tool_calls'])}) ---")
        for i, tc in enumerate(result["tool_calls"]):
            print(f"  Call {i+1}: {tc['name']}({tc['arguments']})")

        print(f"\n--- Tool Results ({len(result['tool_results'])}) ---")
        for i, tr in enumerate(result["tool_results"]):
            print(f"  Result {i+1}: {tr[:200]}")

        if result["corrections"]:
            print(f"\n--- Corrections ({len(result['corrections'])}) ---")
            for i, corr in enumerate(result["corrections"]):
                print(f"  Correction {i+1}: {corr.get('type', 'unknown')} - {corr.get('error', corr.get('reason', ''))[:200]}")

        print(f"\n--- Reasoning Steps ({len(result['steps'])}) ---")
        for i, step in enumerate(result["steps"]):
            print(f"  Step {i+1} [{step['step_type']}]: {step['content'][:200]}")

        print(f"\n--- Final Answer (iterations: {result['n_iterations']}, success: {result['success']}) ---")
        print(result["answer"])
        print("=" * 60)

    # --- Tool-calling mode (Issue #24) ---
    elif args.tools:
        print("\n" + "=" * 60)
        print("TOOL-CALLING MODE (Issue #24)")
        print("=" * 60)

        # Create demo tool registry and executor
        registry = create_demo_registry()
        executor = ToolExecutor(registry)

        print(f"Registered tools: {registry.list_names()}")
        print(f"Prompt: \"{args.prompt}\"")
        print("-" * 60)

        # Create tool-calling session
        session = ToolCallingSession(
            model=model,
            tokenizer=tokenizer,
            executor=executor,
            device=device,
            max_turns=args.max_turns,
            temperature=args.temp,
            top_k=args.top_k,
            max_tokens_per_turn=args.tokens,
        )

        result = session.run(args.prompt)

        print(f"\n--- Tool Calls ({len(result['tool_calls'])}) ---")
        for i, tc in enumerate(result["tool_calls"]):
            print(f"  Call {i+1}: {tc['name']}({tc['arguments']})")

        print(f"\n--- Tool Results ({len(result['tool_results'])}) ---")
        for i, tr in enumerate(result["tool_results"]):
            print(f"  Result {i+1}: {tr[:200]}")

        if result["thinking"]:
            print(f"\n--- Thinking ({len(result['thinking'])}) ---")
            for i, th in enumerate(result["thinking"]):
                print(f"  Think {i+1}: {th[:200]}")

        print(f"\n--- Final Answer (turns: {result['n_turns']}) ---")
        print(result["answer"])
        print("=" * 60)

    # --- Single-sequence generation (default) ---
    elif not args.prompts and args.batch <= 1:
        # Generate using fast incremental generation (O(1) per token)
        print(f"\nPrompt: \"{args.prompt}\"")
        print("-" * 60)

        prompt_ids = tokenizer.encode(args.prompt)
        if not prompt_ids:
            prompt_ids = [0]

        t0 = time.time()
        generated_ids = model.generate_incremental(
            prompt_ids,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        elapsed = time.time() - t0
        output = tokenizer.decode(generated_ids)

        print(output)
        print("-" * 60)
        print(f"Generated {args.tokens} tokens in {elapsed:.2f}s ({args.tokens/elapsed:.1f} tokens/sec)")
        print(f"Method: incremental O(1) per-token (Hamiltonian state cache)")


if __name__ == "__main__":
    main()