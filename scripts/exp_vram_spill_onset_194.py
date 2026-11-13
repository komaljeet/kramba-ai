"""#194: where does WDDM VRAM spill start on the 6 GB card? Ballast sweep at 1B k5.

Author: Rambarun Komaljeet

Wraps ``scripts/exp_bcd_blocksize_194.py`` (production 1B settings, grad_accum 8,
fp8, balanced partition, --bcd-layers-per-block 5) and changes three things:

  * the BCD rotation starts at the worst blocks: blocks.5-9, blocks.0-4,
    blocks.10-14 (the k5 smoke peaks: 87.6 / 86.8 / 85.2 % reserved). The other
    blocks never train here; their weights stay at init, parked on the CPU, which
    is what they are during these turns in the real rotation too.
  * a ballast tensor of ``--ballast-mib`` MiB is allocated on the GPU right after
    the scheduler is built (after its one-time empty_cache) and held to the end.
    It is written once at allocation and fully rewritten (add_) after every
    optimizer step, so the driver cannot leave it non-resident without paying
    for it in step time. It lives in PyTorch's caching allocator, so
    max_memory_reserved INCLUDES it: peak reserved here = model + ballast, the
    exact number the #163 guard compares to the card.
  * the #163 guard is disabled (--max-vram-pct 0): with blocks.5-9 first, step 1
    is already at 87.6% with no ballast. Nothing else of the trainer changes.

Per optimizer step, ``mem_<tag>.jsonl`` adds what the step rows do not have:
reserved/allocated now, driver free/total (cudaMemGetInfo), allocator retries and
OOM counts, and the ballast size.
"""
import os, sys, json, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ballast_mib = 0
    order = [2, 1, 3]
    argv = sys.argv[1:]
    if "--ballast-mib" in argv:
        i = argv.index("--ballast-mib")
        ballast_mib = int(argv[i + 1]); del argv[i:i + 2]
    tag = argv[argv.index("--tag") + 1]
    out = argv[argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    import hvn.core.block_coordinate as bc
    import train_edge_conversational as T
    import exp_bcd_blocksize_194 as H
    assert os.path.abspath(bc.__file__).startswith(REPO), bc.__file__

    B = {}
    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *x, **k):
        orig_init(self, model, optimizer, *x, **k)
        n = len(self.blocks)
        names = [nm for nm, _ in self.blocks]
        self.order = order + [i for i in range(n) if i not in order]
        self._cycle_pos = 0
        self._step_in_block = 0
        self._active_idx = self.order[0]
        self._activate(self._active_idx)
        torch.cuda.empty_cache()
        print(f"[spill] rotation order {[names[i][:24] for i in self.order]}", flush=True)
        if ballast_mib:
            b = torch.empty(ballast_mib << 20, dtype=torch.uint8, device="cuda")
            b.fill_(1)
            torch.cuda.synchronize()
            B["t"] = b
        torch.cuda.reset_peak_memory_stats()
        print(f"[spill] ballast {ballast_mib} MiB; reserved now "
              f"{torch.cuda.memory_reserved() / 2**20:.0f} MiB", flush=True)
    bc.BlockCoordinateScheduler.__init__ = init

    mf = open(os.path.join(out, f"mem_{tag}.jsonl"), "w")
    cnt = {"n": 0}

    def post(opt, args, kw):
        # registered BEFORE the harness's hook, so it runs first (the harness
        # exits the process from its hook on the last step)
        if "t" in B:
            B["t"].add_(1)            # rewrite every byte: keep the ballast resident
        torch.cuda.synchronize()
        cnt["n"] += 1
        s = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        mf.write(json.dumps({
            "step": cnt["n"], "wall": time.time(), "ballast_mib": ballast_mib,
            "reserved": torch.cuda.memory_reserved(), "allocated": torch.cuda.memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "drv_free": free, "drv_total": total,
            "num_alloc_retries": s.get("num_alloc_retries", 0), "num_ooms": s.get("num_ooms", 0),
            "pid": os.getpid()}) + "\n")
        mf.flush()
    register_optimizer_step_post_hook(post)

    orig_main = T.main

    def tmain():
        sys.argv += ["--max-vram-pct", "0"]
        return orig_main()
    T.main = tmain

    print(f"[spill] pid {os.getpid()}", flush=True)
    sys.argv = [sys.argv[0]] + argv
    H.main()


if __name__ == "__main__":
    main()
