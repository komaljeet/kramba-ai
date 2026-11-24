"""#216: paper estimate of streamed full backprop at 1B. No GPU, no torch.

Every input below is a measured number with its source, or a size from the
parameter count. Run: python scripts/estimate_streamed_fullbp_216.py
Output is copied into results/streamed_fullbp_est_216/estimate_numbers.txt.
"""

GB = 1e9

# ---- sizes (results/bcd_memory/ga8_k4.log: "Muon ... 210 matrices / 1,009,516,544 params;
#      AdamW aux ... 103,543,249 params"; embedding block = 102,926,336 active params) ----
P_MUON = 1_009_516_544          # all 24 layers' matrices
P_ADAM = 103_543_249            # embedding (tied decoder) + 1-D params
P_EMB = 102_926_336
N_LAYERS, N_ATTN = 24, 2        # attention at layers 6 and 18
D, SEQ, MB = 2048, 4096, 8      # d_model, tokens per micro-batch, micro-batches per step
HID = D * SEQ * 4               # one micro-batch hidden state, fp32 (#202: 4 MB per 512-pos slice)
K = 4                           # layers per segment (4 layers/block max in VRAM)
N_SEG = N_LAYERS // K

# ---- measured times, seconds per step of 8 micro-batches (results/speed_hunt_213/
#      perlayer_default.txt and phases_default.txt, main 2845f80, copied into this folder) ----
FWD_DN_ACTIVE = 0.130           # trainable DeltaNet layer forward (129-132 ms)
FWD_ATTN_ACTIVE = 0.163         # trainable attention layer forward
BWD_DN_ACTIVE = 0.430           # trainable DeltaNet recompute + backward with weight grads (420-433 ms)
BWD_ATTN_ACTIVE = 0.627         # trainable attention recompute + backward
LOSS_TRAINABLE_DECODER = 0.455 + 1.669 + 0.063   # embedding turn: loss.fwd + bwd.loss_recompute + fwd.embed
MUON_PER_4LAYER = [0.528, 0.521, 0.530, 0.529, 0.521, 0.529]  # optim, blocks 0-3 .. 20-23
ADAMW_EMB = 0.052               # optim on the embedding turn
MISC = 0.03                     # clip + glue + post (each <= 0.012)
BCD_STEP = 9.89                 # #210 / #215

# ---- measured transfer rates (results/bcd_stall_200/stall_steps_ctrl.jsonl, profiled step 3,
#      token_embedding: h2d_on_compute 16.84 GB in 1.568 s = pageable sync; side stream 15.49 GB
#      in 1.187 s = pinned staging). results/frozen_layer_203/peak.json: h2d_pinned 13.44 GB/s. ----
H2D_PINNED = 15.494591008 / 1.18676002
H2D_PAGEABLE = 16.840996928 / 1.567698742
# D2H: NOT MEASURED on this card. Only used as a break-even below.

# ---- host RAM (results/frozen_codes_204/time_base*.jsonl, codes off, 4 DataLoader workers) ----
WS_MAX = 5.09 * GB              # trainer process working set, max
AVAIL_MIN = 3.24 * GB           # lowest free system RAM seen
SYS_TOTAL = 16.42 * GB
PARKED_BF16_TODAY = P_MUON * 2  # frozen 2-D weights parked in bf16 (block_coordinate.py park_dtype_for)
WORKER_RAM = (7.16 - 6.04) / 2 * GB   # docs/state.md 2026-10-23: 4 vs 2 workers, peak total RSS

def main():
    out = []
    p = out.append
    fwd = (N_LAYERS - N_ATTN) * FWD_DN_ACTIVE + N_ATTN * FWD_ATTN_ACTIVE
    # two-level checkpoint: re-forward layers 0..K-2 of each segment (16 DeltaNet + layers 6, 18)
    refwd = (N_SEG * (K - 1) - N_ATTN) * FWD_DN_ACTIVE + N_ATTN * FWD_ATTN_ACTIVE
    bwd = (N_LAYERS - N_ATTN) * BWD_DN_ACTIVE + N_ATTN * BWD_ATTN_ACTIVE
    muon = sum(MUON_PER_4LAYER)
    opt = muon + ADAMW_EMB
    comp_L2 = fwd + refwd + bwd + LOSS_TRAINABLE_DECODER + opt + MISC
    comp_L1 = comp_L2 - refwd
    p("compute, s/step (8 micro-batches)")
    for k, v in [("forward, all 24 layers trainable", fwd), ("re-forward inside segments (L2 only)", refwd),
                 ("recompute + backward with weight grads", bwd), ("loss + embedding, trainable decoder", LOSS_TRAINABLE_DECODER),
                 ("Muon, all 210 matrices, on GPU", muon), ("AdamW embedding", ADAMW_EMB), ("misc", MISC),
                 ("TOTAL L2 (segment inputs parked)", comp_L2), ("TOTAL L1 (every layer input parked)", comp_L1)]:
        p(f"  {k:45s} {v:7.2f}")
    p(f"  Muon per param on GPU: {muon / P_MUON * 1e9:.2f} ns; AdamW: {ADAMW_EMB / P_EMB * 1e9:.2f} ns")

    master = P_MUON * 4
    mom = P_MUON * 4
    adam = P_ADAM * 8
    seg_in = N_SEG * MB * HID
    all_in = N_LAYERS * MB * HID
    h2d = 3 * master + mom + seg_in + adam    # masters for fwd, re-fwd, bwd; momentum; seg inputs; Adam
    d2h = seg_in + master + mom + adam
    p("\ntransfers per step, GB (L2)")
    p(f"  H2D {h2d / GB:.2f}  = 3 x fp32 masters {master / GB:.2f} + momentum {mom / GB:.2f} + segment inputs {seg_in / GB:.2f} + Adam {adam / GB:.2f}")
    p(f"  D2H {d2h / GB:.2f}  = segment inputs + masters + momentum + Adam")
    p(f"  measured H2D pinned {H2D_PINNED:.2f} GB/s, pageable {H2D_PAGEABLE:.2f} GB/s; D2H unmeasured")
    p(f"  H2D busy at pinned rate {h2d / GB / H2D_PINNED:.2f} s = {h2d / GB / H2D_PINNED / comp_L2:.0%} of compute")

    stall = 0.10
    low = comp_L1 + stall
    likely = comp_L2 + stall
    ser = h2d / GB / H2D_PAGEABLE + d2h / GB / H2D_PAGEABLE
    high = (comp_L2 + ser) * (10.32 / 9.89)
    p("\nstep time, s  (days = 925,000 x step / 86400)")
    for k, v in [("low (L1, copies hidden)", low), ("likely (L2, copies hidden)", likely),
                 ("high (L2, all copies serial at pageable rate, +4.3% whole-run)", high), ("BCD k4 today", BCD_STEP)]:
        p(f"  {k:62s} {v:6.2f} s  {925_000 * v / 86400:6.1f} days  {v / BCD_STEP:4.2f}x BCD")
    be = d2h / GB / (40 - comp_L2 - h2d / GB / H2D_PAGEABLE)
    p(f"  break-even D2H rate for a 40 s step (H2D serial at pageable): {be:.2f} GB/s")
    # equal quality, #214 208M: BCD needs 6.7x the steps at step 4000
    p(f"  full BP steps matching BCD's 925,000 (6.7x, 208M): {925_000 / 6.7:,.0f} -> {925_000 / 6.7 * likely / 86400:.1f} days at likely")

    rest = WS_MAX - PARKED_BF16_TODAY
    avail = WS_MAX + AVAIL_MIN
    ram_L2 = rest + master + mom + adam + seg_in
    ram_L1 = rest + master + mom + adam + all_in
    lean = ram_L2 - P_MUON * 2 - P_MUON * 2 - seg_in / 2
    p("\nhost RAM, GB")
    p(f"  rest of trainer process today (ws {WS_MAX / GB:.2f} - bf16 parked {PARKED_BF16_TODAY / GB:.2f}) {rest / GB:.2f}")
    p(f"  fp32 masters {master / GB:.2f}, Muon momentum fp32 {mom / GB:.2f}, AdamW fp32 {adam / GB:.2f}, segment inputs {seg_in / GB:.2f}, all layer inputs {all_in / GB:.2f}")
    p(f"  need L2 exact {ram_L2 / GB:.2f}; L1 exact {ram_L1 / GB:.2f}; L2 lean (bf16 momentum, bf16 master+SR, bf16 seg inputs) {lean / GB:.2f}")
    p(f"  available to the trainer today (ws max + lowest free) {avail / GB:.2f}; with 0 workers +{4 * WORKER_RAM / GB:.2f} = {(avail + 4 * WORKER_RAM) / GB:.2f}")
    p(f"  everything else on the PC {(SYS_TOTAL - avail) / GB:.2f}; with 32 GB installed (x2 usable) available ~{(2 * SYS_TOTAL - (SYS_TOTAL - avail)) / GB:.1f}")

    emb = P_EMB * 4 * 2
    per_layer = P_MUON / N_LAYERS * 4
    layer = per_layer * 3 + P_MUON / N_LAYERS * 2 + per_layer + 2 * per_layer
    seg_layer_in = K * MB * HID
    hgrad = 2 * MB * HID
    work = (0.3 * GB, 0.6 * GB)
    lo = emb + layer + seg_layer_in + hgrad + work[0]
    hi = emb + layer + seg_layer_in + hgrad + work[1]
    p("\nVRAM allocated at the backward peak, GB (L2, estimate)")
    p(f"  embedding + grad {emb / GB:.2f}; current layer master/grad/momentum + fp8 copies + next in + previous out {layer / GB:.2f}")
    p(f"  segment layer inputs, 8 mb {seg_layer_in / GB:.2f}; hidden grads in+out {hgrad / GB:.2f}; working set {work[0] / GB:.1f}-{work[1] / GB:.1f}")
    p(f"  total {lo / GB:.2f}-{hi / GB:.2f} (today 1B k4: 4.99 allocated / 5.30 reserved = 82%)")
    p(f"  reserved at today's ratio 1.06: {lo * 1.062 / GB:.2f}-{hi * 1.062 / GB:.2f} = {lo * 1.062 / 6.44e9:.0%}-{hi * 1.062 / 6.44e9:.0%} of the card")
    print("\n".join(out))


if __name__ == "__main__":
    main()
