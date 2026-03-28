# Experiment Results Log

Hardware: 1xH100 80GB, nproc_per_node=1 (grad_accum_steps=8)
Note: Absolute scores will be worse than 8-GPU runs due to fewer steps in 10-min cap. Relative comparisons are valid.

## Phase 1: Baseline Reproduction

| Run | val_bpb | val_loss | compressed_size | total_steps | step_avg_ms | peak_mem_MiB | notes |
|-----|---------|----------|-----------------|-------------|-------------|--------------|-------|
| phase1_baseline | **1.3653** | 2.3053 | 7.3MB (int6+lzma) | 957 | 627.56 | 21636 | Reference: all new features OFF |

## Phase 2: Hybrid DeltaNet Sweep

Rewrote GatedDeltaNetAttention with proper architecture: Mamba-style alpha decay gate, per-head beta, output gating (RMSNorm + SiLU), using fla `chunk_gated_delta_rule` kernel with `use_qk_l2norm_in_kernel=True`.

| Run | val_bpb | val_loss | compressed_size | total_steps | step_avg_ms | notes |
|-----|---------|----------|-----------------|-------------|-------------|-------|
| phase2a_v1 (broken, no alpha gate) | 1.6507 | 2.7871 | 5.4MB | 525 | 1143.30 | Old impl, chunk_delta_rule, no gating |
| phase2a_v2 (proper GatedDeltaNet) | **2.0185** | 3.4081 | 5.8MB | 307 | 1956.10 | Full impl, 3x slower than baseline |
| phase2b, 2c | SKIPPED | | | | | DeltaNet too slow on 1GPU to be competitive |

### Phase 2 Conclusions

DeltaNet is a **dead end** for the 1xH100 / 10-min constrained setting:
- `fullgraph=False` (required for fla kernel) loses torch.compile optimizations
- fla chunk_gated_delta_rule kernel is ~3x slower per step than flash attention
- Even with better per-step learning, far fewer total steps -> worse final bpb
- Would need 8 GPUs to make the speed tradeoff viable

## Phase 3: 8-bit Muon + Low-Rank Sweep

| Run | val_bpb | val_loss | compressed_size | total_steps | step_avg_ms | peak_mem_MiB | notes |
|-----|---------|----------|-----------------|-------------|-------------|--------------|-------|
| phase3a (int8 only) | **1.3778** | 2.3264 | 7.2MB | 950 | 631.73 | 21604 | +0.0125 vs baseline, same speed |
| phase3b (int8 + rank64) | **2.5194** | 4.2538 | 4.8MB | 234 | 2573.49 | 21604 | SVD way too slow (4x), dead end |
| phase3c (int8 + rank128) | SKIPPED | | | | | | Would be even slower than rank64 |

### Phase 3 Conclusions

- **int8 momentum**: Marginal degradation (+0.0125 bpb), no speed/memory benefit on 1 GPU. Not worth it.
- **Low-rank SVD truncation**: Fatal step time penalty (4x slower). Dead end.

## Phase 5: Hyperparameter Sweeps (baseline architecture)

Focus: Optimize within the proven baseline architecture. Sweep LR, warmdown, model width/depth, MLP mult.

| Run | val_bpb | val_loss | total_steps | step_avg_ms | config delta vs baseline | notes |
|-----|---------|----------|-------------|-------------|--------------------------|-------|
| (pending) | | | | | | |
