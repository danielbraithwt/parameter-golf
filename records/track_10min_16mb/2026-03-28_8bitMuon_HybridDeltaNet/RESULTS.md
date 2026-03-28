# Experiment Results Log

Hardware: 1xH100 80GB, nproc_per_node=1 (grad_accum_steps=8)
Note: Absolute scores will be worse than 8-GPU runs due to fewer steps in 10-min cap. Relative comparisons are valid.

## Phase 1: Baseline Reproduction

| Run | val_bpb | val_loss | compressed_size | total_steps | step_avg_ms | peak_mem_MiB | notes |
|-----|---------|----------|-----------------|-------------|-------------|--------------|-------|
| phase1_baseline | **1.3653** | 2.3053 | 7.3MB (int6+lzma) | 957 | 627.56 | 21636 | Reference: all new features OFF |

## Phase 2: Hybrid DeltaNet Sweep

Fixed: added `use_qk_l2norm_in_kernel=True` and `fullgraph=False` for torch.compile compatibility.

| Run | val_bpb | val_loss | compressed_size | total_steps | step_avg_ms | notes |
|-----|---------|----------|-----------------|-------------|-------------|-------|
| phase2a (full_attn=3,7,10) | pending | | | | | 8 DeltaNet + 3 full attn layers |
| phase2b (full_attn=5,10) | pending | | | | | 9 DeltaNet + 2 full attn layers |
| phase2c (full_attn=10) | pending | | | | | 10 DeltaNet + 1 full attn layer |

## Phase 3: 8-bit Muon + Low-Rank Sweep

| Run | val_bpb | val_loss | compressed_size | total_steps | step_avg_ms | peak_mem_MiB | notes |
|-----|---------|----------|-----------------|-------------|-------------|--------------|-------|
| phase3a (int8 only) | **1.3778** | 2.3264 | 7.2MB | 950 | 631.73 | 21604 | +0.0125 vs baseline, same speed |
| phase3b (int8 + rank64) | **2.5194** | 4.2538 | 4.8MB | 234 | 2573.49 | 21604 | SVD way too slow (4x), dead end |
| phase3c (int8 + rank128) | SKIPPED | | | | | | Would be even slower than rank64 |

## Phase 3 Conclusions

- **int8 momentum**: Marginal degradation (+0.0125 bpb), no speed/memory benefit on 1 GPU. Not worth it.
- **Low-rank SVD truncation**: Fatal step time penalty (4x slower). Dead end.

## Phase 4: Combined Best

| Run | val_bpb | val_loss | compressed_size | total_steps | notes |
|-----|---------|----------|-----------------|-------------|-------|
| (TBD after Phase 2 DeltaNet results) | | | | | |

## Additional Sweeps

### Hyperparameter Sweeps (Phase 5)

| Run | val_bpb | val_loss | total_steps | step_avg_ms | config delta vs baseline | notes |
|-----|---------|----------|-------------|-------------|--------------------------|-------|
| (pending) | | | | | | |
