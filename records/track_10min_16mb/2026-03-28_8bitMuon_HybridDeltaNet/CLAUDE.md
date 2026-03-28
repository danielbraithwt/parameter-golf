# 8-bit Low-Rank Muon + Hybrid DeltaNet Experiment

## Goal

Beat the current SOTA of 1.1194 BPB on FineWeb validation under the 16MB / 10-minute constraint.

Two hypotheses:
1. **Hybrid DeltaNet** — Replacing most softmax attention layers with Gated DeltaNet (linear attention, O(Td^2)) yields faster steps, meaning more training in 10 minutes.
2. **8-bit Low-Rank Muon** — Quantizing momentum to int8 and truncating to top-K singular directions saves memory and may regularize training.

## Base

Built on the current SOTA submission (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, 1.1194 BPB). All existing techniques retained: LeakyReLU^2, Parallel Muon, BigramHash, XSA, ValueEmbedding, EMA, legal TTT, int6+lzma.

## Experiment Schedule

### Phase 1: Baseline reproduction

Run the unmodified SOTA to get a reference number on our hardware.

```bash
DELTANET_ENABLED=0 MUON_INT8_MOMENTUM=0 MUON_LOW_RANK_K=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Record: val_bpb, step_avg_ms, total_steps, peak_memory_MiB.

### Phase 2: Hybrid DeltaNet

Test DeltaNet on/off, then sweep layer placement.

```bash
# 2a: DeltaNet with default placement (full attn at 3,7,10)
DELTANET_ENABLED=1 FULL_ATTN_LAYERS=3,7,10 MUON_INT8_MOMENTUM=0 MUON_LOW_RANK_K=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 2b: Fewer full-attn layers
DELTANET_ENABLED=1 FULL_ATTN_LAYERS=5,10 MUON_INT8_MOMENTUM=0 MUON_LOW_RANK_K=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 2c: Only last layer is full attn
DELTANET_ENABLED=1 FULL_ATTN_LAYERS=10 MUON_INT8_MOMENTUM=0 MUON_LOW_RANK_K=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Record: val_bpb, step_avg_ms for each. Pick best placement.

### Phase 3: 8-bit Muon + Low-Rank

Test int8 momentum alone, then with rank truncation.

```bash
# 3a: 8-bit momentum only
DELTANET_ENABLED=0 MUON_INT8_MOMENTUM=1 MUON_LOW_RANK_K=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 3b: 8-bit + low-rank K=64
DELTANET_ENABLED=0 MUON_INT8_MOMENTUM=1 MUON_LOW_RANK_K=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 3c: 8-bit + low-rank K=128
DELTANET_ENABLED=0 MUON_INT8_MOMENTUM=1 MUON_LOW_RANK_K=128 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Record: val_bpb, peak_memory_MiB for each. Check for degradation or improvement.

### Phase 4: Combined best

Take best DeltaNet config from Phase 2 + best Muon config from Phase 3, run together. If step time drops, try scaling up (more layers or wider model).

## Key Environment Variables

| Variable | Default | What |
|----------|---------|------|
| `DELTANET_ENABLED` | 1 | Toggle hybrid DeltaNet |
| `FULL_ATTN_LAYERS` | 3,7,10 | Which layers keep full softmax attention |
| `DELTANET_BETA_INIT` | 2.0 | DeltaNet gate bias init (sigmoid(2)~0.88) |
| `MUON_INT8_MOMENTUM` | 1 | Quantize momentum buffers to int8 |
| `MUON_LOW_RANK_K` | 0 | Top-K singular directions to keep (0=disabled) |

## Dependencies

Requires `fla-core` for DeltaNet Triton kernels: `pip install fla-core`
