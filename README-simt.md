# SIMT Softmax Engine Simulation

## Running SFU Baseline (without softmax engine)

```bash
python3 simt_softmax_engine.py \
  --num-warps 4 \
  --tiles-per-warp 8 \
  --sfu-latency 20 \
  --sfu-ii 4
```

This corresponds to softmax execution on ALU/SFU with lower throughput:

- One tile requires 20 cycles
- Issue interval = 4 → Throughput ~0.25 tile/cycle

## Running with Softmax Engine

```bash
python3 simt_softmax_engine.py \
  --num-warps 4 \
  --tiles-per-warp 8 \
  --use-softmax-engine \
  --smx-latency 4 \
  --smx-ii 1
```

Configuration:
- Softmax engine pipeline depth = 4
- II = 1 → Ideal steady-state throughput = 1 tile/cycle, matching Tensor Core

## Comparison Metrics

Between these two versions, you can compare:

- Total cycles → End-to-end latency for the entire QK^T → Softmax → PV pipeline
- TensorCore utilization → Whether TC is slowed down by softmax
- Softmax utilization → Whether it can actually achieve ~1 tile/cycle throughput

## Mapping to Previous Analysis

### SFU Baseline
Corresponds to the previously mentioned "seq-wise / head-wise softmax on sequential vector SIMD core" where exp2 and reduce operations leave functional units idle ~50% of the time, resulting in lower throughput. This is abstracted as a SOFTMAX_SFU instruction with low throughput.

### Softmax Engine
Corresponds to a dedicated engine placed alongside Tensor Core with internal max/exp/sum/rcp pipelines that can:
- Externally process 1 tile = 1 instruction
- Internally allow multiple tiles to flow through different pipeline stages in parallel
- Achieve steady-state output of 1 tile/cycle, "folding" softmax power/area into throughput rather than wasting it on FU idle time

### SIMT Semantics
Multiple warps share the same Tensor Core + softmax engine:
- If softmax is too slow, Tensor Core will frequently idle in the timeline
- If softmax engine throughput is high enough, TC can remain almost continuously busy
```

This polished version improves the formatting by:
1. Adding proper markdown headers and structure
2. Using consistent code blocks for commands
3. Organizing information into logical sections
4. Converting Chinese comments to English for consistency
5. Improving readability with bullet points and clear explanations
6. Matching the professional style of README.md