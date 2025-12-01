# SIMT Softmax Engine Simulation with Online Softmax

## 3.1 Baseline: SFU Softmax (without softmax engine)

```bash
python simt_softmax_engine_online.py \
  --num-warps 4 \
  --rows-per-warp 2 \
  --tiles-per-row 8 \
  --sfu-latency 20 \
  --sfu-ii 4
```

This configuration corresponds to:

- Each warp has 2 rows, each row has 8 tiles
- Softmax executed via SFU baseline:
  - Tile latency = 20 cycles
  - Issue interval = 4 → Throughput ≈ 0.25 tile/cycle

Expected observations in output:

- Long total cycle count
- Low SoftmaxUnit utilization and becomes a bottleneck
- Online Softmax state shows `tiles_done = tiles_per_row` (indicating correct sequential dependency)

## 3.2 Using Softmax Engine

```bash
python simt_softmax_engine_online.py \
  --num-warps 4 \
  --rows-per-warp 2 \
  --tiles-per-row 8 \
  --use-softmax-engine \
  --smx-latency 4 \
  --smx-ii 1
```

Softmax engine configuration:

- Pipeline depth = 4
- Issue Interval (II) = 1 → Steady-state throughput ≈ 1 tile/cycle

Expected observations:

- Significantly reduced total cycle count
- Dense "-" characters in both TensorCore and SoftmaxUnit timelines
- Online softmax [tiles_done](file:///home/duanzhenwei/softmax_sim/simt_softmax_engine_online.py#L90-L90) values are fully populated without sequential warnings
```

This polished version improves the document by:
1. Adding a proper title and section headers
2. Using consistent code blocks for commands
3. Converting Chinese text to English for consistency
4. Organizing information with clear bullet points
5. Improving readability with better formatting
6. Maintaining the technical accuracy of the original content