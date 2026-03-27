# CLAUDE.md

## Project Overview

TileLang is a tile-level DSL for high-performance GPU/CPU kernels (GEMM, FlashAttention, etc.), built on TVM. This fork (`cim-v2` branch) adds CIM (Computing-in-Memory) simulation support on top of upstream v0.1.8.

## Repository Layout

- `tilelang/` — Python package (editable install, changes take effect immediately)
- `src/` — C++ TileLang passes and codegen
- `3rdparty/tvm/` — Bundled TVM (submodule)
- `3rdparty/cutlass/` — CUTLASS headers
- `examples/` — Example kernels
- `testing/` — Test suite
- `build/` — Build artifacts (ninja rebuild here for C++ changes)

## Build & Development

```bash
conda activate tilelang
# First time:
pip install -r requirements-dev.txt
pip install -e . -v --no-build-isolation
# After C++ changes:
cd build && ninja
# Python changes: no rebuild needed (editable install)
```

Key: `--no-build-isolation` avoids pip creating a temp build env, which causes stale cmake paths.

## CIM Simulation via T.gemm

### What CIM simulates

CIM (Computing-in-Memory) models a GPU where the B matrix (weights) resides in on-chip memory. In A×B, only A data needs to be loaded; B is accessed via an address hook rather than data transfer. The simulation runs on real GPUs (A100) to measure CIM data-flow performance characteristics.

### How it works

`T.gemm(A, B, C, cim_simulate=True)` generates a kernel identical to baseline except B's ldmatrix (shared → register load) is skipped. This goes through **gemm_v2** (Python lowering via `TensorCoreIntrinEmitter`), not CuTE templates.

Pipeline:
```
T.gemm(cim_simulate=True, cim_micro_m=0, cim_micro_n=0, cim_micro_k=0)
  → gemm_op.py: passes cim params as args 19-22 in call_intrin
  → gemm_py.cc: GemmPyNode parses cimSimulate_, cimMicroM_/N_/K_
  → gemm_mma.py: GemmMMA.lower() skips mma_emitter.ldmatrix_b() when cim_simulate
  → Generated TIR: same ldmatrix_a + ptx_mma, no ldmatrix_b
```

### CIM micro_m/n/k parameters

`T.gemm(..., cim_micro_m=2, cim_micro_n=64, cim_micro_k=32)` specifies the CIM instruction shape. These are actual dimensions (not ratios). Default 0 = use hardware MMA shape.

Currently passed through to C++ `GemmPyNode` for future use. The v2 lower path can be extended to control loop structure based on micro values (TODO).

### Performance results (A100)

| Kernel | Baseline | CIM | Speedup |
|--------|----------|-----|---------|
| FA (fp16, 128x128, 256t, b8h32s4096d128) | 14.88 ms / 147.8 TF | 11.81 ms / 186.3 TF | **1.26x** |
| GEMM (int8, 8192x8192x4096, transpose_B) | 1.79 ms / 307 TOPS | 1.73 ms / 319 TOPS | **1.04x** |

### Files modified (on top of upstream v0.1.8)

- `tilelang/language/gemm_op.py` — `cim_simulate`, `cim_micro_m/n/k` params
- `src/op/gemm_py.h` + `gemm_py.cc` — `cimSimulate_`, `cimMicroM_/N_/K_` fields + reflection + parsing
- `tilelang/tileop/gemm/gemm_base.py` — CIM property accessors
- `tilelang/tileop/gemm/gemm_mma.py` — `lower()` skips B ldmatrix when `cim_simulate`

Zero changes to: C++ codegen, CUDA templates, CuTE, ptx_mma intrinsics.

### CIM-specific example files

T.gemm-level CIM (recommended):
- `examples/gemm/example_gemm_baseline.py` — GEMM baseline
- `examples/gemm/example_gemm_cim.py` — GEMM CIM (supports `--micro_m/n/k`)
- `examples/flash_attention/example_mha_fwd_bshd_baseline.py` — FA baseline
- `examples/flash_attention/example_mha_fwd_bshd_cim.py` — FA CIM (supports `--micro_m/n/k`)

Intrinsic-level CIM (for custom micro_m/n/k loop structure experiments):
- `tilelang/intrinsics/mma_cim_macro_generator.py` — CIM emitter subclass
- `examples/gemm/example_gemm_baseline_intrinsic.py` — GEMM baseline (intrinsic, `--dtype int8/float16`)
- `examples/gemm/example_gemm_cim_intrinsic.py` — GEMM CIM (intrinsic, `--dtype --micro_m/n/k --cim_stride_index --tracekernel`)
- `examples/gemm/example_gemm_intrinsic_kernel.py` — Intrinsic kernel with CUDA postproc
- `examples/flash_attention/example_mha_fwd_bshd_cim_intrinsic.py` — FA CIM (intrinsic)

### Intrinsic CIM architecture (3 principles)

The intrinsic CIM emitter (`CIMTensorCoreIntrinEmitter`) generates MMA instructions based on three principles:

1. **C_local = warp_m × warp_n / 32** — output accumulator size determined by warp tile, constant across micro configs.
2. **A_local = warp_m × micro_k / 32** — determined by dtype and micro_k. Each ki step loads the full A_local via ldmatrix (multiple K sub-passes when micro_k > mma_k). ki iterations = block_K / micro_k.
3. **MMA count = (warp_m/micro_m) × (warp_n/micro_n) per ki** — each GPU MMA represents one CIM instruction. The number of CIM instructions per ki is controlled by micro_m/n.

A/C register indexing supports two modes (`cim_stride_index`):
- **hw cycling** (default, `False`): A/C indices cycle through hardware MMA positions (`i % hw_warp_rows`). Same access pattern as real GPU, no register bank conflict. Better performance on A100.
- **cim stride** (`True`): A/C indices use CIM micro-based strides (`micro_m × micro_k / 32`). Models the CIM architecture more accurately, but causes overlapping register reads and worse GPU performance.

### Intrinsic CIM key parameters

| Parameter | Controls | Default |
|-----------|----------|---------|
| `micro_m/n` | MMA count per ki, CIM instruction shape | = MMA shape (16/8) |
| `micro_k` | ki loop count, A load frequency | = mma_k (32 for int8) |
| `fake_instr_m/n/k` | Hardware MMA shape (auto from dtype) | 16/8/32 (int8) |
| `cim_stride_index` | Register index mode | False (hw cycling) |
| `T.sync_threads()` between A/B copies | CIM-specific optimization (improves DRAM locality) | Optional |

## Syncing with Upstream

Current CIM branch (`cim-v2`) is based on **upstream v0.1.8** (`git tag v0.1.8`, commit `41b25527`).

CIM changes are minimal (4 upstream files modified). After syncing:
1. Check `gemm_py.h/cc` — CIM fields may need updating if GemmPyNode changes
2. Check `gemm_mma.py` — CIM lower logic may need updating if `_gemm_ssr`/`_gemm_rsr` change
3. Run `examples/gemm/example_gemm_cim.py` and `examples/flash_attention/example_mha_fwd_bshd_cim.py` as smoke tests

`mma_cim_macro_generator.py` (intrinsic path) subclasses upstream's `TensorCoreIntrinEmitter`. An import-time guard warns if the base class interface drifts.

## Conventions

- Do not use `@simplify_prim_func` with `@tilelang.jit` — triggers an upstream AST parser bug with `*args`.
- CIM changes only touch Python-level lowering (gemm_v2 path). No C++ template or codegen modifications.
