# CLAUDE.md

## Project Overview

TileLang is a tile-level DSL for high-performance GPU/CPU kernels (GEMM, FlashAttention, etc.), built on TVM. This fork (`cim-v2` branch) adds CIM (Computing-in-Memory) simulation support on top of upstream v0.1.8.

## Repository Layout

- `tilelang/` — Python package (editable install, changes take effect immediately)
- `src/` — C++ TileLang passes and codegen
- `src/tl_templates/cuda/` — CUDA template headers (included at NVRTC compile time)
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

## CIM Simulation Architecture

### What CIM simulates

CIM (Computing-in-Memory) models a hypothetical GPU where the B matrix (weights) resides in on-chip memory. In A×B, only A data needs to be loaded; B is accessed via an address hook rather than data transfer. The simulation currently runs on real GPUs (A100) to measure CIM data-flow performance characteristics.

### Three-layer parameter system

CIM kernels decouple three concerns:

**Layer 1: `A_in_dtype` / `B_in_dtype`** — The simulated data type for memory traffic. Controls data packing ratios and copy strides (e.g., int8 packs 2× denser than fp16). Only affects global→shared copy shape calculations (`A_shape`, `B_shape`, `data_map[dtype] // 16`).

**Layer 2: `micro_m/n/k`** — The simulated CIM instruction shape. Can be arbitrary (e.g., 1×64×64). Controls:
- Loop structure: `for ki in T.serial(block_K // micro_k)`
- Warp tiling: `warp_rows = warp_row_tiles // micro_m`
- Buffer allocation sizes: `local_size_a = (micro_m * micro_k) // warp_size`
- C_shared shape: `(block_M // micro_m, block_N // micro_n, fake_instr_m, fake_instr_n)`

**Layer 3: `fake_instr_m/n/k`** — The real GPU MMA instruction shape. Must be a valid PTX MMA shape (e.g., `m16n8k16` for fp16). The emitter is **always hardcoded to `a_dtype="float16", b_dtype="float16"`** regardless of `A_in_dtype`/`B_in_dtype`. This means the actual PTX instruction is always an fp16 MMA, even when the simulated data type is int8 or int4. This is intentional: CIM simulation measures data-flow latency, not computational correctness.

Valid `fake_instr` shapes (determined by the hardcoded fp16 dtype):
- `m16n8k16` (default, most common)
- Other shapes would require changing the hardcoded dtype in the emitter

**Data positions within `ldmatrix_a` and `mma` are governed by `fake_instr`, not `micro`.** The CIM micro shape controls the loop cadence and buffer sizing (simulating "how data flows if CIM hardware processed micro_m×micro_n×micro_k per instruction"), while the actual shared-memory access pattern follows the real GPU instruction layout.

### Code path through the stack

1. **Python API** (`tilelang/language/tir/op.py`): `T.ptx_mma(..., cim_simulate=True)` appends a boolean flag to the TIR call args.

2. **C++ codegen** (`src/target/codegen_cuda.cc`): Detects the cim flag. When active, emits `tl::mma_sync<..., false, true>` (Saturate=false, CimSimulate=true). Non-CIM path is unchanged.

3. **CUDA template** (`src/tl_templates/cuda/instruction/mma.h`): `MmaDispatcher` with `CimSimulate=true` calls `call_fma_cim_simulation` instead of `call_fma`. This replaces all B register values with `&b[0]`'s address:
   ```cpp
   const auto b0_addr = reinterpret_cast<std::uintptr_t>(&b[0]);
   Impl::fma(d[DIdx]..., a[AIdx]...,
             ((void)BIdx, static_cast<BReg>(b0_addr))..., c[CIdx]...);
   ```

4. **CIM Macro Generator** (`tilelang/intrinsics/mma_cim_macro_generator.py`): Subclasses upstream `TensorCoreIntrinEmitter`. Key overrides:
   - `ldmatrix_b`: loads only 1 tile (address hook, not full data)
   - `mma`: passes `B_shared.access_ptr(offset=...)` + `cim_simulate=True`
   - `stmatrix`: respects `fake_warp_rows/cols`

### Shared memory swizzle

Swizzle (XOR-based bank conflict avoidance) is **preserved in CIM mode**. The `T.annotate_layout` + `make_swizzle_layout` annotations on A_shared produce identical XOR patterns in cp_async (global→shared) and ldmatrix (shared→local) as the upstream non-CIM path.

### CIM-specific files

- `tilelang/intrinsics/mma_cim_macro_generator.py` — CIM emitter (subclass, ~320 lines)
- `examples/gemm/example_gemm_cim_simulate.py` — CIM GEMM benchmark
- `examples/gemm/example_gemm_intrinsic_kernel.py` — Intrinsic-level kernel with CUDA postproc
- `examples/gemm/example_gemm_mma_intrinsic_baseline.py` — Non-CIM baseline for comparison
- `examples/flash_attention/example_mha_fwd_bshd_cim_simulate.py` — CIM FlashAttention

## Syncing with Upstream

Current CIM branch (`cim-v2`) is based on **upstream v0.1.8** (`git tag v0.1.8`, commit `41b25527`).

`mma_cim_macro_generator.py` is a **subclass override** of upstream's `mma_macro_generator.py::TensorCoreIntrinEmitter`. It overrides 6 methods (`__init__`, `_initialize_k_dim`, `_initialize_micro_size`, `ldmatrix_a`, `ldmatrix_b`, `mma`, `stmatrix`) and inherits everything else. This means:

- **On every upstream sync**, check whether `mma_macro_generator.py` has changed. If the base class `__init__` signature, internal attribute names, or overridden method signatures changed, `mma_cim_macro_generator.py` must be updated accordingly.
- An import-time compatibility guard (`_EXPECTED_BASE_INIT_PARAMS`, `_EXPECTED_BASE_METHODS`) will emit a warning if the base class interface has drifted.
- After syncing, always run `examples/gemm/example_gemm_cim_simulate.py` as a smoke test.

## Conventions

- When modifying C++ codegen for CIM, keep the non-CIM code path identical to upstream (conditional emission only when `cim_flag=true`).
