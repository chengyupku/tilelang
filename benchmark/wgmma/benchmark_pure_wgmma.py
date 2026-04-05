#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import tempfile


CUDA_SOURCE = r"""
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include "src/tl_templates/cuda/instruction/wgmma.h"

#ifndef WG_N
#define WG_N 64
#endif

#ifndef ACCUMULATORS
#define ACCUMULATORS 1
#endif

static_assert(WG_N >= 8 && WG_N <= 256 && WG_N % 8 == 0,
              "WG_N must be a multiple of 8 in [8, 256].");
static_assert(ACCUMULATORS > 0, "ACCUMULATORS must be positive.");

using namespace cute;
using namespace cutlass::gemm::collective::detail;

constexpr int kWgM = 64;
constexpr int kWgK = 16;
constexpr int kThreadsPerWarpgroup = 128;
constexpr int kWarpgroupsPerBlock = 1;
constexpr int kThreadsPerBlock = kThreadsPerWarpgroup * kWarpgroupsPerBlock;
constexpr int kAccRegsPerWarpgroup = WG_N / 2;
constexpr int kFlopsPerWgmma = kWgM * WG_N * kWgK * 2;

using SmemLayoutAtomB =
    decltype(ss_smem_selector<SM90::GMMA::Major::K, half_t, Int<WG_N>, Int<kWgK>>());
using BLayout =
    decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<WG_N>, Int<kWgK>>{}, Step<_1, _2>{}));

#define CUDA_CHECK(stmt)                                                        \
  do {                                                                          \
    cudaError_t err__ = (stmt);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s failed at %s:%d: %s\n", #stmt, __FILE__,         \
                   __LINE__, cudaGetErrorString(err__));                        \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

struct Options {
  int device = 0;
  int blocks = 0;
  int iters = 200000;
  int warmup = 10;
  int runs = 20;
};

void print_usage(const char* argv0) {
  std::fprintf(stderr,
               "Usage: %s [--device N] [--blocks N] [--iters N] [--warmup N] "
               "[--runs N]\n",
               argv0);
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto require_value = [&](const char* flag) -> int {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
      }
      return std::atoi(argv[++i]);
    };

    if (arg == "--device") {
      opts.device = require_value("--device");
    } else if (arg == "--blocks") {
      opts.blocks = require_value("--blocks");
    } else if (arg == "--iters") {
      opts.iters = require_value("--iters");
    } else if (arg == "--warmup") {
      opts.warmup = require_value("--warmup");
    } else if (arg == "--runs") {
      opts.runs = require_value("--runs");
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown flag: %s\n", arg.c_str());
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (opts.blocks < 0 || opts.iters <= 0 || opts.warmup < 0 || opts.runs <= 0 ||
      opts.device < 0) {
    std::fprintf(stderr, "All numeric arguments must be non-negative, and "
                         "--iters/--runs must be positive.\n");
    std::exit(EXIT_FAILURE);
  }

  return opts;
}

extern "C" __global__ __launch_bounds__(kThreadsPerBlock)
void wgmma_bench(uint32_t* out, int iters) {
  struct alignas(128) SharedStorage {
    half_t b[cosize_v<BLayout>];
  };
  __shared__ SharedStorage shared;

  Tensor sB = make_tensor(make_smem_ptr(shared.b), BLayout{});
  const uint64_t desc_b =
      static_cast<uint64_t>(SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(sB));

  uint32_t a_regs[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
  float acc[ACCUMULATORS][kAccRegsPerWarpgroup];

#pragma unroll
  for (int acc_idx = 0; acc_idx < ACCUMULATORS; ++acc_idx) {
#pragma unroll
    for (int reg_idx = 0; reg_idx < kAccRegsPerWarpgroup; ++reg_idx) {
      acc[acc_idx][reg_idx] = static_cast<float>(acc_idx * kAccRegsPerWarpgroup + reg_idx + 1);
    }
  }

#pragma unroll 1
  for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      cute::warpgroup_fence_operand(a_regs[i]);
    }

#pragma unroll
    for (int acc_idx = 0; acc_idx < ACCUMULATORS; ++acc_idx) {
#pragma unroll
      for (int reg_idx = 0; reg_idx < kAccRegsPerWarpgroup; ++reg_idx) {
        cute::warpgroup_fence_operand(acc[acc_idx][reg_idx]);
      }
    }

    cute::warpgroup_arrive();

#pragma unroll
    for (int acc_idx = 0; acc_idx < ACCUMULATORS; ++acc_idx) {
      tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32,
                   kWgM, WG_N, kWgK, false, true>(
          a_regs, desc_b, reinterpret_cast<uint32_t*>(acc[acc_idx]), iter != 0);
    }

    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();

#pragma unroll
    for (int acc_idx = 0; acc_idx < ACCUMULATORS; ++acc_idx) {
#pragma unroll
      for (int reg_idx = 0; reg_idx < kAccRegsPerWarpgroup; ++reg_idx) {
        cute::warpgroup_fence_operand(acc[acc_idx][reg_idx]);
      }
    }
  }

  if (threadIdx.x == 0) {
    uint32_t sink_bits = 0;
#pragma unroll
    for (int acc_idx = 0; acc_idx < ACCUMULATORS; ++acc_idx) {
#pragma unroll
      for (int reg_idx = 0; reg_idx < kAccRegsPerWarpgroup; ++reg_idx) {
        sink_bits ^= __float_as_uint(acc[acc_idx][reg_idx]);
      }
    }
    out[blockIdx.x] = sink_bits;
  }
}

int main(int argc, char** argv) {
  const Options opts = parse_args(argc, argv);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "No CUDA device available.\n");
    return EXIT_FAILURE;
  }
  if (opts.device >= device_count) {
    std::fprintf(stderr, "Requested device %d, but only %d CUDA devices are "
                         "visible.\n",
                 opts.device, device_count);
    return EXIT_FAILURE;
  }

  CUDA_CHECK(cudaSetDevice(opts.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));

  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, wgmma_bench));

  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm, wgmma_bench, kThreadsPerBlock, 0));

  const int blocks =
      opts.blocks > 0 ? opts.blocks : prop.multiProcessorCount * max_blocks_per_sm * 2;
  const size_t out_elems = static_cast<size_t>(blocks);
  const size_t out_bytes = out_elems * sizeof(uint32_t);

  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

  std::vector<uint32_t> h_out(out_elems, 0u);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < opts.warmup; ++i) {
    wgmma_bench<<<blocks, kThreadsPerBlock>>>(d_out, opts.iters);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float best_ms = std::numeric_limits<float>::max();
  float total_ms = 0.0f;

  for (int i = 0; i < opts.runs; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    wgmma_bench<<<blocks, kThreadsPerBlock>>>(d_out, opts.iters);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    best_ms = elapsed_ms < best_ms ? elapsed_ms : best_ms;
    total_ms += elapsed_ms;
  }

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  unsigned long long checksum = 0;
  for (uint32_t v : h_out) {
    checksum += static_cast<unsigned long long>(v);
  }

  const double total_wgmmas = static_cast<double>(blocks) * ACCUMULATORS * opts.iters;
  const double total_flops = total_wgmmas * static_cast<double>(kFlopsPerWgmma);
  const double best_tflops = total_flops / static_cast<double>(best_ms) / 1.0e9;
  const double avg_ms = total_ms / opts.runs;
  const double avg_tflops = total_flops / static_cast<double>(avg_ms) / 1.0e9;

  std::printf("device=%s sm=%d sms=%d\n", prop.name, prop.major * 10 + prop.minor,
              prop.multiProcessorCount);
  std::printf("shape=m64n%dk16 variant=rs dtype=f16xf16->f32 "
              "threads_per_block=%d warpgroups_per_block=%d\n",
              WG_N, kThreadsPerBlock, kWarpgroupsPerBlock);
  std::printf("blocks=%d accumulators=%d iters=%d max_blocks_per_sm=%d "
              "active_warpgroups_per_sm=%d occupancy=%0.4f\n",
              blocks, ACCUMULATORS, opts.iters, max_blocks_per_sm,
              max_blocks_per_sm * kWarpgroupsPerBlock,
              static_cast<double>(max_blocks_per_sm * kThreadsPerBlock) /
                  prop.maxThreadsPerMultiProcessor);
  std::printf("num_regs=%d static_smem_bytes=%zu dynamic_smem_bytes=%d\n", attr.numRegs,
              attr.sharedSizeBytes, attr.maxDynamicSharedSizeBytes);
  std::printf("wgmmas_per_warpgroup=%lld total_wgmmas=%lld total_flops=%0.0f\n",
              static_cast<long long>(ACCUMULATORS) * opts.iters,
              static_cast<long long>(total_wgmmas), total_flops);
  std::printf("best_ms=%0.6f avg_ms=%0.6f best_tflops=%0.2f avg_tflops=%0.2f\n",
              best_ms, avg_ms, best_tflops, avg_tflops);
  std::printf("checksum=%llu\n", checksum);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_out));
  return EXIT_SUCCESS;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and run a near-pure Hopper WGMMA benchmark. "
            "The hot loop issues only RS-WGMMA plus required warpgroup "
            "arrive/commit/wait instructions, and performs exactly one "
            "global-memory store per block at the end."
        )
    )
    parser.add_argument(
        "--arch",
        default="auto",
        help="Target SM. For Hopper WGMMA this should normally be sm_90a.",
    )
    parser.add_argument(
        "--wg-n",
        type=int,
        default=64,
        help="WGMMA N dimension. Must be a multiple of 8 in [8, 256].",
    )
    parser.add_argument(
        "--accumulators",
        type=int,
        default=1,
        help="Independent accumulator sets issued per loop iteration.",
    )
    parser.add_argument(
        "--cuda-home",
        default=None,
        help="CUDA toolkit root. Auto-detected from CUDA_HOME, /usr/local/cuda, or nvcc.",
    )
    parser.add_argument("--nvcc", default=None, help="Explicit nvcc path.")
    parser.add_argument("--cuobjdump", default=None, help="Explicit cuobjdump path.")
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Build directory. Defaults to a temporary directory unless --keep-build-dir is set.",
    )
    parser.add_argument(
        "--keep-build-dir",
        action="store_true",
        help="Keep generated source and binary on disk.",
    )
    parser.add_argument(
        "--dump-sass",
        action="store_true",
        help="Print the SASS of the wgmma_bench kernel after compilation.",
    )
    parser.add_argument(
        "--verify-sass",
        action="store_true",
        help="Check the kernel SASS for explicit memory ops and HGMMA count.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index passed to the compiled benchmark binary.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=0,
        help="Grid size. 0 means auto: 2x max active blocks per SM.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200000,
        help="WGMMA loop iterations per warpgroup.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup launches.")
    parser.add_argument("--runs", type=int, default=20, help="Timed launches.")
    parser.add_argument(
        "--extra-nvcc-flag",
        action="append",
        default=[],
        help="Extra flag forwarded to nvcc. May be specified multiple times.",
    )
    return parser.parse_args()


def detect_cuda_home(explicit: str | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(pathlib.Path(env_value))
    candidates.extend(
        [
            pathlib.Path("/usr/local/cuda"),
            pathlib.Path("/usr/local/cuda-12.9"),
            pathlib.Path("/usr/local/cuda-12.8"),
            pathlib.Path("/usr/local/cuda-12.4"),
        ]
    )

    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "bin" / "nvcc").exists():
            return candidate

    nvcc = shutil.which("nvcc")
    if nvcc:
        return pathlib.Path(nvcc).resolve().parent.parent

    raise FileNotFoundError(
        "Unable to find a CUDA toolkit. Pass --cuda-home or make nvcc available."
    )


def detect_arch(explicit: str) -> str:
    if explicit != "auto":
        return explicit

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "sm_90a"

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return "sm_90a"
    first = lines[0]
    match = re.fullmatch(r"(\d+)\.(\d+)", first)
    if not match:
        return "sm_90a"
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) >= (9, 0):
        return "sm_90a"
    return f"sm_{major}{minor}"


def resolve_tool(explicit: str | None, cuda_home: pathlib.Path, name: str) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit).resolve()

    cuda_tool = cuda_home / "bin" / name
    if cuda_tool.exists():
        return cuda_tool

    system_tool = shutil.which(name)
    if system_tool:
        return pathlib.Path(system_tool).resolve()

    raise FileNotFoundError(f"Unable to locate {name}.")


def compile_benchmark(
    nvcc: pathlib.Path,
    arch: str,
    wg_n: int,
    accumulators: int,
    build_dir: pathlib.Path,
    repo_root: pathlib.Path,
    extra_nvcc_flags: list[str],
) -> tuple[pathlib.Path, pathlib.Path, str]:
    if wg_n <= 0 or wg_n > 256 or wg_n % 8 != 0:
        raise ValueError("--wg-n must be a multiple of 8 in [8, 256]")
    if accumulators <= 0:
        raise ValueError("--accumulators must be positive")

    source_path = build_dir / "pure_wgmma_bench.cu"
    binary_path = build_dir / "pure_wgmma_bench"
    source_path.write_text(CUDA_SOURCE, encoding="ascii")

    cmd = [
        str(nvcc),
        str(source_path),
        "-O3",
        "-std=c++17",
        "-lineinfo",
        f"-arch={arch}",
        f"-DWG_N={wg_n}",
        f"-DACCUMULATORS={accumulators}",
        f"-I{repo_root}",
        f"-I{repo_root / '3rdparty' / 'cutlass' / 'include'}",
        "-diag-suppress=549",
        "-diag-suppress=550",
        "-diag-suppress=20012",
        "-Xptxas=-v",
        "-o",
        str(binary_path),
        *extra_nvcc_flags,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed.\nCommand: {' '.join(cmd)}\n\n{log}")

    return source_path, binary_path, log


def summarize_ptxas(log: str) -> str:
    used_regs = re.findall(r"Used\s+(\d+)\s+registers", log)
    stack = re.findall(
        r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads",
        log,
    )
    shared = re.findall(r"Used\s+\d+\s+registers,.*?(\d+)\s+bytes smem", log)

    lines: list[str] = []
    if used_regs:
        lines.append(f"registers={used_regs[-1]}")
    if shared:
        lines.append(f"smem_bytes={shared[-1]}")
    if stack:
        frame, spill_store, spill_load = stack[-1]
        lines.append(
            f"stack_frame={frame} spill_stores={spill_store} spill_loads={spill_load}"
        )
    return " ".join(lines) if lines else "ptxas summary unavailable"


def extract_kernel_sass(cuobjdump: pathlib.Path, binary_path: pathlib.Path) -> str:
    proc = subprocess.run(
        [str(cuobjdump), "--dump-sass", str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if "Function : wgmma_bench" in line:
            start = idx
            break

    if start is None:
        raise RuntimeError("wgmma_bench not found in cuobjdump output")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if "Function :" in lines[idx]:
            end = idx
            break

    return "\n".join(lines[start:end]).strip()


def summarize_sass(kernel_sass: str) -> str:
    explicit_mem_ops = ("LDG", "STG", "LDS", "STS", "LDL", "STL")
    counts = {
        op: len(re.findall(rf"\b{op}(?:\b|\.)", kernel_sass))
        for op in explicit_mem_ops
    }
    mem_summary = " ".join(f"{op}={count}" for op, count in counts.items() if count)
    hgmma_count = len(re.findall(r"\bHGMMA(?:\.)", kernel_sass))
    arrive_count = len(re.findall(r"\bWARPGROUP\.ARRIVE\b", kernel_sass))
    wait_count = len(re.findall(r"\bWARPGROUP\.DEPBAR\.LE\b", kernel_sass))

    pieces = [f"HGMMA={hgmma_count}", f"ARRIVE={arrive_count}", f"WAIT={wait_count}"]
    if mem_summary:
        pieces.append(mem_summary)
    else:
        pieces.append("No explicit LD/ST ops found in kernel SASS.")
    return " ".join(pieces)


def run_binary(binary_path: pathlib.Path, args: argparse.Namespace) -> str:
    cmd = [
        str(binary_path),
        "--device",
        str(args.device),
        "--blocks",
        str(args.blocks),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark execution failed.\nCommand: {' '.join(cmd)}\n\n{proc.stderr}"
        )
    return proc.stdout.strip()


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    cuda_home = detect_cuda_home(args.cuda_home)
    arch = detect_arch(args.arch)
    nvcc = resolve_tool(args.nvcc, cuda_home, "nvcc")
    cuobjdump = None
    if args.dump_sass or args.verify_sass:
        cuobjdump = resolve_tool(args.cuobjdump, cuda_home, "cuobjdump")

    if args.build_dir:
        build_dir = pathlib.Path(args.build_dir).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="pure_wgmma_bench_")
        build_dir = pathlib.Path(temp_ctx.name)

    try:
        source_path, binary_path, compile_log = compile_benchmark(
            nvcc=nvcc,
            arch=arch,
            wg_n=args.wg_n,
            accumulators=args.accumulators,
            build_dir=build_dir,
            repo_root=repo_root,
            extra_nvcc_flags=args.extra_nvcc_flag,
        )

        print(f"repo_root={repo_root}")
        print(f"cuda_home={cuda_home}")
        print(f"nvcc={nvcc}")
        print(f"arch={arch}")
        print(f"build_dir={build_dir}")
        print(f"source={source_path}")
        print(f"binary={binary_path}")
        print(f"ptxas: {summarize_ptxas(compile_log)}")

        if args.verify_sass or args.dump_sass:
            assert cuobjdump is not None
            kernel_sass = extract_kernel_sass(cuobjdump, binary_path)
            if args.verify_sass:
                print(f"sass: {summarize_sass(kernel_sass)}")
            if args.dump_sass:
                print("\n===== wgmma_bench SASS =====")
                print(kernel_sass)

        print("\n===== benchmark =====")
        print(run_binary(binary_path, args))

        if args.keep_build_dir:
            print(f"\nKept build artifacts in {build_dir}")
    finally:
        if temp_ctx is not None and not args.keep_build_dir:
            temp_ctx.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
