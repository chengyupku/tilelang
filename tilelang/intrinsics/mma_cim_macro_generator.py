# Copyright (c) Tile-AI Corporation. All Rights Reserved.
"""
CIM (Compute-In-Memory) variant of TensorCoreIntrinEmitter.

Subclasses the upstream ``TensorCoreIntrinEmitter`` and overrides only
the behaviour that differs for CIM simulation:

* Custom ``fake_instr_m/n/k`` to override MMA instruction dimensions
* ``_initialize_micro_size`` derives warp_rows/warp_cols from hardware MMA
  tile shape (M_DIM × n_dim), ensuring ldmatrix/mma/stmatrix use consistent
  tiling.  CIM micro_m/n/k only affects the outer ki loop (A load frequency).
* Simplified ``ldmatrix_a`` / ``ldmatrix_b`` (no BufferRegion legalization,
  CIM-specific loop bounds)
* ``mma`` gains ``cim_simulate`` and ``offset`` arguments forwarded to
  ``T.ptx_mma``
"""

from __future__ import annotations

import tilelang.language as T
from tvm import DataType
from tvm.tir import PrimExpr, Buffer, Var
from tvm.runtime import convert

from .mma_macro_generator import TensorCoreIntrinEmitter as _BaseTensorCoreIntrinEmitter

# ---- Upstream compatibility guard ----
# CIM overrides depend on base-class internals (method signatures and
# instance attributes set during __init__).  When syncing upstream, a
# change here is the first signal that CIM overrides need updating.
import inspect as _inspect, warnings as _warnings

_EXPECTED_BASE_INIT_PARAMS = {
    "a_dtype", "b_dtype", "accum_dtype", "a_transposed", "b_transposed",
    "block_row_warps", "block_col_warps", "warp_row_tiles", "warp_col_tiles",
    "chunk", "reduce_k", "num_elems_per_byte", "is_m_first", "thread_var",
}
_EXPECTED_BASE_METHODS = {
    "get_thread_binding", "extract_thread_binding",
    "_initialize_k_dim", "_initialize_micro_size", "_initialize_local_size",
}
_actual_params = set(_inspect.signature(_BaseTensorCoreIntrinEmitter.__init__).parameters) - {"self"}
if not _EXPECTED_BASE_INIT_PARAMS.issubset(_actual_params):
    _missing = _EXPECTED_BASE_INIT_PARAMS - _actual_params
    _warnings.warn(
        f"CIM: upstream TensorCoreIntrinEmitter.__init__ no longer has params {_missing}. "
        f"mma_cim_macro_generator.py likely needs updating.",
        stacklevel=2,
    )
for _m in _EXPECTED_BASE_METHODS:
    if not hasattr(_BaseTensorCoreIntrinEmitter, _m):
        _warnings.warn(
            f"CIM: upstream TensorCoreIntrinEmitter no longer has method '{_m}'. "
            f"mma_cim_macro_generator.py likely needs updating.",
            stacklevel=2,
        )
del _inspect, _warnings, _actual_params, _m

from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)
from tilelang.utils import is_fragment
from tilelang.intrinsics.mma_layout import (
    mma_load_a_32x16_to_shared_16x32_layout,
    mma_load_a_32x4_to_shared_16x8_layout,
    mma_load_b_32x16_to_shared_16x32_layout,
    mma_load_b_32x4_to_shared_16x8_layout,
)

lift = convert


class CIMTensorCoreIntrinEmitter(_BaseTensorCoreIntrinEmitter):
    """
    CIM-specialised TensorCore intrinsic emitter.

    Extends the upstream ``TensorCoreIntrinEmitter`` with:
    - ``fake_instr_m/n/k`` to override MMA instruction dimensions
    - MMA-based warp tiling (warp_rows/warp_cols derived from M_DIM/n_dim)
    - ``cim_simulate`` flag passed through to ``T.ptx_mma``

    The CIM micro shape (micro_m/n/k) is NOT handled by the emitter — it
    only controls the outer ki loop in the caller.  Internally, the emitter
    always tiles in hardware MMA units so ldmatrix/mma/stmatrix indexing is
    consistent.
    """

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
        fake_instr_m: int | None = None,
        fake_instr_n: int | None = None,
        fake_instr_k: int | None = None,
        fake_warp_rows: int | None = None,
        fake_warp_cols: int | None = None,
        cim_micro_m: int | None = None,
        cim_micro_n: int | None = None,
        cim_micro_k: int | None = None,
        cim_stride_index: bool = False,
    ):
        # Pre-set fake instruction dims *before* the base __init__ calls
        # _initialize_k_dim / _initialize_micro_size, so they can be picked up.
        self.M_DIM = 16 if fake_instr_m is None else fake_instr_m
        self.n_dim = 16 if fake_instr_n is None else fake_instr_n
        self._has_user_k_dim = fake_instr_k is not None
        self.k_dim = fake_instr_k  # may be None; _initialize_k_dim will fill in

        self.fake_warp_rows = fake_warp_rows
        self.fake_warp_cols = fake_warp_cols

        # CIM micro dims for local buffer stride calculation
        self._cim_micro_m = cim_micro_m
        self._cim_micro_n = cim_micro_n
        self._cim_micro_k = cim_micro_k
        # Index mode: True = CIM stride (micro-based, models CIM arch),
        #             False = hw cycling (GPU-native, better perf on real GPU)
        self.cim_stride_index = cim_stride_index

        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            is_m_first=is_m_first,
            thread_var=thread_var,
        )

        # Validate effective warp dims (fake overrides for mma/stmatrix).
        _eff_warp_rows = fake_warp_rows if fake_warp_rows is not None else self.warp_rows
        _eff_warp_cols = fake_warp_cols if fake_warp_cols is not None else self.warp_cols
        if _eff_warp_rows == 0 or _eff_warp_cols == 0:
            raise ValueError(
                f"Invalid CIM warp configuration: "
                f"warp_rows={self.warp_rows}, warp_cols={self.warp_cols}, "
                f"fake_warp_rows={fake_warp_rows}, fake_warp_cols={fake_warp_cols}"
            )

        # CIM-based local strides for mma/stmatrix indexing.
        # When micro dims are given, stride = micro_m * micro_k/n / warp_size.
        # This ensures mma loop accesses fit within A_local/C_local sized by
        # warp_m*micro_k/32 and warp_m*warp_n/32 respectively.
        if cim_micro_m is not None and cim_micro_k is not None:
            self.cim_local_size_a = (cim_micro_m * cim_micro_k) // self.WARP_SIZE
        else:
            self.cim_local_size_a = self.local_size_a  # fallback to MMA-based
        if cim_micro_m is not None and cim_micro_n is not None:
            self.cim_local_size_out = (cim_micro_m * cim_micro_n) // self.WARP_SIZE
        else:
            self.cim_local_size_out = self.local_size_out

    # ------------------------------------------------------------------
    # Overridden initialisation helpers
    # ------------------------------------------------------------------

    def _initialize_k_dim(self, a_dtype="float16"):
        if self.k_dim is not None or self._has_user_k_dim:
            return
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = 256 // a_dtype.bits

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        """Compute warp tiling from hardware MMA dimensions.

        Unlike the old approach that hardcoded ``warp_cols = 1`` and relied on
        ``fake_warp_rows/cols`` to patch mma/stmatrix, we now derive warp_rows
        and warp_cols directly from the hardware MMA tile shape (M_DIM × n_dim).
        This ensures ldmatrix, mma, and stmatrix all use the same tiling.
        """
        self.warp_rows = self.warp_row_tiles // m_dim
        self.warp_cols = self.warp_col_tiles // self.n_dim
        self.micro_size_x = m_dim
        self.micro_size_y = self.n_dim
        self.micro_size_k = k_dim

    # ------------------------------------------------------------------
    # ldmatrix overrides (simplified buffer access, no BufferRegion)
    # ------------------------------------------------------------------

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf,
                   ki: PrimExpr, rk: PrimExpr | None = 0,
                   a_local_offset: int = 0):
        """Load A from shared into local fragment.

        Args:
            A_shared_buf: Buffer or BufferRegion for shared A data.
            a_local_offset: element offset into A_local_buf for this load.
                Used when loading multiple mma_k sub-slices to fill A_local.
        """
        # Handle BufferRegion (from T.gemm) — extract the underlying Buffer
        from tvm.tir import BufferRegion
        if isinstance(A_shared_buf, BufferRegion):
            A_shared_buf = A_shared_buf.buffer
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed

        ldmatrix_available = not (DataType(a_dtype).bits != 16 and a_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(a_dtype).bits == 8:
                mma_load_layout = mma_load_a_32x16_to_shared_16x32_layout
            elif DataType(a_dtype).bits == 32:
                mma_load_layout = mma_load_a_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {a_dtype}")

        thread_binding = self.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk=0):
            stride = A_shared_buf.shape[-1]
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            trans = self.a_transposed

            for i in T.serial(warp_rows):
                wi = warp_m * warp_row_tiles + i * micro_size_x
                wk = rk * chunk + ki * micro_size_k
                A_shared_buf_elem = (
                    A_shared_buf[wk, wi] if a_transposed else A_shared_buf[wi, wk]
                )

                if ldmatrix_available:
                    T.ptx_ldmatrix(
                        a_dtype,
                        T.bool(trans),
                        4,
                        ".b16",
                        A_local_buf.data,
                        a_local_offset + i * local_size_a,
                        T.address_of(A_shared_buf_elem),
                        get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed),
                    )
                else:
                    for j in T.serial(local_size_a):
                        mi, mk = mma_load_layout(tx, j)
                        A_local_buf[a_local_offset + i * local_size_a + j] = A_shared_buf[wk + mk, wi + mi]

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer,
                   ki: PrimExpr, rk: PrimExpr | None = 0):
        warp_col_tiles = self.warp_col_tiles
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        b_transposed = self.b_transposed
        replicate_b = self.n_dim == 16

        ldmatrix_available = not (DataType(b_dtype).bits != 16 and not b_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(b_dtype).bits == 8:
                mma_load_layout = mma_load_b_32x16_to_shared_16x32_layout
            elif DataType(b_dtype).bits == 32:
                mma_load_layout = mma_load_b_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {b_dtype}")

        thread_binding = self.get_thread_binding()

        @T.macro
        def _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk=0):
            stride = B_shared_buf.shape[-1]
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            trans = not b_transposed

            # CIM loads only 1 tile (address hook, not full data)
            for i in T.serial(1):
                wi = warp_n * warp_col_tiles + i * micro_size_y
                wk = rk * chunk + ki * micro_size_k

                if ldmatrix_available:
                    B_shared_buf_elem = (
                        B_shared_buf[wi, wk] if b_transposed else B_shared_buf[wk, wi]
                    )
                    T.ptx_ldmatrix(
                        b_dtype,
                        T.bool(trans),
                        4 if replicate_b else 2,
                        ".b16",
                        B_local_buf.data,
                        i * local_size_b,
                        T.address_of(B_shared_buf_elem),
                        get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed),
                    )
                else:
                    for j in T.serial(local_size_b):
                        mi, mk = mma_load_layout(tx, j)
                        B_local_buf[i * local_size_b + j] = B_shared_buf[wk + mk, wi + mi]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    # ------------------------------------------------------------------
    # mma override (cim_simulate + offset + CIM micro-based warp dims)
    # ------------------------------------------------------------------

    def mma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer,
            k_inner: PrimExpr | None = 0, cim_simulate: bool = False,
            offset: PrimExpr | None = 0):
        # CIM loop: fake_warp_rows × fake_warp_cols iterations,
        # each MMA call = one CIM instruction.
        warp_rows = self.warp_rows if self.fake_warp_rows is None else self.fake_warp_rows
        warp_cols = self.warp_cols if self.fake_warp_cols is None else self.fake_warp_cols
        hw_warp_rows = self.warp_rows
        hw_warp_cols = self.warp_cols
        # A/C stride: CIM-based or hw-cycling depending on cim_stride_index
        use_cim_stride = self.cim_stride_index
        local_size_a_hw = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out_hw = self.local_size_out
        local_size_a_cim = self.cim_local_size_a
        local_size_out_cim = self.cim_local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        if use_cim_stride:
            a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a_cim if a_is_fragment else 0
        else:
            a_local_stride: PrimExpr = k_inner * hw_warp_rows * local_size_a_hw if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * hw_warp_cols * local_size_b if b_is_fragment else 0

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for j, i in T.grid(warp_cols, warp_rows):
                if use_cim_stride:
                    a_off = a_local_stride + i * local_size_a_cim
                    c_off = i * warp_cols * local_size_out_cim + j * local_size_out_cim
                else:
                    i_hw = i % hw_warp_rows
                    j_hw = (i // hw_warp_rows + j) % hw_warp_cols
                    a_off = a_local_stride + i_hw * local_size_a_hw
                    c_off = i_hw * hw_warp_cols * local_size_out_hw + j_hw * local_size_out_hw
                b_off = b_local_stride + (j if use_cim_stride else j_hw) * local_size_b
                T.ptx_mma(
                    accum_dtype, mma_prefix, "row", "col",
                    a_dtype_abbrv, b_dtype_abbrv, accum_dtype_abbrv,
                    A_local_buf.data, a_off,
                    B_local_buf.access_ptr(1, offset=offset),
                    b_off,
                    C_local_buf.data, c_off,
                    T.bool(False), None, cim_simulate,
                )
                if replicate_b:
                    T.ptx_mma(
                        accum_dtype, mma_prefix, "row", "col",
                        a_dtype_abbrv, b_dtype_abbrv, accum_dtype_abbrv,
                        A_local_buf.data, a_off,
                        B_local_buf.data,
                        b_off + lift(local_size_b) // 2,
                        C_local_buf.data, c_off + lift(local_size_out_hw) // 2,
                        T.bool(False), None, cim_simulate,
                    )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    # ------------------------------------------------------------------
    # stmatrix override (CIM micro-based warp dims)
    # ------------------------------------------------------------------

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows if self.fake_warp_rows is None else self.fake_warp_rows
        warp_cols = self.warp_cols if self.fake_warp_cols is None else self.fake_warp_cols
        hw_warp_rows = self.warp_rows
        hw_warp_cols = self.warp_cols
        use_cim_stride = self.cim_stride_index
        local_size_out_hw = self.local_size_out
        local_size_out_cim = self.cim_local_size_out

        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, n_dim = self.M_DIM, self.n_dim
        local_size_out_inner = local_size_out_cim if use_cim_stride else local_size_out_hw
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        thread_binding = self.get_thread_binding()

        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                if use_cim_stride:
                    c_off_base = i * warp_cols * local_size_out_cim + j * local_size_out_cim
                else:
                    i_hw = i % hw_warp_rows
                    j_hw = (i // hw_warp_rows + j) % hw_warp_cols
                    c_off_base = i_hw * hw_warp_cols * local_size_out_hw + j_hw * local_size_out_hw
                for local_id_o in T.serial(local_size_out_inner // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        if C_buf_dims == 2:
                            C_buf[
                                (warp_m * warp_rows + i) * M_DIM + row,
                                (warp_n * warp_cols + j) * n_dim + col,
                            ] = C_local_buf[c_off_base + local_id]
                        else:
                            C_buf[
                                warp_m * warp_rows + i, warp_n * warp_cols + j, row, col,
                            ] = C_local_buf[c_off_base + local_id]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                if use_cim_stride:
                    c_off_base = i * warp_cols * local_size_out_cim + j * local_size_out_cim
                else:
                    i_hw = i % hw_warp_rows
                    j_hw = (i // hw_warp_rows + j) % hw_warp_cols
                    c_off_base = i_hw * hw_warp_cols * local_size_out_hw + j_hw * local_size_out_hw
                for local_id_o in T.serial(local_size_out_inner // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        C_buf[
                            (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                            (pid_n * BLOCK_N + warp_n * warp_cols + j) * n_dim + col,
                        ] = C_local_buf[c_off_base + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )


# Backward-compatible alias: existing code imports TensorCoreIntrinEmitter
# from this module.
TensorCoreIntrinEmitter = CIMTensorCoreIntrinEmitter
