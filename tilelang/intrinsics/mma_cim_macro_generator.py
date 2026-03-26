# Copyright (c) Tile-AI Corporation. All Rights Reserved.
"""
CIM (Compute-In-Memory) variant of TensorCoreIntrinEmitter.

Subclasses the upstream ``TensorCoreIntrinEmitter`` and overrides only
the behaviour that differs for CIM simulation:

* Custom ``fake_instr_*`` / ``fake_warp_*`` constructor parameters
* Hardcoded ``warp_cols = 1`` in ``_initialize_micro_size``
* Simplified ``ldmatrix_a`` / ``ldmatrix_b`` (no BufferRegion legalization,
  CIM-specific loop bounds)
* ``mma`` gains ``cim_simulate`` and ``offset`` arguments forwarded to
  ``T.ptx_mma``
* ``stmatrix`` respects ``fake_warp_rows`` / ``fake_warp_cols``
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
    - ``fake_warp_rows/cols`` to override warp tiling in ``mma``/``stmatrix``
    - ``cim_simulate`` flag passed through to ``T.ptx_mma``
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
    ):
        # Pre-set fake instruction dims *before* the base __init__ calls
        # _initialize_k_dim / _initialize_micro_size, so they can be picked up.
        self.M_DIM = 16 if fake_instr_m is None else fake_instr_m
        self.n_dim = 16 if fake_instr_n is None else fake_instr_n
        self._has_user_k_dim = fake_instr_k is not None
        self.k_dim = fake_instr_k  # may be None; _initialize_k_dim will fill in

        self.fake_warp_rows = fake_warp_rows
        self.fake_warp_cols = fake_warp_cols

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

        # Re-validate with effective warp dims (fake overrides).
        _eff_warp_rows = fake_warp_rows if fake_warp_rows is not None else self.warp_rows
        _eff_warp_cols = fake_warp_cols if fake_warp_cols is not None else self.warp_cols
        if _eff_warp_rows == 0 or _eff_warp_cols == 0:
            raise ValueError(
                f"Invalid threads configuration for this tile shape, "
                f"{self.warp_rows} x {self.warp_cols} with threads {self.threads}"
            )

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
        """CIM always uses ``warp_cols = 1`` and ``micro_size_y = n_dim``."""
        warp_rows = self.warp_row_tiles // m_dim
        if warp_rows == 0 and getattr(self, "fake_warp_rows", None) is not None:
            warp_rows = 1
        self.warp_rows = warp_rows
        self.warp_cols = 1
        self.micro_size_x = m_dim
        self.micro_size_y = self.n_dim
        self.micro_size_k = k_dim

    # ------------------------------------------------------------------
    # ldmatrix overrides (simplified buffer access, no BufferRegion)
    # ------------------------------------------------------------------

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer,
                   ki: PrimExpr, rk: PrimExpr | None = 0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = T.ceildiv(
            T.ceildiv(warp_row_tiles * self.micro_size_k, self.WARP_SIZE), 8
        )
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
                        i * local_size_a,
                        T.address_of(A_shared_buf_elem),
                        get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed),
                    )
                else:
                    for j in T.serial(local_size_a):
                        mi, mk = mma_load_layout(tx, j)
                        A_local_buf[i * local_size_a + j] = A_shared_buf[wk + mk, wi + mi]

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
    # mma override (cim_simulate + offset + reversed loop + fake warp dims)
    # ------------------------------------------------------------------

    def mma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer,
            k_inner: PrimExpr | None = 0, cim_simulate: bool = False,
            offset: PrimExpr | None = 0):
        warp_rows = self.warp_rows if self.fake_warp_rows is None else self.fake_warp_rows
        warp_cols = self.warp_cols if self.fake_warp_cols is None else self.fake_warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * local_size_b if b_is_fragment else 0

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for j, i in T.grid(warp_cols, warp_rows):
                T.ptx_mma(
                    accum_dtype, mma_prefix, "row", "col",
                    a_dtype_abbrv, b_dtype_abbrv, accum_dtype_abbrv,
                    A_local_buf.data,
                    a_local_stride + i * local_size_a,
                    B_local_buf.access_ptr(1, offset=offset),
                    b_local_stride + j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                    None,
                    cim_simulate,
                )
                if replicate_b:
                    T.ptx_mma(
                        accum_dtype, mma_prefix, "row", "col",
                        a_dtype_abbrv, b_dtype_abbrv, accum_dtype_abbrv,
                        A_local_buf.data,
                        a_local_stride + i * local_size_a,
                        B_local_buf.data,
                        b_local_stride + j * local_size_b + lift(local_size_b) // 2,
                        C_local_buf.data,
                        i * warp_cols * local_size_out + j * local_size_out
                        + lift(local_size_out) // 2,
                        T.bool(False),
                        None,
                        cim_simulate,
                    )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    # ------------------------------------------------------------------
    # stmatrix override (fake warp dims)
    # ------------------------------------------------------------------

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows if self.fake_warp_rows is None else self.fake_warp_rows
        warp_cols = self.warp_cols if self.fake_warp_cols is None else self.fake_warp_cols
        local_size_out = self.local_size_out

        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, n_dim = self.M_DIM, self.n_dim
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        thread_binding = self.get_thread_binding()

        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        if C_buf_dims == 2:
                            C_buf[
                                (warp_m * warp_rows + i) * M_DIM + row,
                                (warp_n * warp_cols + j) * n_dim + col,
                            ] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]
                        else:
                            C_buf[
                                warp_m * warp_rows + i, warp_n * warp_cols + j, row, col,
                            ] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        C_buf[
                            (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                            (pid_n * BLOCK_N + warp_n * warp_cols + j) * n_dim + col,
                        ] = C_local_buf[
                            i * warp_cols * local_size_out + j * local_size_out + local_id
                        ]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )


# Backward-compatible alias: existing code imports TensorCoreIntrinEmitter
# from this module.
TensorCoreIntrinEmitter = CIMTensorCoreIntrinEmitter
