from .gemm_base import GemmBase
from .inst import GemmInst
from tilelang.layout import make_swizzled_layout
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,
)
from tilelang.intrinsics.mma_cim_macro_generator import (
    CIMTensorCoreIntrinEmitter,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


class GemmMMA(GemmBase):
    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
        )
        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tir.Var):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GemmInst.MMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        cim_simulate = self.cim_simulate
        cim_micro_m = self.cim_micro_m
        cim_micro_n = self.cim_micro_n
        cim_micro_k = self.cim_micro_k

        # Determine if CIM micro shape differs from hardware MMA shape
        from tvm import DataType
        mma_m, mma_n = 16, 8  # hardware MMA m/n (fixed for all dtypes)
        mma_k = 256 // DataType(self.in_dtype).bits  # hardware mma_k (dtype-dependent)
        eff_micro_m = cim_micro_m if cim_micro_m > 0 else mma_m
        eff_micro_n = cim_micro_n if cim_micro_n > 0 else mma_n
        eff_micro_k = cim_micro_k if cim_micro_k > 0 else mma_k
        # Only use CIM emitter when micro shape actually differs from hardware MMA
        has_cim_micro = cim_simulate and (eff_micro_m != mma_m or eff_micro_n != mma_n or eff_micro_k != mma_k)

        if cim_simulate:
            fake_warp_rows = warp_row_tiles // eff_micro_m
            fake_warp_cols = warp_col_tiles // eff_micro_n
            mma_emitter = CIMTensorCoreIntrinEmitter(
                a_dtype=self.in_dtype,
                b_dtype=self.in_dtype,
                accum_dtype=self.accum_dtype,
                a_transposed=self.trans_A,
                b_transposed=self.trans_B,
                block_row_warps=m_warp,
                block_col_warps=n_warp,
                warp_row_tiles=warp_row_tiles,
                warp_col_tiles=warp_col_tiles,
                chunk=self.chunk,
                thread_var=thread_var,
                fake_instr_m=16,  # always use standard MMA m
                fake_instr_n=8,   # always use standard MMA n
                fake_instr_k=mma_k,
                fake_warp_rows=fake_warp_rows,
                fake_warp_cols=fake_warp_cols,
                cim_micro_m=eff_micro_m,
                cim_micro_n=eff_micro_n,
                cim_micro_k=eff_micro_k,
            )
            # Override micro_size_k for K loop bound
            micro_size_k_for_loop = eff_micro_k if has_cim_micro else None
        else:
            mma_emitter = TensorCoreIntrinEmitter(
                a_dtype=self.in_dtype,
                b_dtype=self.in_dtype,
                accum_dtype=self.accum_dtype,
                a_transposed=self.trans_A,
                b_transposed=self.trans_B,
                block_row_warps=m_warp,
                block_col_warps=n_warp,
                warp_row_tiles=warp_row_tiles,
                warp_col_tiles=warp_col_tiles,
                chunk=self.chunk,
                thread_var=thread_var,
            )
            micro_size_k_for_loop = None  # use emitter's default

        in_dtype = self.in_dtype
        warp_rows = mma_emitter.warp_rows
        warp_cols = mma_emitter.warp_cols
        local_size_a = mma_emitter.local_size_a
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = micro_size_k_for_loop if micro_size_k_for_loop else mma_emitter.micro_size_k
        # For CIM with custom micro_k: A_local sized by warp_m × micro_k / 32
        # and K sub-passes needed to fill it
        if has_cim_micro:
            WARP_SIZE = 32
            a_local_size = warp_row_tiles * eff_micro_k // WARP_SIZE
            k_sub_steps = eff_micro_k // mma_k  # number of mma_k slices per micro_k
            hw_load_size = warp_rows * local_size_a  # elements per ldmatrix pass
        else:
            a_local_size = warp_rows * local_size_a
            k_sub_steps = 1
            hw_load_size = warp_rows * local_size_a
        # We use region for memory input to support strided gemm
        # T.gemm(A_shared[0:128, :], B_shared, C_local)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        cim_simulate = self.cim_simulate
        # Pass cim_simulate to emitter's mma() whenever CIM is active
        mma_kwargs = {"cim_simulate": True} if cim_simulate else {}

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((a_local_size), in_dtype)
                if not cim_simulate:
                    B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load A into fragment — K sub-passes when micro_k > mma_k
                    for k_sub in T.serial(0, k_sub_steps):
                        if cim_simulate:
                            mma_emitter.ldmatrix_a(
                                A_local, A_region,
                                ki * k_sub_steps + k_sub,
                                a_local_offset=k_sub * hw_load_size,
                            )
                        else:
                            TensorCoreIntrinEmitter.ldmatrix_a(
                                mma_emitter, A_local, A_region,
                                ki * k_sub_steps + k_sub,
                            )

                    if cim_simulate:
                        # CIM: pass B_shared directly to mma (B lives in CIM SRAM)
                        mma_emitter.mma(A_local, B_buf, C_buf, ki, **mma_kwargs)
                    else:
                        mma_emitter.ldmatrix_b(B_local, B_region, ki)
                        mma_emitter.mma(A_local, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)

                for ki in T.serial(0, (block_K // micro_size_k)):
                    if clear_accum:
                        T.clear(C_buf)
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            # alloc_buffers body
            # insert into parent block
            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                if not cim_simulate:
                    B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    if cim_simulate:
                        # CIM: pass B_shared directly (B lives in CIM SRAM)
                        mma_emitter.mma(A_buf, B_buf, C_buf, ki, **mma_kwargs)
                    else:
                        mma_emitter.ldmatrix_b(B_local, B_region, ki)
                        mma_emitter.mma(A_buf, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rrr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """

                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_buf, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rrr, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
