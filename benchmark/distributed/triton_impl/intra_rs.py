''''''''''Adapted from Triton-distributed/tutorials/05-intra-node-reduce-scatter.py'''''''''

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
Intra-node ReduceScatter
========================
In this tutorial, you will write a intra-node reduce-scatter operation.

In doing so, you will learn about:

* use copy engine to communicate data within a node
* How to write an intra-node reduce-scatter kernel.

.. code-block:: bash

    # To run this tutorial
    bash ./launch.sh ./tutorials/05-intra-node-reduce-scatter.py

"""

import torch
import triton
import triton.language as tl
from triton_dist.kernels.nvidia.common_ops import barrier_all_intra_node_atomic_cas_block
from triton_dist.utils import p2p_native_atomic_required
from typing import List, Optional
from triton_dist import pynvshmem
import triton_dist

import os

SIGNAL_DTYPE = torch.uint64

################### triton kernel ####################


@triton.jit
def kernel_ring_reduce(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_split, N]
    # shape of matrix
    M_per_rank,
    N,
    begin_idx,
    num_splits: tl.constexpr,
    # reduce tile shape
    BLOCK_SIZE_M: tl.constexpr = 256,
    BLOCK_SIZE_N: tl.constexpr = 64,
):
    """
    The difference between this kernel and a normal reduction operation lies in the accumulation order.
    In a normal reduction, the accumulation sta`rts from the 0th row.
    This kernel starts the accumulation from the `offset`.
    """
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M_per_rank * num_splits, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    output_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    for tile_id in range(pid, total_tiles, num_pid):
        tile_id_m = tile_id // num_tiles_n
        tile_id_n = tile_id % num_tiles_n
        # accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=out_ptr.dtype.element_ty)
        cur_rank = (begin_idx + 1) % num_splits
        accum = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
        for i in range(1, num_splits):
            cur_rank = (i + begin_idx + 1) % num_splits
            data = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
            accum += data

        output_desc.store([tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N], accum)


def ring_reduce(
    input,  # [M_per_node, N]
    output,  # [M_per_rank, N]
    begin_idx,
    num_splits,
    stream,
    num_sms=-1,
):
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    total_M, N = input.shape
    M_per_split = total_M // num_splits
    assert output.shape[0] == M_per_split and total_M % num_splits == 0
    if num_sms == -1:
        grid = lambda META: (triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
        with torch.cuda.stream(stream):
            kernel_ring_reduce[grid](
                input,
                output,
                M_per_split,
                N,
                begin_idx,
                num_splits,
                BLOCK_SIZE_M=256,
                BLOCK_SIZE_N=64,
                num_warps=4,
            )
    else:
        grid = lambda META: (min(
            triton.cdiv(M_per_split, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), num_sms), )
        with torch.cuda.stream(stream):
            kernel_ring_reduce[grid](
                input,
                output,
                M_per_split,
                N,
                begin_idx,
                num_splits,
                BLOCK_SIZE_M=256,
                BLOCK_SIZE_N=128,
                num_warps=8,
            )

    return output


def intra_node_scatter(input_intra_node, scatter_bufs_intra_node: List[torch.Tensor], local_rank, stream):
    M, N = input_intra_node.shape
    local_world_size = len(scatter_bufs_intra_node)
    M_per_rank = M // local_world_size

    # send input_intra_node[remote_rank * M_per_rank : (remote_rank + 1) * M_per_rank] on the current rank to
    # input_intra_node[rank * M_per_rank : (rank + 1)] on the remote rank.
    with torch.cuda.stream(stream):
        for i in range(0, local_world_size):
            """
            Rank level swizzle: Each rank perform scatter start from the next rank of the current.
            In this way, the send/recv communication volume of each rank is balanced.

            For time start from 0 to local_world_size, the communication order between ranks:
                time 0: 0->1, 1->2, 2->3, 3->0
                time 1: 0->2, 1->3, 2->0, 3->1
                time 2: 0->3, 1->0, 2->1, 3->2
                time 3: 0->0, 1->1, 2->2, 3->3
            """
            remote_local_rank = (local_rank + i + 1) % local_world_size

            remote_buf = scatter_bufs_intra_node[remote_local_rank][local_rank * M_per_rank:(local_rank + 1) *
                                                                    M_per_rank, :]
            local_buf = input_intra_node[remote_local_rank * M_per_rank:(remote_local_rank + 1) * M_per_rank, :]
            # use copy engine to perform scatter(torch will use `cudamemcpy` to copy continuous data)
            remote_buf.copy_(local_buf)


@p2p_native_atomic_required
def reducer_scatter_intra_node(input, scatter_bufs, sync_buf, local_rank, local_world_size):

    stream = torch.cuda.current_stream()
    M, N = input.shape
    M_per_rank = M // local_world_size

    output = torch.empty((M_per_rank, N), dtype=input.dtype, device=input.device)
    # step 1: intra node reduce-scatter
    intra_node_scatter(input, scatter_bufs, local_rank, stream)

    # step 2: waits for all ranks to complete the scatter.
    barrier_all_intra_node_atomic_cas_block[(1, )](local_rank, local_rank, local_world_size, sync_buf)
    # step 3: perform reduction to get the result of the intra-node reduce-scatter.
    ring_reduce(scatter_bufs[local_rank], output, local_rank, local_world_size, stream)
    return output

