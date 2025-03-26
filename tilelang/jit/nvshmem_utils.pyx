# distutils: language = c++
# cython: language_level=3
# distutils: define_macros=MPICH_SKIP_MPICXX=1,OMPI_SKIP_MPICXX=1

import cython
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
import numpy as np
from functools import reduce
import operator

cdef extern from "<torch/csrc/distributed/c10d/Types.hpp>":
    pass

cdef extern from "<c10/core/ScalarType.h>" namespace "c10":
    cpdef enum class ScalarType "c10::ScalarType":
        Undefined
        Byte
        Char
        Short
        Int
        Long
        Half
        Float
        Double
        ComplexHalf
        ComplexFloat
        ComplexDouble
        Bool
        QInt8
        QUInt8
        QInt32
        BFloat16
        QUInt4x2
        QUInt2x4
        Bits1x8
        Bits2x4
        Bits4x2
        Bits8
        Bits16

cdef extern from "<c10/cuda/CUDAFunctions.h>" namespace "c10::cuda":
    int current_device() nogil
    void device_synchronize() nogil

cdef extern from "<ATen/ops/from_blob.h>" namespace "at":
    cdef cppclass kCUDA "at::kCUDA":
        pass
    
cdef extern from "<torch/cuda.h>":
    pass

cdef extern from "<c10/cuda/CUDAGuard.h>" namespace "at::cuda":
    cdef cppclass CUDAGuard:
        CUDAGuard(int) nogil

cdef extern from "<cuda_runtime.h>":
    cdef enum cudaError:
        cudaSuccess
    ctypedef cudaError cudaError_t
    const char* cudaGetErrorString(cudaError_t)
    cudaError_t cudaMemset(void* devPtr, int value, size_t count) nogil
    cudaError_t cudaFree(void* devPtr) nogil

cdef inline void CUDA_CHECK(cudaError_t status) nogil:
    if status != cudaSuccess:
        with gil:
            raise RuntimeError(f"CUDA error: {cudaGetErrorString(status)}")

cdef extern from "<nvshmemx.h>":
    void* nvshmem_malloc(size_t size) nogil
    void nvshmem_free(void* ptr) nogil
    int nvshmem_my_pe() nogil
    int nvshmem_team_my_pe(int team) nogil
    int nvshmem_team_n_pes(int team) nogil
    void* nvshmem_ptr(void* ptr, int pe) nogil
    cdef int NVSHMEMX_TEAM_NODE

cdef extern from "torch/torch.h" namespace "torch":
    cdef cppclass Tensor:
        pass
    int elementSize(ScalarType dtype) nogil
    
cdef extern from "torch/torch.h" namespace "at":
    
    cdef cppclass TensorOptions:
        TensorOptions() nogil
        TensorOptions dtype(ScalarType dtype) nogil
        TensorOptions device(kCUDA) nogil
        TensorOptions device_index(int index) nogil

    Tensor from_blob(void* data, vector[int64_t] sizes, void (*deleter)(void*), TensorOptions options) nogil
    Tensor from_blob(void* data, vector[int64_t] sizes, TensorOptions options) nogil

cdef void tensor_deleter(void* void_ptr) noexcept nogil:
    device_synchronize()
    nvshmem_free(void_ptr)
    
# Add this new extern declaration
cdef extern from "torch/extension.h":
    object THPVariable_Wrap(Tensor tensor)
    

cdef extern from "<nvshmemx.h>":
    int nvshmemx_init_status() nogil
    int NVSHMEM_STATUS_IS_INITIALIZED
    ctypedef struct nvshmemx_uniqueid_t:
        pass
    void nvshmemx_get_uniqueid(nvshmemx_uniqueid_t *id) nogil
    ctypedef struct nvshmemx_init_attr_t:
        pass
    void nvshmemx_set_attr_uniqueid_args(int rank, int nranks, nvshmemx_uniqueid_t *id, nvshmemx_init_attr_t *attr) nogil
    void nvshmemx_init_attr(int type, nvshmemx_init_attr_t *attr) nogil
    int NVSHMEMX_INIT_WITH_UNIQUEID

cdef void check_nvshmem_init() nogil:
    cdef int status = nvshmemx_init_status()
    if status < NVSHMEM_STATUS_IS_INITIALIZED:
        with gil:
            raise RuntimeError(f"nvshmem not initialized: status {status}")

cdef extern from "<mpi.h>":
    ctypedef struct MPI_Comm:
        pass
    ctypedef struct MPI_Datatype:
        pass
    ctypedef struct MPI_Status:
        pass
    
    int MPI_Init(int* argc, char*** argv) nogil
    int MPI_Comm_rank(MPI_Comm comm, int* rank) nogil
    int MPI_Comm_size(MPI_Comm comm, int* size) nogil
    int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) nogil
    MPI_Comm MPI_COMM_WORLD
    MPI_Datatype MPI_UINT8_T



@cython.embedsignature(True)
def init_nvshmem():
    cdef int argc = 0
    cdef char** argv = NULL
    cdef int rank, nranks
    
    MPI_Init(&argc, &argv)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank)
    MPI_Comm_size(MPI_COMM_WORLD, &nranks)

    cdef nvshmemx_uniqueid_t id
    if rank == 0:
        nvshmemx_get_uniqueid(&id)
    
    MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);

    cdef nvshmemx_init_attr_t init_attr
    nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &init_attr)
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &init_attr)
    
    cdef int mype = nvshmem_my_pe()
    if rank != mype:
        raise RuntimeError("NVShmem init: rank does not match PE!")

@cython.embedsignature(True)
def nvshmem_create_tensor(vector[int64_t] shape, ScalarType dtype):
    """
    Create a PyTorch tensor managed by NVSHMEM
    
    Args:
        shape: tensor shape
        dtype: tensor data type
    
    Returns:
        PyTorch tensor
    """
    cdef Tensor result
    
    check_nvshmem_init()
    cdef int current_dev = current_device()
    cdef TensorOptions option_gpu = TensorOptions().dtype(dtype).device(kCUDA).device_index(current_dev)
    cdef int64_t element_size = elementSize(dtype)
    cdef int64_t size = element_size * reduce(operator.mul, shape, 1)
    
    assert size != 0
    
    device_synchronize()
    cdef void* ptr = nvshmem_malloc(size)
    assert ptr != NULL
    
    CUDA_CHECK(cudaMemset(ptr, 0, size))
    
    result = from_blob(
        ptr,
        shape,
        tensor_deleter,
        option_gpu
    )

    return THPVariable_Wrap(result)
    

# @cython.embedsignature(True)
# def nvshmem_create_tensor_list(vector[int64_t] shape, ScalarType dtype):
#     """
#     Create a list of PyTorch tensors managed by NVSHMEM
    
#     Args:
#         shape: tensor shape
#         dtype: tensor data type
        
#     Returns:
#         List of PyTorch tensors
#     """
#     check_nvshmem_init()
#     cdef int current_dev = current_device()
#     cdef TensorOptions option_gpu = TensorOptions().dtype(dtype).device(kCUDA).device_index(current_dev)
#     cdef size_t size = elementSize(dtype) * reduce(operator.mul, shape, 1)
    
#     assert size != 0
    
#     cdef int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE)
#     cdef int rank = nvshmem_my_pe()
#     cdef int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)
    
#     cdef list tensors = []
    
#     device_synchronize()
#     cdef void* ptr = nvshmem_malloc(size)
#     assert ptr != NULL
    
#     CUDA_CHECK(cudaMemset(ptr, 0, size))
    
#     cdef int rank_offset = rank - local_rank
#     cdef int rank_global
#     cdef void* rptr
    
#     for i in range(local_world_size):
#         rank_global = i + rank_offset
#         if rank == rank_global:
#             tensors.append(from_blob(
#                 ptr,
#                 shape,
#                 lambda void_ptr: (
#                     device_synchronize(),
#                     nvshmem_free(void_ptr),
#                     device_synchronize()
#                 ),
#                 option_gpu
#             ))
#         else:
#             rptr = nvshmem_ptr(ptr, rank_global)
#             assert rptr != NULL, f"rank {rank}"
#             tensors.append(from_blob(rptr, shape, option_gpu))
    
#     return tensors 