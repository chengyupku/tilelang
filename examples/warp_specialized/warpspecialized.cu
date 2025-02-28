
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void __launch_bounds__(256) non_split_kv_kernel(__grid_constant__ const CUtensorMap KV_desc, __grid_constant__ const CUtensorMap K_pe_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap Q_pe_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[128];
  float logsum[2];
  float scores_max[2];
  float scores_max_prev[2];
  float acc_s[32];
  float scores_scale[2];
  float scores_sum[2];
  __shared__ half_t scores_scale_shared[64];
  half_t acc_s_cast[32];
  __shared__ uint64_t _mbarrier[11];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(Q_pe_desc);
    tl::prefetch_tma_descriptor(KV_desc);
    tl::prefetch_tma_descriptor(K_pe_desc);
    tl::prefetch_tma_descriptor(Output_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 256);
    tl::mbarrier_init(_mbarrier[5], 256);
    tl::mbarrier_init(_mbarrier[6], 128);
    tl::mbarrier_init(_mbarrier[7], 128);
    tl::mbarrier_init(_mbarrier[8], 128);
    tl::mbarrier_init(_mbarrier[9], 128);
    tl::mbarrier_init(_mbarrier[10], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[6], 65536);
    }
    if (((int)threadIdx.x) == 128) {
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[16384])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[20480])), 64, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[24576])), 128, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[28672])), 192, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[32768])), 256, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[36864])), 320, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[40960])), 384, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_load(Q_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[45056])), 448, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    }
    if (((int)threadIdx.x) == 128) {
      tl::mbarrier_expect_tx(_mbarrier[6], 8192);
    }
    if (((int)threadIdx.x) == 128) {
      tl::tma_load(Q_pe_desc, _mbarrier[6], (&(((half_t*)buf_dyn_shmem)[0])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    }
    tl::mbarrier_arrive(_mbarrier[6]);
    for (int k = 0; k < 128; ++k) {
      tl::mbarrier_wait(_mbarrier[((k & 1) + 4)], (((k & 3) >> 1) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[(k & 1)], 65536);
      }
      if (((int)threadIdx.x) == 128) {
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 49152)])), 0, 0, (k * 64), ((int)blockIdx.y));
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 53248)])), 64, 0, (k * 64), ((int)blockIdx.y));
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 57344)])), 128, 0, (k * 64), ((int)blockIdx.y));
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 61440)])), 192, 0, (k * 64), ((int)blockIdx.y));
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 65536)])), 256, 0, (k * 64), ((int)blockIdx.y));
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 69632)])), 320, 0, (k * 64), ((int)blockIdx.y));
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 73728)])), 384, 0, (k * 64), ((int)blockIdx.y));
        tl::tma_load(KV_desc, _mbarrier[(k & 1)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 77824)])), 448, 0, (k * 64), ((int)blockIdx.y));
      }
      tl::mbarrier_arrive(_mbarrier[(k & 1)]);
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[((k & 1) + 2)], 8192);
      }
      if (((int)threadIdx.x) == 128) {
        tl::tma_load(K_pe_desc, _mbarrier[((k & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 8192)])), 0, 0, (k * 64), ((int)blockIdx.y));
      }
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 2)]);
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[(k & 1)], ((k & 3) >> 1));
      tl::gemm_rs<64, 256, 64, 4, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 32768) + 49152)])), (&(acc_o[0])));
      tl::mbarrier_arrive(_mbarrier[((k & 1) + 4)]);
    }
  } else {
    const dim3 blockIdx = tl::rasterization2DRow<10>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(acc_o + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      logsum[i_1] = 0.000000e+00f;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      scores_max[i_2] = -CUDART_INF_F;
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[6], 0);
    for (int k_1 = 0; k_1 < 128; ++k_1) {
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        *(float2*)(acc_s + (i_3 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[(k_1 & 1)], ((k_1 & 3) >> 1));
      tl::gemm_ss<64, 64, 512, 4, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[16384])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 32768) + 49152)])), (&(acc_s[0])));
      tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 2)], ((k_1 & 3) >> 1));
      tl::gemm_ss<64, 64, 64, 4, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 4096) + 8192)])), (&(acc_s[0])));
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        scores_max_prev[i_4] = scores_max[i_4];
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        scores_max[i_5] = -CUDART_INF_F;
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          scores_max[i_6] = max(scores_max[i_6], acc_s[((((rv & 7) * 4) + (i_6 * 2)) + (rv >> 3))]);
        }
        scores_max[i_6] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_6]);
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        scores_scale[i_7] = exp2f(((scores_max_prev[i_7] * 6.011229e-02f) - (scores_max[i_7] * 6.011229e-02f)));
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        float2 __1;
        float2 __2;
          float2 __3;
            float2 v_ = *(float2*)(acc_s + (i_8 * 2));
            float2 v__1 = make_float2(6.011229e-02f, 6.011229e-02f);
            __3.x = (v_.x*v__1.x);
            __3.y = (v_.y*v__1.y);
          float2 v__2 = make_float2((scores_max[(i_8 & 1)] * 6.011229e-02f), (scores_max[(i_8 & 1)] * 6.011229e-02f));
          __2.x = (__3.x-v__2.x);
          __2.y = (__3.y-v__2.y);
        __1.x = exp2f(__2.x);
        __1.y = exp2f(__2.y);
        *(float2*)(acc_s + (i_8 * 2)) = __1;
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        scores_sum[i_9] = 0.000000e+00f;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          scores_sum[i_9] = (scores_sum[i_9] + acc_s[((((rv_1 & 7) * 4) + (i_9 * 2)) + (rv_1 >> 3))]);
        }
        scores_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_9]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        logsum[i_10] = ((logsum[i_10] * scores_scale[i_10]) + scores_sum[i_10]);
      }
      tl::syncthreads_partial(_mbarrier[7]);
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 5) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i_11 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) + 4096)])), __pack_half2(((half_t)acc_s[(i_11 * 8)]), ((half_t)acc_s[((i_11 * 8) + 1)])), __pack_half2(((half_t)acc_s[((i_11 * 8) + 2)]), ((half_t)acc_s[((i_11 * 8) + 3)])), __pack_half2(((half_t)acc_s[((i_11 * 8) + 4)]), ((half_t)acc_s[((i_11 * 8) + 5)])), __pack_half2(((half_t)acc_s[((i_11 * 8) + 6)]), ((half_t)acc_s[((i_11 * 8) + 7)])));
      }
      if (((((int)threadIdx.x) & 3) >> 1) == 0) {
        scores_scale_shared[((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((half_t)scores_scale[(((int)threadIdx.x) & 1)]);
      }


      #pragma unroll
      for (int i_13 = 0; i_13 < 16; ++i_13) {
        uint1 __5;
        float2 v__4 = *(float2*)(acc_s + (i_13 * 2));
        ((half2*)(&(__5.x)))->x = (half_t)(v__4.x);
        ((half2*)(&(__5.x)))->y = (half_t)(v__4.y);
        *(uint1*)(acc_s_cast + (i_13 * 2)) = __5;
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 64; ++i_14) {
        for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
          acc_o[((i_14 * 2) + vec_2)] = (acc_o[((i_14 * 2) + vec_2)] * scores_scale[(i_14 & 1)]);
        }
      }
      tl::fence_proxy_async();
      tl::gemm_rs<64, 256, 64, 4, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((k_1 & 1) * 32768) + 49152)])), (&(acc_o[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 & 1) + 4)]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 64; ++i_15) {
      for (int vec_3 = 0; vec_3 < 2; ++vec_3) {
        acc_o[((i_15 * 2) + vec_3)] = (acc_o[((i_15 * 2) + vec_3)] / logsum[(i_15 & 1)]);
      }
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      logsum[i_16] = (__log2f(logsum[i_16]) + (scores_max[i_16] * 6.011229e-02f));
    }
    tl::syncthreads_partial(_mbarrier[9]);
    #pragma unroll
    for (int i_17 = 0; i_17 < 32; ++i_17) {
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((i_17 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), __pack_half2(((half_t)acc_o[(i_17 * 8)]), ((half_t)acc_o[((i_17 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_17 * 8) + 2)]), ((half_t)acc_o[((i_17 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_17 * 8) + 4)]), ((half_t)acc_o[((i_17 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_17 * 8) + 6)]), ((half_t)acc_o[((i_17 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    tl::syncthreads_partial(_mbarrier[10]);
    if (((int)threadIdx.x) == 0) {
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[16384])), 0, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[20480])), 64, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[24576])), 128, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[28672])), 192, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[32768])), 256, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[36864])), 320, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[40960])), 384, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
      tl::tma_store(Output_desc, (&(((half_t*)buf_dyn_shmem)[45056])), 448, (((int)blockIdx.x) * 64), ((int)blockIdx.y));
    }
  }
}