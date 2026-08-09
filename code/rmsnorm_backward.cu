#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/cmath>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(x)               \
  do {                              \
    cudaError_t e = (x);            \
    if (e != cudaSuccess) {         \
      printf(                       \
          "CUDA error %s:%d: %s\n", \
          __FILE__,                 \
          __LINE__,                 \
          cudaGetErrorString(e));   \
      exit(1);                      \
    }                               \
  } while (0)

struct __align__(16) bf16x8 {
  __nv_bfloat16 v[8];
};

struct __align__(16) f32x4 {
  float v[4];
};

struct plus_f2 {
  __device__ float2 operator()(const float2& a, const float2& b) const {
    return make_float2(a.x + b.x, a.y + b.y);
  }
};

__device__ __forceinline__ float2
block_sum_f2(cg::thread_block& block, float2 v, float2* smem) {
  auto warp = cg::tiled_partition<32>(block);
  v = cg::reduce(warp, v, plus_f2{});
  if (warp.thread_rank() == 0) {
    smem[warp.meta_group_rank()] = v;
  }
  block.sync();
  v = warp.thread_rank() < warp.meta_group_size() ? smem[warp.thread_rank()]
                                                  : make_float2(0.f, 0.f);
  v = cg::reduce(warp, v, plus_f2{});
  if (block.thread_rank() == 0) {
    smem[0] = v;
  }
  block.sync();
  return smem[0];
}

template <int N_READS = 8>
__global__ void rmsnorm_bwd_naive(
    const bf16x8* __restrict__ x,
    const bf16x8* __restrict__ w,
    const bf16x8* __restrict__ g,
    bf16x8* __restrict__ gx,
    bf16x8* __restrict__ gw,
    int nvec,
    float eps) {
  // One thread block per row: block i owns row i and nothing else, so the
  // row-wise reduction never has to leave the block.
  auto block = cg::this_thread_block();
  int row = blockIdx.x;
  int tid = threadIdx.x;
  // shared memory for the reduction: one float2 per warp
  __shared__ float2 smem[32];
  // Each thread takes one 128-bit chunk = 8 bf16 values. nvec is the row
  // length in chunks, so nvec threads cover the whole row
  size_t off = (size_t)row * nvec + tid;
  bf16x8 xv = x[off]; // input
  bf16x8 gv = g[off]; // grad
  bf16x8 wv = w[tid]; // weight

  // Two sums at once, packed into a float2:
  // .x accumulates  V_ij * X_ij -> the correction term
  // .y accumulates  X_ij^2 -> the normalizer r_i
  float2 factors = make_float2(0.f, 0.f);
// reduction within a thread
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    float t = __bfloat162float(xv.v[i]);
    // wg is V_ij = gY_ij * w_j
    float wg = __bfloat162float(wv.v[i]) * __bfloat162float(gv.v[i]);
    factors.x += wg * t;
    factors.y += t * t;
  }
  // Warp-level reduce->shared memory-> block reduce
  factors = block_sum_f2(block, factors, smem);

  int axis_size = nvec * N_READS;
  float meangwx = factors.x / axis_size;
  float normalizer = rsqrtf(factors.y / axis_size + eps);
  float normalizer3 = normalizer * normalizer * normalizer;

  bf16x8 gxv, gwv;
#pragma unroll
  for (int i = 0; i < N_READS; i++) {
    float xi = __bfloat162float(xv.v[i]);
    float wi = __bfloat162float(wv.v[i]);
    float gi = __bfloat162float(gv.v[i]);
    gxv.v[i] =
        __float2bfloat16(normalizer * wi * gi - xi * meangwx * normalizer3);
    gwv.v[i] = __float2bfloat16(gi * xi * normalizer);
  }
  gx[off] = gxv;
  gw[off] = gwv; // M x N partials for a second kernel to reduce
}

template <int BLOCK_SIZE, int N_CHUNKS>
__global__ void rmsnorm_bwd_pipelined(
    const bf16x8* __restrict__ x,
    const bf16x8* __restrict__ w,
    const bf16x8* __restrict__ g,
    bf16x8* __restrict__ gx,
    float* __restrict__ gw,
    int n_rows,
    float eps) {
  constexpr int STAGES = 2;
  constexpr int NVEC = BLOCK_SIZE * N_CHUNKS;
  constexpr int AXIS = NVEC * 8;

  auto block = cg::this_thread_block();

  __shared__ float2 smem[32];
  extern __shared__ bf16x8 sbuf[];
  bf16x8* sx = sbuf;
  bf16x8* sg = sbuf + STAGES * NVEC;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_blocks = gridDim.x;
  int num_tiles = cuda::ceil_div(n_rows, num_blocks);

  bf16x8 wv[N_CHUNKS], xv[N_CHUNKS], gv[N_CHUNKS];
  // this block's running dL/dw, in registers, in fp32
  float gwb[N_CHUNKS][8];
#pragma unroll
  for (int j = 0; j < N_CHUNKS; j++) {
#pragma unroll
    for (int k = 0; k < 8; k++) {
      gwb[j][k] = 0.f;
    }
  }

  x += (size_t)NVEC * bid;
  g += (size_t)NVEC * bid;
  gx += (size_t)NVEC * bid;
  gw += (size_t)AXIS * bid;

  // prefetch the first row, and load the weights to registers while we wait --
  // w is reused by every row this block touches, so it never leaves registers
#pragma unroll
  for (int j = 0; j < N_CHUNKS; j++) {
    int off = j * BLOCK_SIZE + tid;
    if (bid < n_rows) {
      __pipeline_memcpy_async(&sx[off], &x[off], sizeof(bf16x8));
      __pipeline_memcpy_async(&sg[off], &g[off], sizeof(bf16x8));
    }
    wv[j] = w[off];
  }
  __pipeline_commit();

  for (int tile = 0; tile < num_tiles; tile++) {
    x += (size_t)NVEC * num_blocks;
    g += (size_t)NVEC * num_blocks;

    // issue the loads for the next row into the other stage
    int next = tile + 1;
    int index = next % STAGES;
    int64_t next_row = (int64_t)bid + (int64_t)next * num_blocks;
    if (next < num_tiles && next_row < n_rows) {
#pragma unroll
      for (int j = 0; j < N_CHUNKS; j++) {
        int off = j * BLOCK_SIZE + tid;
        __pipeline_memcpy_async(
            &sx[index * NVEC + off], &x[off], sizeof(bf16x8));
        __pipeline_memcpy_async(
            &sg[index * NVEC + off], &g[off], sizeof(bf16x8));
      }
    }
    __pipeline_commit(); // always commit -- empty batch at the tail
    __pipeline_wait_prior(1); // wait for THIS row, not the one in flight

    // not every block gets a full complement of rows
    int64_t cur_row = (int64_t)bid + (int64_t)tile * num_blocks;
    if (cur_row < n_rows) {
      int cur = (tile % STAGES) * NVEC; // the stage that just landed
      float2 factors = make_float2(0.f, 0.f);
#pragma unroll
      for (int j = 0; j < N_CHUNKS; j++) {
        int off = j * BLOCK_SIZE + tid;
        xv[j] = sx[cur + off];
        gv[j] = sg[cur + off];
#pragma unroll
        for (int k = 0; k < 8; k++) {
          float t = __bfloat162float(xv[j].v[k]);
          float wg =
              __bfloat162float(wv[j].v[k]) * __bfloat162float(gv[j].v[k]);
          factors.x += wg * t;
          factors.y += t * t;
        }
      }
      factors = block_sum_f2(block, factors, smem);
      float meangwx = factors.x / AXIS;
      float normalizer = rsqrtf(factors.y / AXIS + eps);
      float normalizer3 = normalizer * normalizer * normalizer;

      bf16x8* gx_row = gx + (size_t)tile * num_blocks * NVEC;
#pragma unroll
      for (int j = 0; j < N_CHUNKS; j++) {
        int off = j * BLOCK_SIZE + tid;
        bf16x8 ov;
#pragma unroll
        for (int k = 0; k < 8; k++) {
          float xi = __bfloat162float(xv[j].v[k]);
          float wi = __bfloat162float(wv[j].v[k]);
          float gi = __bfloat162float(gv[j].v[k]);
          // dL/dw accumulates across rows and is NOT written here
          gwb[j][k] += gi * xi * normalizer;
          ov.v[k] = __float2bfloat16(
              normalizer * wi * gi - xi * meangwx * normalizer3);
        }
        gx_row[off] = ov; // dL/dX is final and leaves immediately
      }
    }
  }

#pragma unroll
  for (int j = 0; j < N_CHUNKS; j++) {
    float* p = gw + (size_t)(j * BLOCK_SIZE + tid) * 8;
    f32x4 lo, hi;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      lo.v[k] = gwb[j][k];
      hi.v[k] = gwb[j][k + 4];
    }
    reinterpret_cast<f32x4*>(p)[0] = lo;
    reinterpret_cast<f32x4*>(p)[1] = hi;
  }
}

__global__ void colsum_bf16(
    const bf16x8* __restrict__ in,
    float* __restrict__ out,
    int nvec,
    int rows) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= nvec) {
    return;
  }
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int r = blockIdx.y; r < rows; r += gridDim.y) {
    bf16x8 v = in[(size_t)r * nvec + col];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      acc[i] += __bfloat162float(v.v[i]);
    }
  }
  float* p = out + ((size_t)blockIdx.y * nvec + col) * 8;
  f32x4 lo, hi;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    lo.v[i] = acc[i];
    hi.v[i] = acc[i + 4];
  }
  reinterpret_cast<f32x4*>(p)[0] = lo;
  reinterpret_cast<f32x4*>(p)[1] = hi;
}

__global__ void colsum_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n,
    int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float acc = 0.f;
  for (int r = 0; r < rows; r++) {
    acc += in[(size_t)r * n + i];
  }
  out[i] = acc;
}

template <typename Kernel>
static int persistent_blocks(
    Kernel kernel,
    int block_size,
    size_t smem_bytes,
    int n_rows) {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int sm_count = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int blocks_per_sm = 1;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, block_size, smem_bytes));
  int want = sm_count * std::max(blocks_per_sm, 1);
  return std::max(1, std::min(want, n_rows));
}

// cpu reference
static void check_bwd(
    const char* name,
    const std::vector<__nv_bfloat16>& hx,
    const std::vector<__nv_bfloat16>& hw,
    const std::vector<__nv_bfloat16>& hg,
    const std::vector<__nv_bfloat16>& hgx,
    const std::vector<float>& hgw,
    int M,
    int N,
    float eps) {
  std::vector<double> gw_ref(N, 0.0);
  float gx_err = 0.f, gx_max = 0.f;
  for (int r = 0; r < M; r++) {
    const __nv_bfloat16* xr = hx.data() + (size_t)r * N;
    const __nv_bfloat16* gr = hg.data() + (size_t)r * N;
    const __nv_bfloat16* gxr = hgx.data() + (size_t)r * N;
    float s2 = 0.f, swgx = 0.f;
    for (int i = 0; i < N; i++) {
      float xi = __bfloat162float(xr[i]);
      float gi = __bfloat162float(gr[i]);
      float wi = __bfloat162float(hw[i]);
      s2 += xi * xi;
      swgx += wi * gi * xi;
    }
    float normalizer = 1.f / std::sqrt(s2 / N + eps);
    float meangwx = swgx / N;
    float normalizer3 = normalizer * normalizer * normalizer;
    for (int i = 0; i < N; i++) {
      float xi = __bfloat162float(xr[i]);
      float gi = __bfloat162float(gr[i]);
      float wi = __bfloat162float(hw[i]);
      float ref = normalizer * wi * gi - xi * meangwx * normalizer3;
      gw_ref[i] += (double)(gi * xi * normalizer);
      gx_err = std::max(gx_err, std::fabs(__bfloat162float(gxr[i]) - ref));
      gx_max = std::max(gx_max, std::fabs(ref));
    }
  }
  double gw_err = 0.0, gw_max = 0.0;
  for (int i = 0; i < N; i++) {
    gw_err = std::max(gw_err, std::fabs((double)hgw[i] - gw_ref[i]));
    gw_max = std::max(gw_max, std::fabs(gw_ref[i]));
  }
  printf(
      "  %-16s dx: rel=%.4g abs=%.4g   dw: rel=%.4g abs=%.4g\n",
      name,
      gx_err / (gx_max + 1e-30f),
      gx_err,
      gw_err / (gw_max + 1e-30),
      gw_err);
}

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 8192;
  int M = argc > 2 ? atoi(argv[2]) : 16384;
  int nvec = N / 8;
  size_t elems = (size_t)M * N;
  size_t bytes = elems * sizeof(__nv_bfloat16);
  float eps = 1e-5f;

  printf("N=%d  M=%d  x=%.0f MB\n", N, M, bytes / 1e6);

  std::vector<__nv_bfloat16> hx(elems), hw(N), hg(elems), hgx(elems);
  std::vector<float> hgw(N);
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (size_t i = 0; i < elems; i++)
    hx[i] = __float2bfloat16(nd(rng) * 0.02f);
  for (int i = 0; i < N; i++)
    hw[i] = __float2bfloat16(1.0f + nd(rng) * 0.1f);
  for (size_t i = 0; i < elems; i++)
    hg[i] = __float2bfloat16(nd(rng));

  const int NSPLIT = 64;
  __nv_bfloat16 *dx, *dw, *dg, *dgx, *dgw_rows;
  float *dgw, *dgw_part;
  CUDA_CHECK(cudaMalloc(&dx, bytes));
  CUDA_CHECK(cudaMalloc(&dw, N * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dg, bytes));
  CUDA_CHECK(cudaMalloc(&dgx, bytes));
  CUDA_CHECK(cudaMalloc(&dgw_rows, bytes));
  CUDA_CHECK(cudaMalloc(&dgw, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dgw_part, (size_t)NSPLIT * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(
      dw, hw.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dg, hg.data(), bytes, cudaMemcpyHostToDevice));

  auto bx = reinterpret_cast<const bf16x8*>(dx);
  auto bw = reinterpret_cast<const bf16x8*>(dw);
  auto bg = reinterpret_cast<const bf16x8*>(dg);
  auto bgx = reinterpret_cast<bf16x8*>(dgx);
  auto bgw_rows = reinterpret_cast<bf16x8*>(dgw_rows);
  auto bench =
      [&](const char* name, double traffic, std::function<void()> launch) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 20; i++)
          launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        int iters = 100;
        cudaEventRecord(s);
        for (int i = 0; i < iters; i++)
          launch();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        ms /= iters;
        printf(
            "%-16s %8.2f us   %7.0f GB/s\n",
            name,
            ms * 1e3,
            traffic / (ms / 1e3) / 1e9);
        cudaEventDestroy(s);
        cudaEventDestroy(e);
      };

  auto download = [&] {
    CUDA_CHECK(cudaMemcpy(hgx.data(), dgx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hgw.data(), dgw, N * sizeof(float), cudaMemcpyDeviceToHost));
  };

  if (nvec <= 1024 && nvec % 32 == 0) {
    double traffic = 5.0 * (double)bytes +
        2.0 * (double)NSPLIT * N * sizeof(float) + (double)N * sizeof(float);
    bench("bwd_naive", traffic, [&] {
      rmsnorm_bwd_naive<<<M, nvec>>>(bx, bw, bg, bgx, bgw_rows, nvec, eps);
      colsum_bf16<<<dim3(cuda::ceil_div(nvec, 128), NSPLIT), 128>>>(
          bgw_rows, dgw_part, nvec, M);
      colsum_f32<<<cuda::ceil_div(N, 256), 256>>>(dgw_part, dgw, N, NSPLIT);
    });
    printf(
        "  3 kernels, %.0f MiB moved (%.2f x M*N bf16), writes %d x %d bf16 "
        "dw partials\n",
        traffic / 1048576.0,
        traffic / (double)bytes,
        M,
        N);
    download();
    check_bwd("bwd_naive", hx, hw, hg, hgx, hgw, M, N, eps);
  } else {
    printf(
        "bwd_naive       skipped (needs nvec <= 1024 and nvec %% 32 == 0, "
        "nvec=%d)\n",
        nvec);
  }

  auto bench_pipe = [&](const char* name, auto block_tag, auto chunks_tag) {
    constexpr int BLOCK = decltype(block_tag)::value;
    constexpr int CHUNKS = decltype(chunks_tag)::value;
    if (nvec != BLOCK * CHUNKS) {
      return;
    }
    auto kernel = rmsnorm_bwd_pipelined<BLOCK, CHUNKS>;
    size_t smem = (size_t)2 * 2 * BLOCK * CHUNKS * sizeof(bf16x8);
    if (smem > 48000) {
      CUDA_CHECK(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          (int)smem));
    }
    int nb = persistent_blocks(kernel, BLOCK, smem, M);
    float* part = nullptr;
    CUDA_CHECK(cudaMalloc(&part, (size_t)nb * N * sizeof(float)));
    double traffic = 3.0 * (double)bytes +
        2.0 * (double)nb * N * sizeof(float) + (double)N * sizeof(float);
    bench(name, traffic, [&] {
      kernel<<<nb, BLOCK, smem>>>(bx, bw, bg, bgx, part, M, eps);
      colsum_f32<<<cuda::ceil_div(N, 256), 256>>>(part, dgw, N, nb);
    });
    printf(
        "  2 kernels, %.0f MiB moved (%.2f x M*N bf16), blocks=%d  "
        "smem=%zu KB\n",
        traffic / 1048576.0,
        traffic / (double)bytes,
        nb,
        smem / 1024);
    download();
    check_bwd(name, hx, hw, hg, hgx, hgw, M, N, eps);
    CUDA_CHECK(cudaFree(part));
  };

  using i64 = std::integral_constant<int, 64>;
  using i128 = std::integral_constant<int, 128>;
  using i256 = std::integral_constant<int, 256>;
  using i512 = std::integral_constant<int, 512>;
  bench_pipe("bwd_pipe_128x1", i128{}, std::integral_constant<int, 1>{});
  bench_pipe("bwd_pipe_128x2", i128{}, std::integral_constant<int, 2>{});
  bench_pipe("bwd_pipe_128x4", i128{}, std::integral_constant<int, 4>{});
  bench_pipe("bwd_pipe_128x8", i128{}, std::integral_constant<int, 8>{});
  bench_pipe("bwd_pipe_64x8", i64{}, std::integral_constant<int, 8>{});
  bench_pipe("bwd_pipe_256x2", i256{}, std::integral_constant<int, 2>{});
  bench_pipe("bwd_pipe_256x4", i256{}, std::integral_constant<int, 4>{});
  bench_pipe("bwd_pipe_512x2", i512{}, std::integral_constant<int, 2>{});

  CUDA_CHECK(cudaGetLastError());
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(dg);
  cudaFree(dgx);
  cudaFree(dgw_rows);
  cudaFree(dgw);
  cudaFree(dgw_part);
  return 0;
}
