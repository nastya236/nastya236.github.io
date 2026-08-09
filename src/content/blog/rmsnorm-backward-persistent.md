---
title: 'RMSNorm backward (almost) without the column reduction'
description: 'The RMSNorm backward pass is memory-bandwidth-bound, and the naive weight gradient doubles the traffic. Here is how to get rid of it.'
pubDate: 2026-08-02
draft: true
---

RMSNorm is the default by now — in an autoregressive Transformer block you see it at least twice, often four times once you count the QK norms (and the same count again in the backward pass). Both passes are memory-bandwidth-bound: there's nowhere near enough arithmetic to hide the loads and stores behind. But backward is tricker than forward (we will see later why).
I ran into this while working on the CUDA backend in MLX <a href="#ref-1">[1]</a>. The idea isn't mine — I took it from QuACK's <a href="#ref-2">[2]</a> RMSNorm backward.
To see where the problem comes from, let's look at the math.

## The forward pass

Let $X \in \mathbb{R}^{M \times N}$ and $w \in \mathbb{R}^N$. RMSNorm scales each
row by its root mean square and then applies an element-wise multiplication by the
learnable weight:

$$
r_i = \frac{1}{\sqrt{\dfrac{1}{N}\sum_{j=1}^{N} X_{ij}^2 + \varepsilon}},
\qquad
\hat{X}_{ij} = r_i X_{ij},
\qquad
Y_{ij} = w_j \hat{X}_{ij}.
$$

The forward pass seems easy: we have one reduction over each row, which can be
handled quite naturally by launching one thread block per row (since hidden dim 
for mjority of models is <= 8192, it is covered by 1 thread block). So each thread
processes multiple elements, followed by a warp-level reduction, a thread-block
reduction, and finally the output write. Strightforward.

## The backward pass

Now let's look at the RMSNorm backward pass. We receive
$\frac{\partial L}{\partial Y} = gY$, the incoming gradient. We need to compute
two things:

- the gradient with respect to the weights, $\frac{\partial L}{\partial w}$;
- the gradient with respect to the input, $\frac{\partial L}{\partial X}$.

Let

$$
V_{ij} = gY_{ij} w_j
$$

denote the incoming gradient after applying the gain. Then the gradient with
respect to the input is
(Just a little more math and it is done)
$$
\frac{\partial L}{\partial X_{ij}}
= r_i V_{ij}
- r_i^3 X_{ij} \cdot \frac{1}{N}
\sum_{k=1}^{N} V_{ik} X_{ik}.
$$

Now, the gradient with respect to the weights is

$$
\frac{\partial L}{\partial w_j}
= \sum_{i=1}^{M} gY_{ij}\,\hat{X}_{ij}
= \sum_{i=1}^{M} gY_{ij}\,X_{ij}\,r_i.
$$

## The problem: summing across rows

To compute the weight gradient we need to accumulate contributions from all $M$ rows.
With a one-thread-block-per-row implementation, each block can compute its own
per-row contribution to $\frac{\partial L}{\partial w}$ — but those contributions
still have to be combined across rows. Another reduction axis.

There are two ways to do this:

1. use atomics to accumulate directly into the final $(N,)$ gradient;
2. write the per-row contributions to an intermediate $(M, N)$ tensor and do a
   second reduction over $M$.

The first sounds appealing: no intermediate at all. But it serializes every block
in the grid on the same $N$ addresses, and you inherit a non-deterministic
summation order over thousands of terms, so the same input won't give you the same
gradient twice. 

## The naive kernel

Let's talk about the second approach. First, we compute the per-row contributions
to the weight gradient and write the intermediate tensor $(M, N)$ to memory. Then
we reduce it down the columns.

That reduction cannot be done in one pass. The output is only $N$ floats, so if
each thread owns one output there are at most $N$ threads — 64 blocks of 128 on a
machine with ~148 SMs (B200), and eight times fewer once you use 16-byte vectorized
loads. At that occupancy there are nowhere near enough loads in flight to keep HBM
busy, and the reduction alone would cost more than the rest of the kernel. So you
split the rows across blocks too, every block emits a partial row, and those
partials need a final combine:

```cpp
rmsnorm_bwd_naive<<<M, nvec>>>(...);                     // gx + M x N partials
colsum_bf16<<<dim3(cdiv(nvec,128), NSPLIT), 128>>>(...); // M -> 64 partial rows
colsum_f32<<<cdiv(N,256), 256>>>(...);                   // 64 -> 1
```

Here is what the first kernel looks like in CUDA:

```cpp
template <int N_READS = 8>
__global__ void rmsnorm_bwd_naive(const bf16x8* __restrict__ x,
                                  const bf16x8* __restrict__ w,
                                  const bf16x8* __restrict__ g,
                                  bf16x8* __restrict__ gx,
                                  bf16x8* __restrict__ gw, int nvec,
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
```

### Helper: the block reduction

And some helper functions for the reduction. `cg::reduce` needs a scalar operator, so first a `plus` for our
packed pair of accumulators:

```cpp
namespace cg = cooperative_groups;

struct plus_f2 {
  __device__ float2 operator()(const float2& a, const float2& b) const {
    return make_float2(a.x + b.x, a.y + b.y);
  }
};
```

Then the whole block reduction is two `cg::reduce` calls with a shared-memory hop
between them:

```cpp
__device__ __forceinline__ float2
block_sum_f2(cg::thread_block& block, float2 v, float2* smem) {
  auto warp = cg::tiled_partition<32>(block);
  v = cg::reduce(warp, v, plus_f2{});
  if (warp.thread_rank() == 0) {
    smem[warp.meta_group_rank()] = v;
  }
  block.sync();
  v = warp.thread_rank() < warp.meta_group_size()
      ? smem[warp.thread_rank()]
      : make_float2(0.f, 0.f);
  v = cg::reduce(warp, v, plus_f2{});
  if (block.thread_rank() == 0) {
    smem[0] = v;
  }
  block.sync();
  return smem[0];
}
```

On a B200, with hidden dim 8192 and 16384 rows in bf16, this chain takes
**440.54 µs**. It moves 5 passes over the $M \times N$ tensor where only 3 are
unavoidable, which puts it around **38%** of the machine's 8 TB/s.

## Why this is not good enough

This works, but there is an obvious problem: we just wrote $M \times N$ values to
global memory only to read all of them back in the next kernel. And as you remember,
RMSNorm and it's backward are memory-bandwidth-bound operations, so this final column reduction is a
real bottleneck (we will see it later). Two of the five passes exist only to move
$\partial L/\partial w$ through memory — the reduction costs as much as the rest of
the kernel put together.

So the goal of this post is to explain how we can write the RMSNorm backward pass
with only a tiny column reduction — cutting the traffic, and with it the runtime.

## The persistent kernel

We want to reduce the size of the intermediate, which means the reduction across
rows has to happen inside the first kernel. How can we do it? The answer is:
make the kernel persistent.

In the naive version the grid is sized by the data: one block per row, so $M$
blocks. In a persistent kernel it's the opposite and the grid is sized by the machine: we
launch as many blocks as the GPU can keep resident (as many as SMs x k, where k is a heruistic),
and then each block iterates, taking a new row on every step.

"As many as fit" can be parsed using CUDA runtime API:

```cpp
int blocks_per_sm = 1;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, kernel, BLOCK_SIZE, smem_bytes);
int num_blocks = std::min<int64_t>(sm_count * blocks_per_sm, n_rows);
```

Each block then walks the rows in strides of `num_blocks`:

```cpp
int64_t cur_row = bid + tile * num_blocks;
```

Now a block sees many rows instead of one, so it can keep a running sum of its own
contribution to $\frac{\partial L}{\partial w}$ in registers and write it out once,
at the very end.

But this raises a question: how do we hide the latency?

In the standard many-block scenario we get it for free. There are far more blocks
than the SMs can run at once, so whenever a block stalls on a global load the
quater SM scheduler switches to another resident warp and the SM keeps doing useful work.
But persistent kernel gives that up by construction because we launch exactly enough
blocks just to fill the machine. 

So now we have to hide the latency ourselves by pipelining the loads through
shared memory with the `__pipeline_memcpy_async` (available from
Ampere). It copies global memory straight into shared memory without
staging through registers, and (the important part) it is asynchronous.

The components:

1. **Allocate two buffers in shared memory**, one per stage. While we compute on
   stage `t % 2`, the next row is landing in stage `(t+1) % 2`.
2. **Prefetch the first row** before the loop, then `__pipeline_commit()` to close
   that batch of copies.
3. **In the loop**, for row `t`:
   - issue the async copies for row `t+1` into the other buffer;
   - `__pipeline_commit()` to close that batch;
   - `__pipeline_wait_prior(1)` to wait for row `t`, but not for row `t+1`;
   - compute on row `t`, which is now sitting in shared memory.

In CUDA it looks like this. `N_CHUNKS` is how many 8-wide chunks each thread
handles along a row, so `BLOCK_SIZE * N_CHUNKS * 8 = N` and one block covers
exactly one row. `sx` and `sg` are the two-stage shared memory buffers, and
`num_tiles` is how many rows this block will end up visiting.

```cpp
...
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

    int next = tile + 1;
    int index = next % STAGES;
    int64_t next_row = (int64_t)bid + (int64_t)next * num_blocks;
    if (next < num_tiles && next_row < n_rows) {
#pragma unroll
      for (int j = 0; j < N_CHUNKS; j++) {
        int off = j * BLOCK_SIZE + tid;
        __pipeline_memcpy_async(&sx[index * NVEC + off], &x[off],
                                sizeof(bf16x8));
        __pipeline_memcpy_async(&sg[index * NVEC + off], &g[off],
                                sizeof(bf16x8));
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);

    // not every block gets a full complement of rows
    int64_t cur_row = (int64_t)bid + (int64_t)tile * num_blocks;
    if (cur_row < n_rows) {
      int cur = (tile % STAGES) * NVEC;  // the stage that just landed
      ....
    }
  }
```


The full kernel:

```cpp
template <int BLOCK_SIZE, int N_CHUNKS>
__global__ void rmsnorm_bwd_pipelined(const bf16x8* __restrict__ x,
                                      const bf16x8* __restrict__ w,
                                      const bf16x8* __restrict__ g,
                                      bf16x8* __restrict__ gx,
                                      float* __restrict__ gw, int n_rows,
                                      float eps) {
  constexpr int STAGES = 2;
  constexpr int NVEC = BLOCK_SIZE * N_CHUNKS;
  constexpr int AXIS = NVEC * 8;

  auto block = cg::this_thread_block();

  __shared__ float2 smem[32];
  // dynamic: two stages each for x and g
  extern __shared__ bf16x8 sbuf[];
  bf16x8* sx = sbuf;
  bf16x8* sg = sbuf + STAGES * NVEC;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_blocks = gridDim.x;
  int num_tiles = (n_rows + num_blocks - 1) / num_blocks;

  bf16x8 wv[N_CHUNKS], xv[N_CHUNKS], gv[N_CHUNKS];
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

    int next = tile + 1;
    int index = next % STAGES;
    int64_t next_row = (int64_t)bid + (int64_t)next * num_blocks;
    if (next < num_tiles && next_row < n_rows) {
#pragma unroll
      for (int j = 0; j < N_CHUNKS; j++) {
        int off = j * BLOCK_SIZE + tid;
        __pipeline_memcpy_async(&sx[index * NVEC + off], &x[off],
                                sizeof(bf16x8));
        __pipeline_memcpy_async(&sg[index * NVEC + off], &g[off],
                                sizeof(bf16x8));
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);

    int64_t cur_row = (int64_t)bid + (int64_t)tile * num_blocks;
    if (cur_row < n_rows) {
      int cur = (tile % STAGES) * NVEC;
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
          gwb[j][k] += gi * xi * normalizer;
          ov.v[k] =
              __float2bfloat16(normalizer * wi * gi - xi * meangwx * normalizer3);
        }
        gx_row[off] = ov;
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
```

Same shape as the naive version, minus the middle (M, N) reduction:

```cpp
kernel<<<nb, BLOCK, smem>>>(...);      
colsum_f32<<<cdiv(N,256), 256>>>(...);
```

## Results

One B200, bf16:

![Time per call against row count, for hidden dimensions 1024, 2048, 4096 and 8192. Dashed lines are the naive three-kernel path, solid lines the persistent two-kernel one. Both are straight, and the solid lines sit well below the dashed ones everywhere except at the smallest shape.](/rmsnorm-sweep.svg)

*Dashed = naive (3 kernels), solid = persistent (2 kernels). Colour is the hidden
dimension.*

## References

<span id="ref-1"></span>\[1] **MLX** — Apple. An array framework for machine
learning on Apple silicon and CUDA.
<https://github.com/ml-explore/mlx>

<span id="ref-2"></span>\[2] **QuACK** — QuACK: A Quirky Assortment of CuTe Kernels
<https://github.com/Dao-AILab/quack>


