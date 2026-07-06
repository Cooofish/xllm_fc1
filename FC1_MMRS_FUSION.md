# FC1 MMRS 融合算子实现整理

本文整理 `xllm_flashcomm1` 中 FC1 的 MatMul + ReduceScatter 融合算子链路。这里的融合算子指 FC1 sequence-parallel 场景下，把 row-parallel linear 的本地 matmul 和后续 reduce-scatter 合并为一次 `torch_npu::npu_mm_reduce_scatter_base` 调用。

## 开关关系

FC1 有一个总开关和一个融合算子子开关：

- `--enable_flashcomm1`：FC1 总开关，默认关闭。关闭后不会进入 sequence sharding，MMRS 也不会生效。
- `--enable_mmrs_fusion`：MMRS 融合算子开关，默认关闭。只有 FC1 已经启用且当前上下文满足 sequence-sharded 条件时，才决定 row-parallel reduce 是否尝试走 fused MMRS。
- `--mmrs_comm_mode`：传给 `torch_npu::npu_mm_reduce_scatter_base` 的通信模式，目前支持 `aiv`、`ai_cpu`、`none`。
- `--flashcomm1_min_prefill_tokens` / `--flashcomm1_min_decode_tokens`：控制 FC1 在 token 数达到阈值后才启用。当前代码实际只允许 prefill 侧进入 FC1。

代码位置：

- `xllm/core/common/flash_comm1_context.cpp`
- `xllm/core/common/flash_comm1_context.h`
- `xllm/core/common/options.h`
- `xllm/core/runtime/options.h`
- `xllm/xllm.cpp`

## 总体链路

MMRS 的调用链如下：

```text
Qwen3-Next model forward
  -> build_flash_comm1_context(...)
  -> shard_sequence(...)
  -> attention / mlp / gated-delta-net forward(..., fc1_ctx)
  -> row_parallel_linear.forward(input, row_parallel_reduce_mode_for_fc1(ctx), &ctx)
  -> RowParallelLinearImpl::forward(...)
  -> xllm::kernel::matmul_reduce_scatter(...)
  -> npu::matmul_reduce_scatter(...)
  -> torch_npu::npu_mm_reduce_scatter_base(...)
```

其中 `row_parallel_reduce_mode_for_fc1(ctx)` 根据 `ctx.enable_mmrs_fusion` 返回：

- `MATMUL_REDUCE_SCATTER`：尝试融合算子。
- `REDUCE_SCATTER`：普通 matmul 后走 reduce-scatter。

## FC1 上下文如何准备 MMRS

`build_flash_comm1_context` 在满足以下条件时才启用 FC1：

- `enable_flashcomm1=true`
- 当前是 prefill 路径
- 编译和运行设备是 NPU
- `dp_size == 1`
- `cp_size == 1`
- TP world size 大于 1
- 存在 TP process group
- token 数大于 `flashcomm1_min_prefill_tokens`

启用后，`FlashComm1Context` 会记录：

- `original_num_tokens`：原始 token 数。
- `padded_num_tokens`：按 `tp_world_size * 16` 对齐后的 token 数。
- `pad_size`：padding token 数。
- `padded_local_num_tokens`：每个 TP rank 的 padding 后本地 token 数。
- `enable_mmrs_fusion`：是否尝试 fused MMRS。
- `mmrs_comm_mode`：MMRS 通信模式。
- `tp_group`：reduce-scatter 使用的 process group。

sequence sharding 入口是 `shard_sequence`。它会把 token 维补齐到 `padded_num_tokens`，然后每个 TP rank 拿连续的 `padded_local_num_tokens` 行。

## RowParallelLinear 中怎么尝试融合

核心实现位于：

- `xllm/core/layers/common/linear.cpp`
- `xllm/core/layers/common/linear.h`

`RowParallelLinearImpl::forward(input, reduce_mode, fc1_ctx)` 中，只有非量化路径会尝试 MMRS。判断条件是：

```cpp
wants_mmrs(reduce_mode) &&
fc1_ctx &&
fc1_ctx->is_sequence_sharded() &&
fc1_ctx->enable_mmrs_fusion
```

进一步还要满足 shape 条件：

- input 必须是 2D。
- input 的第 0 维必须等于 `fc1_ctx->original_num_tokens`。
- 如果有 bias，则当前不允许存在 padding，即 `fc1_ctx->pad_size == 0`。

满足后，RowParallelLinear 会：

1. 必要时把 input padding 到 `fc1_ctx->padded_num_tokens`。
2. 构造期望输出 shape：
   - 第 0 维：`fc1_ctx->padded_local_num_tokens`
   - 第 1 维：`weight_.size(0)`
3. 创建 `mmrs_output`，主要用于传递期望 shape 和 dtype。
4. 将权重转成 MMRS 需要的 K-N layout：
   - `mmrs_weight_transposed()`
   - 内部缓存 `weight_.transpose(0, 1).contiguous()`
5. 构造 `MatmulReduceScatterParams`：
   - `a = mmrs_input`
   - `b = mmrs_weight_transposed()`
   - `bias = bias`
   - `process_group = process_group_`
   - `output = mmrs_output`
   - `comm_mode = fc1_ctx->mmrs_comm_mode`
6. 调用 `xllm::kernel::matmul_reduce_scatter(mmrs_params)`。
7. 如果返回 shape 等于期望 local shape，则认为 fused MMRS 命中并直接返回。
8. 如果返回 shape 不符合预期，则 fallback 到普通 matmul，再进入后续 reduce-scatter。

## Fallback 设计

MMRS 没有命中时不会直接失败，而是回退：

- shape 不满足时：普通 matmul + reduce-scatter。
- fused op 返回非 local shape 时：尝试修正 padding 后，继续走后续 reduce-scatter。
- NPU kernel 参数检查失败时：kernel wrapper 内部回退到 `torch::matmul`，调用者后续再 reduce-scatter。
- HCCL group 名为空时：kernel wrapper 内部回退到 `torch::matmul`。

这使得 `enable_mmrs_fusion=true` 更像“优先尝试融合”，不是硬要求所有 row-parallel linear 都必须走融合算子。

## 量化路径当前不走 MMRS

当前 fused MMRS 只接在非量化 row-parallel linear 路径。以下路径会记录 warning 并跳过 fused MMRS：

- smoothquant
- fp8
- w8a8
- w8a8_dynamic

原因是这些路径的 matmul 参数、scale/dequant、输出 dtype 处理和当前 `npu_mm_reduce_scatter_base` 封装还没有打通。代码中通过 `log_mmrs_quant_skip` 明确打印跳过原因。

## Kernel API 封装

跨设备统一 API：

- `xllm/core/kernels/param.h`
- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`

新增结构：

```cpp
struct MatmulReduceScatterParams {
  torch::Tensor a;
  torch::Tensor b;
  std::optional<torch::Tensor> bias;
  ProcessGroup* process_group = nullptr;
  int64_t original_num_tokens = 0;
  std::optional<torch::Tensor> output;
  std::string reduce_op = "sum";
  int64_t comm_turn = 0;
  int64_t stream_mode = 1;
  std::string comm_mode = "ai_cpu";
};
```

`ops_api.cpp` 中：

- NPU 编译时调用 `npu::matmul_reduce_scatter(...)`。
- 非 NPU 编译时退化成普通 `matmul`。

NPU 实现文件：

- `xllm/core/kernels/npu/matmul_reduce_scatter.cpp`

它最终调用：

```cpp
at_npu::native::custom_ops::npu_mm_reduce_scatter_base(...)
```

## NPU kernel wrapper 的参数检查

`matmul_reduce_scatter_reject_reason` 会在调用 torch_npu 前检查：

- `process_group` 不能为空。
- `output` 必须存在且 defined。
- `a`、`b`、`output` 必须是 2D。
- input dtype 只支持 FP16 / BF16。
- `a`、`b`、`output` dtype 必须一致。
- bias 如果存在，dtype 必须和 input 一致。
- matmul K 维必须匹配：`a.size(1) == b.size(0)`。
- output N 维必须匹配：`output.size(1) == b.size(1)`。

不满足时打印 warning，然后返回普通 `matmul(a, b) + bias`。

## HCCL group 和通信模式

wrapper 通过：

```cpp
process_group->hccl_comm_name(/*init_comm=*/true)
```

获取 HCCL group 名，并传给 `npu_mm_reduce_scatter_base`。

`comm_mode` 处理逻辑：

- `ai_cpu`：传给 torch_npu。
- `aiv`：传给 torch_npu。
- `none` 或空：不传 explicit comm mode，使用 torch_npu 默认。
- 其他值：打印 warning，使用 torch_npu 默认。

另外有一组 BF16 risky shape 保护逻辑：

- 当 `comm_mode` 是空或 `aiv`
- dtype 是 BF16
- `M == 2048`
- `N == 5120`
- 某些 world size 和 K 组合命中已知风险

则自动把 `effective_comm_mode` 从 `aiv` 切到 `ai_cpu`。

## 和普通 ReduceScatter 的关系

如果 `enable_mmrs_fusion=false`，`row_parallel_reduce_mode_for_fc1` 返回 `REDUCE_SCATTER`。此时流程是：

```text
row-parallel matmul
  -> maybe_pad_and_reduce(...)
  -> reduce_scatter_padded_local(...)
```

`reduce_scatter_padded_local` 会：

1. 校验输入是完整 real-token 输出。
2. 必要时 padding 到 `padded_num_tokens`。
3. 创建本地输出 `[padded_local_num_tokens, hidden]`。
4. 调用 `ctx.tp_group->reduce_scatter(padded_input, output)`。

因此 MMRS 的目标就是把“matmul 后再 reduce_scatter”这两步压成一个 NPU fused op，减少中间 full output 和通信调度开销。

## 当前限制

1. 当前只接入非量化 row-parallel linear。
2. 只在 FC1 sequence-sharded prefill 路径尝试。
3. 输入要求 2D，dtype 限定 FP16/BF16。
4. bias + padding 的情况当前不尝试 MMRS。
5. fused op 返回 shape 不符合 local output 预期时会 fallback。
6. MoE 的部分 FC1 reduce 仍显式使用普通 `REDUCE_SCATTER`，没有统一走 MMRS。

## 关键日志

命中 fused op：

```text
FC1 MMRS torch_npu hit
FC1 MMRS row-parallel fused output accepted
```

跳过或回退：

```text
FC1 MMRS skipped in row-parallel ... path
FC1 MMRS skipped for unsupported row-parallel shape
FC1 MMRS skipped before row-parallel matmul
FC1 MMRS torch_npu skipped
FC1 MMRS returned non-local shape
```

通信模式保护：

```text
FC1 MMRS comm_mode switched from aiv to ai_cpu for known risky torch_npu shape
```
