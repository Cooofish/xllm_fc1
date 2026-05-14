/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "flash_comm1_context.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "framework/parallel_state/parallel_state.h"
#include "platform/device.h"

DECLARE_bool(enable_flashcomm1);
DECLARE_int32(flashcomm1_min_prefill_tokens);
DECLARE_int32(flashcomm1_min_decode_tokens);

namespace xllm {

FlashComm1Context build_flash_comm1_context(int32_t num_tokens,
                                            bool is_prefill,
                                            const ParallelArgs& parallel_args) {
  int32_t actual_tp_size = parallel_args.world_size() /
                           (parallel_args.dp_size() * parallel_args.cp_size());

  LOG(INFO) << "[FC1] build_flash_comm1_context called: num_tokens="
            << num_tokens << ", is_prefill=" << is_prefill
            << ", FLAGS_enable_flashcomm1=" << FLAGS_enable_flashcomm1
            << ", world_size=" << parallel_args.world_size()
            << ", dp_size=" << parallel_args.dp_size()
            << ", cp_size=" << parallel_args.cp_size()
            << ", calculated_tp_size=" << actual_tp_size
            << ", tp_size_field=" << parallel_args.tp_size()
            << ", tp_group_=" << (parallel_args.tp_group_ ? "exists" : "null")
            << ", device_type=" << Device::type_str();

  FlashComm1Context ctx;

  if (!FLAGS_enable_flashcomm1) {
    LOG(INFO) << "[FC1] ✗ Disabled: FLAGS_enable_flashcomm1=false";
    return ctx;
  }

  if (Device::type_str() != "npu") {
    LOG(INFO) << "[FC1] ✗ Disabled: device type=" << Device::type_str()
              << " != npu";
    return ctx;
  }

  if (actual_tp_size <= 1) {
    LOG(INFO) << "[FC1] ✗ Disabled: calculated_tp_size=" << actual_tp_size
              << " (world_size=" << parallel_args.world_size()
              << " / (dp_size=" << parallel_args.dp_size()
              << " * cp_size=" << parallel_args.cp_size() << ")) <= 1";
    return ctx;
  }

  ProcessGroup* tp_group = parallel_args.tp_group_;
  if (!tp_group) {
    LOG(INFO) << "[FC1] ✗ Disabled: tp_group_ is null (cannot perform TP "
                 "communication)";
    return ctx;
  }

  int32_t threshold = is_prefill ? FLAGS_flashcomm1_min_prefill_tokens
                                 : FLAGS_flashcomm1_min_decode_tokens;

  LOG(INFO) << "[FC1] threshold=" << threshold << " (is_prefill=" << is_prefill
            << ")";

  if (num_tokens < threshold) {
    LOG(INFO) << "[FC1] ✗ Disabled: num_tokens=" << num_tokens
              << " < threshold=" << threshold;
    return ctx;
  }

  LOG(INFO) << "[FC1] ✓ All checks passed! FC1 ENABLED";

  ctx.enabled = true;
  ctx.tp_rank = tp_group->rank();
  ctx.tp_world_size = tp_group->world_size();
  ctx.original_num_tokens = num_tokens;
  ctx.is_prefill = is_prefill;
  ctx.tp_group = tp_group;

  int32_t remainder = num_tokens % ctx.tp_world_size;
  ctx.pad_size = remainder == 0 ? 0 : ctx.tp_world_size - remainder;
  ctx.padded_num_tokens = num_tokens + ctx.pad_size;
  ctx.local_num_tokens = ctx.padded_num_tokens / ctx.tp_world_size;

  LOG(INFO) << "[FC1] Context: enabled=" << ctx.enabled
            << ", tp_rank=" << ctx.tp_rank
            << ", tp_world_size=" << ctx.tp_world_size
            << ", original_tokens=" << ctx.original_num_tokens
            << ", local_tokens=" << ctx.local_num_tokens
            << ", padded_tokens=" << ctx.padded_num_tokens
            << ", pad_size=" << ctx.pad_size;

  return ctx;
}

torch::Tensor shard_sequence(const torch::Tensor& input,
                             const FlashComm1Context& ctx) {
  if (!ctx.is_sequence_sharded()) {
    return input;
  }

  LOG(INFO) << "[FC1] shard_sequence: input shape=" << input.sizes();
  auto chunks = input.chunk(ctx.tp_world_size, 0);
  auto output = chunks[ctx.tp_rank].contiguous();
  LOG(INFO) << "[FC1] shard_sequence: output chunk[" << ctx.tp_rank
            << "] shape=" << output.sizes();
  return output;
}

torch::Tensor gather_sequence(const torch::Tensor& input,
                              const FlashComm1Context& ctx) {
  if (!ctx.is_sequence_sharded()) {
    return input;
  }

  LOG(INFO) << "[FC1] gather_sequence: input shape=" << input.sizes();

  int32_t current_local_size = input.size(0);
  int32_t expected_even_size = ctx.padded_num_tokens / ctx.tp_world_size;
  int32_t expected_uneven_base = ctx.original_num_tokens / ctx.tp_world_size;
  int32_t expected_uneven_remainder =
      ctx.original_num_tokens % ctx.tp_world_size;
  int32_t expected_uneven_for_rank =
      expected_uneven_base + (ctx.tp_rank < expected_uneven_remainder ? 1 : 0);

  std::vector<int32_t> token_num_list(ctx.tp_world_size);
  if (current_local_size == expected_even_size && ctx.pad_size > 0) {
    for (int32_t i = 0; i < ctx.tp_world_size; ++i) {
      token_num_list[i] = expected_even_size;
    }
    LOG(INFO) << "[FC1] gather_sequence: even distribution detected, using "
                 "token_num_list based on padded_num_tokens="
              << ctx.padded_num_tokens;
  } else {
    int32_t base_size = ctx.original_num_tokens / ctx.tp_world_size;
    int32_t remainder = ctx.original_num_tokens % ctx.tp_world_size;
    for (int32_t i = 0; i < ctx.tp_world_size; ++i) {
      token_num_list[i] = base_size + (i < remainder ? 1 : 0);
    }
    LOG(INFO) << "[FC1] gather_sequence: uneven distribution detected, using "
                 "token_num_list based on original_num_tokens="
              << ctx.original_num_tokens;
  }

  auto gathered = parallel_state::gather(input, ctx.tp_group, token_num_list);
  LOG(INFO) << "[FC1] gather_sequence: gathered shape=" << gathered.sizes()
            << ", token_num_list=[" << token_num_list[0] << ","
            << token_num_list[1] << "]";

  if (ctx.pad_size > 0 && gathered.size(0) > ctx.original_num_tokens) {
    LOG(INFO) << "[FC1] gather_sequence: unpadding from " << gathered.size(0)
              << " to " << ctx.original_num_tokens << " tokens";
    return gathered.slice(0, 0, ctx.original_num_tokens);
  }
  return gathered;
}

torch::Tensor gather_and_unpad_sequence(const torch::Tensor& input,
                                        const FlashComm1Context& ctx) {
  auto gathered = gather_sequence(input, ctx);
  if (ctx.pad_size > 0) {
    LOG(INFO) << "[FC1] gather_and_unpad_sequence: unpadding from "
              << gathered.size(0) << " to " << ctx.original_num_tokens
              << " tokens";
    return gathered.slice(0, 0, ctx.original_num_tokens);
  }
  LOG(INFO) << "[FC1] gather_and_unpad_sequence: no padding needed";
  return gathered;
}

torch::Tensor maybe_pad_for_reduce(const torch::Tensor& input,
                                   const FlashComm1Context& ctx) {
  // Dynamically check if current input needs padding, don't use static
  // ctx.pad_size
  int32_t current_size = input.size(0);
  int32_t remainder = current_size % ctx.tp_world_size;

  if (remainder == 0) {
    LOG(INFO) << "[FC1] maybe_pad_for_reduce: no padding needed, current_size="
              << current_size
              << " is divisible by tp_world_size=" << ctx.tp_world_size;
    return input;
  }

  int32_t dynamic_pad_size = ctx.tp_world_size - remainder;

  LOG(INFO) << "[FC1] maybe_pad_for_reduce: padding input from shape "
            << input.sizes() << ", current_size=" << current_size
            << ", remainder=" << remainder
            << ", dynamic_pad_size=" << dynamic_pad_size;

  auto options = input.options();
  auto padding = torch::zeros({dynamic_pad_size, input.size(-1)}, options);
  auto padded = torch::cat({input, padding}, 0);
  LOG(INFO) << "[FC1] maybe_pad_for_reduce: padded shape=" << padded.sizes();
  return padded;
}

torch::Tensor maybe_pad_and_reduce(torch::Tensor input,
                                   const FlashComm1Context& ctx,
                                   RowParallelReduceMode mode) {
  if (mode == RowParallelReduceMode::NONE) {
    return input;
  }

  LOG(INFO) << "[FC1] maybe_pad_and_reduce: mode=" << static_cast<int>(mode)
            << ", is_sequence_sharded=" << ctx.is_sequence_sharded();

  if (!ctx.is_sequence_sharded()) {
    if (ctx.tp_group && ctx.tp_group->world_size() > 1) {
      LOG(INFO) << "[FC1] maybe_pad_and_reduce: performing all_reduce";
      return parallel_state::reduce(input, ctx.tp_group);
    }
    return input;
  }

  LOG(INFO) << "[FC1] maybe_pad_and_reduce: performing reduce_scatter";
  auto padded = maybe_pad_for_reduce(input, ctx);
  auto result = parallel_state::reduce_scatter(padded, ctx.tp_group);
  LOG(INFO) << "[FC1] maybe_pad_and_reduce: result shape=" << result.sizes();
  return result;
}

torch::Tensor maybe_chunk_residual(const torch::Tensor& residual,
                                   int32_t tp_rank,
                                   int32_t tp_world_size) {
  if (tp_world_size <= 1) {
    return residual;
  }

  auto chunks = residual.chunk(tp_world_size, 0);
  return chunks[tp_rank].contiguous();
}

}  // namespace xllm