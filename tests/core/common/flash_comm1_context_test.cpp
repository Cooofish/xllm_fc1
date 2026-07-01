#include "flash_comm1_context.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

namespace xllm {
namespace {

constexpr int32_t kFc1LocalTokenAlignment = 16;

FlashComm1Context make_context(int32_t num_tokens,
                               int32_t tp_world_size,
                               int32_t tp_rank) {
  FlashComm1Context ctx;
  ctx.enabled = true;
  ctx.tp_world_size = tp_world_size;
  ctx.tp_rank = tp_rank;
  ctx.original_num_tokens = num_tokens;
  const int32_t alignment = tp_world_size * kFc1LocalTokenAlignment;
  const int32_t remainder = num_tokens % alignment;
  ctx.pad_size = remainder == 0 ? 0 : alignment - remainder;
  ctx.padded_num_tokens = num_tokens + ctx.pad_size;
  ctx.local_num_tokens = ctx.local_num_tokens_for_rank(tp_rank);
  ctx.padded_local_num_tokens = ctx.padded_num_tokens / tp_world_size;
  return ctx;
}

TEST(FlashComm1ContextTest, BuildsUnevenTokenNumListForTp4) {
  auto ctx = make_context(/*num_tokens=*/5, /*tp_world_size=*/4, /*tp_rank=*/0);

  EXPECT_EQ(ctx.token_num_list(), (std::vector<int32_t>{2, 1, 1, 1}));
  EXPECT_EQ(ctx.local_num_tokens_for_rank(0), 2);
  EXPECT_EQ(ctx.local_num_tokens_for_rank(1), 1);
  EXPECT_EQ(ctx.local_num_tokens_for_rank(2), 1);
  EXPECT_EQ(ctx.local_num_tokens_for_rank(3), 1);
}

TEST(FlashComm1ContextTest, ShardSequenceUsesPaddedEvenTokenRanges) {
  auto input = torch::arange(28, torch::kFloat32).reshape({7, 4});
  auto zero = torch::zeros({57, 4}, torch::kFloat32);
  auto padded = torch::cat({input, zero}, 0);

  for (int32_t rank = 0; rank < 4; ++rank) {
    auto ctx = make_context(/*num_tokens=*/7, /*tp_world_size=*/4, rank);
    auto shard = shard_sequence(input, ctx);
    const auto start = rank * ctx.padded_local_num_tokens;
    const auto end = start + ctx.padded_local_num_tokens;
    EXPECT_TRUE(torch::equal(shard, padded.slice(0, start, end)))
        << "rank=" << rank;
  }
}

TEST(FlashComm1ContextTest, MaybeShardResidualMatchesPaddedShardSequence) {
  auto residual = torch::arange(20, torch::kFloat32).reshape({5, 4});

  for (int32_t rank = 0; rank < 4; ++rank) {
    auto ctx = make_context(/*num_tokens=*/5, /*tp_world_size=*/4, rank);
    auto shard = maybe_shard_residual(residual, ctx);
    EXPECT_TRUE(torch::equal(shard, shard_sequence(residual, ctx)))
        << "rank=" << rank;
  }
}

}  // namespace
}  // namespace xllm
