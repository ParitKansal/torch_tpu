/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.h"

#include <cstdint>
#include <optional>

#include "ATen/Context.h"
#include "ATen/SDPBackend.h"
#include "ATen/ops/empty.h"
#include "c10/core/ScalarType.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

TEST(ScaledDotProductAttentionAtenKernelsTest,
     PreferOverrideableForUnalignedSequenceForward) {
  const auto opts = at::TensorOptions().dtype(at::kBFloat16);
  const at::Tensor q = at::empty({1, 16, 1568, 64}, opts);
  const at::Tensor k = at::empty({1, 16, 1568, 64}, opts);
  const at::Tensor v = at::empty({1, 16, 1568, 64}, opts);

  const int64_t backend = AtenFusedSdpChoice(
      q, k, v, /*attn_mask=*/std::nullopt, /*dropout_p=*/0.0,
      /*is_causal=*/false, /*scale=*/std::nullopt, /*enable_gqa=*/false);
  EXPECT_EQ(backend, static_cast<int64_t>(at::SDPBackend::overrideable));
}

TEST(ScaledDotProductAttentionAtenKernelsTest,
     PreferOverrideableForNonPowerOfTwoHeadDimForward) {
  const auto opts = at::TensorOptions().dtype(at::kBFloat16);
  const at::Tensor q = at::empty({1, 16, 512, 80}, opts);
  const at::Tensor k = at::empty({1, 16, 512, 80}, opts);
  const at::Tensor v = at::empty({1, 16, 512, 80}, opts);

  const int64_t backend = AtenFusedSdpChoice(
      q, k, v, /*attn_mask=*/std::nullopt, /*dropout_p=*/0.0,
      /*is_causal=*/true, /*scale=*/std::nullopt, /*enable_gqa=*/false);
  EXPECT_EQ(backend, static_cast<int64_t>(at::SDPBackend::overrideable));
}

TEST(ScaledDotProductAttentionAtenKernelsTest,
     PreferOverrideableForAlignedMaskedCausalForward) {
  const auto opts = at::TensorOptions().dtype(at::kBFloat16);
  const at::Tensor q = at::empty({1, 16, 2048, 256}, opts);
  const at::Tensor k = at::empty({1, 16, 2048, 256}, opts);
  const at::Tensor v = at::empty({1, 16, 2048, 256}, opts);

  const int64_t causal_backend = AtenFusedSdpChoice(
      q, k, v, /*attn_mask=*/std::nullopt, /*dropout_p=*/0.0,
      /*is_causal=*/true, /*scale=*/std::nullopt, /*enable_gqa=*/false);
  EXPECT_EQ(causal_backend, static_cast<int64_t>(at::SDPBackend::overrideable));

  const int64_t unmasked_backend = AtenFusedSdpChoice(
      q, k, v, /*attn_mask=*/std::nullopt, /*dropout_p=*/0.0,
      /*is_causal=*/false, /*scale=*/std::nullopt, /*enable_gqa=*/false);
  EXPECT_EQ(unmasked_backend,
            static_cast<int64_t>(at::SDPBackend::cudnn_attention));
}

TEST(ScaledDotProductAttentionAtenKernelsTest,
     KeepFlashForAlignedTrainingAndFallbackForMaskGrad) {
  const auto opts = at::TensorOptions().dtype(at::kBFloat16);
  const at::Tensor q =
      at::empty({1, 16, 2048, 256}, opts).set_requires_grad(true);
  const at::Tensor k =
      at::empty({1, 16, 2048, 256}, opts).set_requires_grad(true);
  const at::Tensor v =
      at::empty({1, 16, 2048, 256}, opts).set_requires_grad(true);

  const int64_t train_backend = AtenFusedSdpChoice(
      q, k, v, /*attn_mask=*/std::nullopt, /*dropout_p=*/0.0,
      /*is_causal=*/true, /*scale=*/std::nullopt, /*enable_gqa=*/false);
  EXPECT_EQ(train_backend,
            static_cast<int64_t>(at::SDPBackend::cudnn_attention));

  const at::Tensor mask =
      at::empty({1, 1, 2048, 2048}, opts).set_requires_grad(true);
  const int64_t mask_grad_backend = AtenFusedSdpChoice(
      q, k, v, /*attn_mask=*/mask, /*dropout_p=*/0.0, /*is_causal=*/false,
      /*scale=*/std::nullopt, /*enable_gqa=*/false);
  EXPECT_EQ(mask_grad_backend, static_cast<int64_t>(at::SDPBackend::math));
}

}  // namespace
}  // namespace torch_tpu
