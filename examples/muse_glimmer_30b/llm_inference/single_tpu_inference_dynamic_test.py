# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
import runner_dynamic  # from muse_glimmer_30b.llm_inference


class SingleTpuInferenceDynamicTest(absltest.TestCase):
  """Manual benchmark entrypoints for Muse Glimmer 30B dynamic inference on TPU.

  Each test method represents a distinct execution mode and is intended to be
  run individually (exclusively) using `--test_filter=<test_name>`, rather
  than running all test methods together.
  """

  def test_muse_glimmer_bounded_dynamism(self):
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=False,
        enable_bounded_dynamism=True,
    )

  def test_muse_glimmer_bounded_dynamism_swa_prefill(self):
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=False,
        enable_bounded_dynamism=True,
        prefill_seq_len=2048,
    )

  def test_muse_glimmer_bounded_dynamism_swa_crossing(self):
    # The sliding window (2048) fills up during the decode steps.
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=False,
        enable_bounded_dynamism=True,
        prefill_seq_len=2044,
    )

  def test_muse_glimmer_no_bounded_dynamism(self):
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=False,
        enable_bounded_dynamism=False,
    )

  def test_muse_glimmer_random_weights_bounded_dynamism(self):
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=True,
        enable_bounded_dynamism=True,
    )

  def test_muse_glimmer_random_weights_bounded_dynamism_swa_prefill(self):
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=True,
        enable_bounded_dynamism=True,
        prefill_seq_len=2048,
    )

  def test_muse_glimmer_random_weights_bounded_dynamism_swa_crossing(self):
    # The sliding window (2048) fills up during the decode steps.
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=True,
        enable_bounded_dynamism=True,
        prefill_seq_len=2044,
    )

  def test_muse_glimmer_random_weights_multi_bucket_bounded_dynamism(self):
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=True,
        enable_bounded_dynamism=True,
        prefill_seq_len=128,
        num_decode_steps=256,
        num_dynamic_buckets=2,
        bucket_size=128,
    )

  def test_muse_glimmer_random_weights_multi_bucket_bounded_dynamism_swa_crossing(
      self,
  ):
    # The sliding window (2048) fills up during the decode steps while the
    # bounded KV cache lengths step from the first 128-aligned bucket
    # ([2, 2047]) into the second one ([2048, 2175]); the layer whose length
    # slices the rope/mask buffers (max_len = 2054) has its range capped at
    # 2053 and must keep the aligned capacity across the bucket switch.
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=True,
        enable_bounded_dynamism=True,
        prefill_seq_len=2044,
        num_dynamic_buckets=2,
        bucket_size=128,
    )

  def test_muse_glimmer_random_weights_no_bounded_dynamism(self):
    runner_dynamic.run_worker(
        device="tpu",
        use_random_weights=True,
        enable_bounded_dynamism=False,
    )


if __name__ == "__main__":
  absltest.main()
