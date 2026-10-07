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
import runner_static


class SingleTpuInferenceStaticTest(absltest.TestCase):
  """Manual benchmark entrypoints for Muse Glimmer 30B static inference on TPU.

  Each test method represents a distinct execution mode and is intended to be
  run individually (exclusively) using `--test_filter=<test_name>`, rather
  than running all test methods together.
  """

  def test_muse_glimmer_static(self):
    runner_static.run_worker(device="tpu", use_random_weights=False)

  def test_muse_glimmer_static_swa_prefill(self):
    runner_static.run_worker(
        device="tpu",
        use_random_weights=False,
        prefill_seq_len=2048,
    )

  def test_muse_glimmer_static_swa_crossing(self):
    # The sliding window (2048) fills up during the decode steps.
    runner_static.run_worker(
        device="tpu",
        use_random_weights=False,
        prefill_seq_len=2044,
    )

  def test_muse_glimmer_random_weights_static(self):
    runner_static.run_worker(device="tpu", use_random_weights=True)

  def test_muse_glimmer_random_weights_static_swa_prefill(self):
    runner_static.run_worker(
        device="tpu",
        use_random_weights=True,
        prefill_seq_len=2048,
    )

  def test_muse_glimmer_random_weights_static_swa_crossing(self):
    # The sliding window (2048) fills up during the decode steps.
    runner_static.run_worker(
        device="tpu",
        use_random_weights=True,
        prefill_seq_len=2044,
    )


if __name__ == "__main__":
  absltest.main()
