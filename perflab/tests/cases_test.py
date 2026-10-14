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

"""Tests for perflab/cases.py."""

import itertools
import os
import random
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
from perflab import cases
from perflab import common
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from tests import seed_test_utils


def _factory() -> tuple[Any, ...]:
  return ("model", (), {}, None)


def _make_spec(
    name: str,
    skipped_run_modes: frozenset[str] = frozenset(),
    stepper: step_lib.StepperType = step_lib.StepperType.FORWARD,
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=name,
      factory=_factory,
      stepper=stepper,
      skipped_run_modes=skipped_run_modes,
  )


class GetSkipReasonTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            known_failures,
            "KNOWN_FAILURES",
            (
                known_failures.KnownFailure(
                    benchmark="known_bad",
                    framework=mode_lib.Framework.TORCH,
                    run_modes=frozenset({common.RunMode.COMPILED}),
                    reason="compiled failure",
                ),
            ),
        )
    )

  def test_returns_none_for_case_to_run(self):
    self.assertIsNone(
        cases.get_skip_reason(
            _make_spec("known_bad"),
            common.RunMode.EAGER_DEFAULT,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        )
    )

  def test_returns_known_failure_reason(self):
    self.assertEqual(
        cases.get_skip_reason(
            _make_spec("known_bad"),
            common.RunMode.COMPILED,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        ),
        "Benchmark known_bad is known to fail: compiled failure",
    )

  def test_returns_skipped_run_mode_reason(self):
    self.assertEqual(
        cases.get_skip_reason(
            _make_spec("healthy", skipped_run_modes=frozenset({"compiled"})),
            common.RunMode.COMPILED,
            framework=mode_lib.Framework.TORCH,
            device_kind=target_lib.DeviceKind.TPU,
        ),
        "Benchmark healthy explicitly skips run mode compiled",
    )

  def test_get_cases_deterministic_across_shards_and_permuted_by_seed(self):
    fake_registry = {
        name: _make_spec(name)
        for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    }
    with mock.patch.dict(registry_lib.REGISTRY, fake_registry, clear=True):
      seed_a_first = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      seed_a_second = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      seed_b = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1002",
          )
      ]
      self.assertEqual(seed_a_first, seed_a_second)
      self.assertCountEqual(seed_a_first, seed_b)
      self.assertNotEqual(seed_a_first, seed_b)

  def test_get_cases_uses_env_seed_when_order_seed_omitted(self):
    fake_registry = {
        name: _make_spec(name)
        for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    }
    with (
        mock.patch.dict(registry_lib.REGISTRY, fake_registry, clear=True),
        mock.patch.dict(
            os.environ,  # PY_ENVIRON_OK=test
            {"TORCH_TPU_BENCHMARK_ORDER_SEED": "cl_1001"},
            clear=False,
        ),
    ):
      from_env = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
          )
      ]
      explicit = [
          name
          for name, _, _ in cases.get_cases(
              target_lib.Platform.CPU,
              mode_lib.Framework.TORCH,
              order_seed="cl_1001",
          )
      ]
      self.assertEqual(from_env, explicit)

  def test_seed_to_int_and_normalize_randomize_ordering_seed(self):
    self.assertEqual(cases.seed_to_int(""), 0)
    self.assertEqual(cases.seed_to_int("0"), 0)
    self.assertEqual(cases.seed_to_int("996893289"), 996893289)
    self.assertGreaterEqual(cases.seed_to_int("²"), 1)
    self.assertGreaterEqual(cases.seed_to_int("38215948283"), 1)
    self.assertLessEqual(cases.seed_to_int("38215948283"), 4294967295)

    hex_sha = "adac8863c4f1d05b3f53a53aab84c8e77490766f"
    sha_seed = cases.seed_to_int(hex_sha)
    self.assertGreaterEqual(sha_seed, 1)
    self.assertLessEqual(sha_seed, 4294967295)
    self.assertEqual(cases.seed_to_int(hex_sha), sha_seed)

    with mock.patch.dict(
        os.environ,  # PY_ENVIRON_OK=test
        {"TEST_RANDOMIZE_ORDERING_SEED": hex_sha},
        clear=False,
    ):
      loader = cases.BenchmarkTestLoader()
      self.assertEqual(
          os.environ["TEST_RANDOMIZE_ORDERING_SEED"],  # PY_ENVIRON_OK=test
          str(sha_seed),
      )
      self.assertEqual(loader._randomize_ordering_seed, sha_seed)

    flags.FLAGS.test_randomize_ordering_seed = hex_sha
    flags.FLAGS["test_randomize_ordering_seed"].present = True
    try:
      with mock.patch.dict(
          os.environ,  # PY_ENVIRON_OK=test
          {
              "TEST_RANDOMIZE_ORDERING_SEED": "",
              "TORCH_TPU_BENCHMARK_ORDER_SEED": "",
          },
          clear=False,
      ):
        loader = cases.BenchmarkTestLoader()
        self.assertEqual(loader._randomize_ordering_seed, sha_seed)
        self.assertEqual(
            os.environ["TEST_RANDOMIZE_ORDERING_SEED"],  # PY_ENVIRON_OK=test
            str(sha_seed),
        )
        self.assertEqual(
            flags.FLAGS.test_randomize_ordering_seed, str(sha_seed)
        )
    finally:
      flags.FLAGS.test_randomize_ordering_seed = ""
      flags.FLAGS["test_randomize_ordering_seed"].present = False

  def test_estimate_case_cost_and_lpt_shard_balancing(self):
    skipped_cost = cases.estimate_case_cost(
        _make_spec("known_bad"),
        common.RunMode.COMPILED,
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
    )
    self.assertEqual(skipped_cost, 0.0)

    run_skipped_cost = cases.estimate_case_cost(
        _make_spec("known_bad"),
        common.RunMode.COMPILED,
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
        skip_behavior="run_skipped",
    )
    self.assertGreater(run_skipped_cost, 0.0)

    torchax_decode_cost = cases.estimate_case_cost(
        _make_spec(
            "llama_1b_decode",
            stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
        ),
        common.RunMode.COMPILED,
        framework=mode_lib.Framework.TORCHAX,
        device_kind=target_lib.DeviceKind.TPU,
    )
    self.assertEqual(torchax_decode_cost, 0.0)

    eager_cnn_cost = cases.estimate_case_cost(
        _make_spec("resnet50"),
        common.RunMode.EAGER_DEFAULT,
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
    )
    compiled_llm_decode_cost = cases.estimate_case_cost(
        _make_spec(
            "llama_1b_decode",
            stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
        ),
        common.RunMode.COMPILED,
        framework=mode_lib.Framework.TORCH,
        device_kind=target_lib.DeviceKind.TPU,
    )
    self.assertGreater(compiled_llm_decode_cost, eager_cnn_cost)

    case_triples = [
        (
            "llama_1b_decode_compiled",
            _make_spec(
                "llama_1b_decode",
                stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
            ),
            common.RunMode.COMPILED,
        ),
        (
            "qwen3_decode_compiled",
            _make_spec(
                "qwen3_decode",
                stepper=step_lib.StepperType.DECODER_ONLY_DECODE,
            ),
            common.RunMode.COMPILED,
        ),
        (
            "resnet18_eager_default",
            _make_spec("resnet18"),
            common.RunMode.EAGER_DEFAULT,
        ),
        (
            "mobilenet_compiled",
            _make_spec("mobilenet"),
            common.RunMode.COMPILED,
        ),
        (
            "skipped_a_compiled",
            _make_spec("skipped_a", skipped_run_modes=frozenset({"compiled"})),
            common.RunMode.COMPILED,
        ),
        (
            "skipped_b_compiled",
            _make_spec("skipped_b", skipped_run_modes=frozenset({"compiled"})),
            common.RunMode.COMPILED,
        ),
        (
            "skipped_c_compiled",
            _make_spec("skipped_c", skipped_run_modes=frozenset({"compiled"})),
            common.RunMode.COMPILED,
        ),
        (
            "skipped_d_compiled",
            _make_spec("skipped_d", skipped_run_modes=frozenset({"compiled"})),
            common.RunMode.COMPILED,
        ),
    ]

    class FakeBenchmarkTest(seed_test_utils.RepeatableTest):
      framework = mode_lib.Framework.TORCH
      platform = target_lib.Platform.V7_1X1X1
      case_params = cases.case_params(case_triples)

      def test_benchmark_imports(self):
        pass

      def test_benchmark_llama_1b_decode_compiled(self):
        pass

      def test_benchmark_qwen3_decode_compiled(self):
        pass

      def test_benchmark_resnet18_eager_default(self):
        pass

      def test_benchmark_mobilenet_compiled(self):
        pass

      def test_benchmark_skipped_a_compiled(self):
        pass

      def test_benchmark_skipped_b_compiled(self):
        pass

      def test_benchmark_skipped_c_compiled(self):
        pass

      def test_benchmark_skipped_d_compiled(self):
        pass

    with mock.patch.dict(
        os.environ,  # PY_ENVIRON_OK=test
        {
            "TEST_RANDOMIZE_ORDERING_SEED": "0",
            "TORCH_TPU_BENCHMARK_ORDER_SEED": "",
        },
        clear=False,
    ):
      loader = cases.BenchmarkTestLoader()
      self.assertIsNone(loader._randomize_ordering_seed)
      ordered_names = loader.getTestCaseNames(FakeBenchmarkTest)
      self.assertEqual(ordered_names[0], "test_benchmark_imports")
      shard_0 = loader.shardTestCaseNames(
          itertools.cycle(range(2)), ordered_names, 0
      )
      shard_1 = loader.shardTestCaseNames(
          itertools.cycle(range(2)), ordered_names, 1
      )
      self.assertCountEqual(list(shard_0) + list(shard_1), ordered_names)
      self.assertEmpty(set(shard_0) & set(shard_1))
      # The two heavy decode compiled benchmarks must be split across the two
      # shards rather than assigned to the same shard.
      heavy_cases = {
          "test_benchmark_llama_1b_decode_compiled",
          "test_benchmark_qwen3_decode_compiled",
      }
      self.assertEqual(len(set(shard_0) & heavy_cases), 1)
      self.assertEqual(len(set(shard_1) & heavy_cases), 1)
      # Zero-cost skipped cases must also be balanced across shards rather than
      # collapsing onto the shard with slightly lower active load.
      skipped_cases = {
          "test_benchmark_skipped_a_compiled",
          "test_benchmark_skipped_b_compiled",
          "test_benchmark_skipped_c_compiled",
          "test_benchmark_skipped_d_compiled",
      }
      self.assertEqual(len(set(shard_0) & skipped_cases), 2)
      self.assertEqual(len(set(shard_1) & skipped_cases), 2)

  def test_each_case_runs_in_exactly_one_shard_whatever_the_name_order(self):
    # Each shard is a separate process with its own loader, which can see the
    # test names in a different order (e.g. absltest shuffles them by
    # --test_randomize_ordering_seed). The shards must still partition the
    # cases.
    case_triples = [
        (f"model_{i}_{mode.value}", _make_spec(f"model_{i}"), mode)
        for i in range(7)
        for mode in (common.RunMode.EAGER_DEFAULT, common.RunMode.COMPILED)
    ]
    params = cases.case_params(case_triples)
    test_cls = type(
        "FakeBenchmarkTest",
        (seed_test_utils.RepeatableTest,),
        {
            "framework": mode_lib.Framework.TORCH,
            "platform": target_lib.Platform.V7_1X1X1,
            "case_params": params,
            **{name: lambda self: None for name in params},
        },
    )
    total_shards = 3

    assigned = []
    with mock.patch.dict(
        os.environ,  # PY_ENVIRON_OK=test
        {
            "TEST_RANDOMIZE_ORDERING_SEED": "0",
            "TORCH_TPU_BENCHMARK_ORDER_SEED": "0123abcd",
        },
        clear=False,
    ):
      for shard_index in range(total_shards):
        loader = cases.BenchmarkTestLoader()
        names = loader.getTestCaseNames(test_cls)
        random.Random(shard_index).shuffle(names)
        assigned.extend(
            loader.shardTestCaseNames(
                itertools.cycle(range(total_shards)), names, shard_index
            )
        )

    self.assertCountEqual(assigned, params)


if __name__ == "__main__":
  absltest.main()
