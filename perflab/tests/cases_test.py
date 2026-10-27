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

from collections.abc import Mapping, Sequence
import itertools
import os
import random
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from perflab import cases
from perflab import common
from perflab import context as context_lib
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from tests import seed_test_utils


def _factory() -> tuple[Any, ...]:
  return ("model", (), {}, None)


_RUN = registry_lib.PresubmitPolicy.RUN


def _make_spec(
    name: str,
    skipped_run_modes: frozenset[str] = frozenset(),
    stepper: step_lib.StepperType = step_lib.StepperType.FORWARD,
    presubmit: registry_lib.PresubmitPolicy = _RUN,
    presubmit_reason: str | None = None,
) -> registry_lib.BenchmarkSpec:
  return registry_lib.BenchmarkSpec(
      name=name,
      factory=_factory,
      stepper=stepper,
      skipped_run_modes=skipped_run_modes,
      presubmit=presubmit,
      presubmit_reason=presubmit_reason,
  )


# One spec per presubmit policy.
_PRESUBMIT_REGISTRY = {
    "run": _make_spec("run", presubmit=registry_lib.PresubmitPolicy.RUN),
    "smoke_only": _make_spec(
        "smoke_only",
        presubmit=registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
        presubmit_reason="too big",
    ),
    "skipped": _make_spec(
        "skipped",
        presubmit=registry_lib.PresubmitPolicy.SKIP,
        presubmit_reason="never in presubmit",
    ),
}


class PresubmitPolicyTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # `--fake_tensor_mode` is only valid on CPU.
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )

  def _case_benchmarks(
      self, run_scope: context_lib.RunScope, fake_tensor_mode: bool
  ) -> set[str]:
    with (
        mock.patch.dict(registry_lib.REGISTRY, _PRESUBMIT_REGISTRY, clear=True),
        flagsaver.flagsaver(
            run_scope=run_scope, fake_tensor_mode=fake_tensor_mode
        ),
    ):
      return {
          spec.name
          for _, spec, _ in cases.get_cases(
              target_lib.Platform.CPU, mode_lib.Framework.TORCH
          )
      }

  def test_full_scope_collects_every_benchmark(self):
    for fake_tensor_mode in (False, True):
      with self.subTest(fake_tensor_mode=fake_tensor_mode):
        self.assertEqual(
            self._case_benchmarks(context_lib.RunScope.FULL, fake_tensor_mode),
            set(_PRESUBMIT_REGISTRY),
        )

  def test_presubmit_scope_collects_runnable_benchmarks(self):
    self.assertEqual(
        self._case_benchmarks(context_lib.RunScope.PRESUBMIT, False), {"run"}
    )

  def test_presubmit_cpu_smoke_also_collects_smoke_only_benchmarks(self):
    self.assertEqual(
        self._case_benchmarks(context_lib.RunScope.PRESUBMIT, True),
        {"run", "smoke_only"},
    )

  def test_get_cases_logs_left_out_benchmarks(self):
    with self.assertLogs(level="INFO") as logs:
      self._case_benchmarks(context_lib.RunScope.PRESUBMIT, False)
    output = "\n".join(logs.output)
    self.assertIn("Left 2 benchmarks out", output)
    self.assertIn("CPU_SMOKE_ONLY (too big): smoke_only", output)
    self.assertIn("SKIP (never in presubmit): skipped", output)

  def test_get_cases_logs_nothing_in_full_scope(self):
    with self.assertNoLogs(level="INFO"):
      self._case_benchmarks(context_lib.RunScope.FULL, False)

  def test_presubmit_skip_reason(self):
    for name, fake_tensor_mode, expected in (
        ("run", False, None),
        ("run", True, None),
        ("smoke_only", True, None),
        (
            "smoke_only",
            False,
            (
                "Benchmark smoke_only runs in presubmit scope only in the"
                " FakeTensorMode CPU smoke (--fake_tensor_mode): too big"
            ),
        ),
        (
            "skipped",
            True,
            (
                "Benchmark skipped does not run in presubmit scope: never in"
                " presubmit"
            ),
        ),
    ):
      with self.subTest(name=name, fake_tensor_mode=fake_tensor_mode):
        self.assertEqual(
            cases.get_presubmit_skip_reason(
                _PRESUBMIT_REGISTRY[name], fake_tensor_mode=fake_tensor_mode
            ),
            expected,
        )

  def test_get_skip_reason_applies_presubmit_policy_in_presubmit_scope(self):
    spec = _PRESUBMIT_REGISTRY["skipped"]
    for run_scope, expected in (
        (context_lib.RunScope.FULL, None),
        (
            context_lib.RunScope.PRESUBMIT,
            (
                "Benchmark skipped does not run in presubmit scope: never in"
                " presubmit"
            ),
        ),
    ):
      with self.subTest(run_scope=run_scope):
        with flagsaver.flagsaver(run_scope=run_scope):
          self.assertEqual(
              cases.get_skip_reason(
                  spec,
                  common.RunMode.COMPILED,
                  framework=mode_lib.Framework.TORCH,
                  device_kind=target_lib.DeviceKind.TPU,
              ),
              expected,
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

  def _shard_assignment(
      self,
      case_names: Sequence[str],
      total_shards: int,
      env: Mapping[str, str],
  ) -> dict[str, int]:
    """Returns the shard each case of a fake benchmark class runs in."""
    params = cases.case_params(
        (name, _make_spec(name), common.RunMode.COMPILED) for name in case_names
    )
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
    assignment = {}
    with mock.patch.dict(
        os.environ,  # PY_ENVIRON_OK=test
        env,
        clear=False,
    ):
      for shard_index in range(total_shards):
        loader = cases.BenchmarkTestLoader()
        names = loader.getTestCaseNames(test_cls)
        for name in loader.shardTestCaseNames(
            itertools.cycle(range(total_shards)), names, shard_index
        ):
          self.assertNotIn(name, assignment)
          assignment[name] = shard_index
    self.assertCountEqual(assignment, params)
    return assignment

  def test_shard_assignment_does_not_depend_on_ordering_seeds(self):
    # CI seeds the ordering with the commit SHA, and auto-bisect reruns one
    # shard at each candidate commit, so a case must stay in the same shard
    # whatever the seeds are.
    case_names = [f"model_{i}_compiled" for i in range(40)]
    first = self._shard_assignment(
        case_names,
        total_shards=4,
        env={
            "TEST_RANDOMIZE_ORDERING_SEED": "12345",
            "TORCH_TPU_BENCHMARK_ORDER_SEED": (
                "adac8863c4f1d05b3f53a53aab84c8e77490766f"
            ),
        },
    )
    second = self._shard_assignment(
        case_names,
        total_shards=4,
        env={
            "TEST_RANDOMIZE_ORDERING_SEED": "67890",
            "TORCH_TPU_BENCHMARK_ORDER_SEED": (
                "0123abcd4567ef890123abcd4567ef8901234567"
            ),
        },
    )
    self.assertEqual(first, second)
    self.assertLen(set(first.values()), 4)

  def test_adding_a_case_does_not_move_other_cases(self):
    env = {
        "TEST_RANDOMIZE_ORDERING_SEED": "0",
        "TORCH_TPU_BENCHMARK_ORDER_SEED": "0123abcd",
    }
    case_names = [f"model_{i}_compiled" for i in range(40)]
    before = self._shard_assignment(case_names, total_shards=4, env=env)
    after = self._shard_assignment(
        case_names + ["new_model_compiled"], total_shards=4, env=env
    )
    del after["test_benchmark_new_model_compiled"]
    self.assertEqual(before, after)

  def test_shard_assignment_does_not_depend_on_iterator_position(self):
    # absltest shares one cyclic shard iterator across all test classes, so a
    # later class can see it mid-cycle.
    names = [f"test_benchmark_model_{i}_compiled" for i in range(20)]
    loader = cases.BenchmarkTestLoader()
    total_shards = 4
    for shard_index in range(total_shards):
      expected = loader.shardTestCaseNames(
          itertools.cycle(range(total_shards)), names, shard_index
      )
      for offset in range(1, total_shards):
        iterator = itertools.cycle(range(total_shards))
        for _ in range(offset):
          next(iterator)
        self.assertEqual(
            loader.shardTestCaseNames(iterator, names, shard_index),
            expected,
            msg=f"shard {shard_index}, iterator offset {offset}",
        )

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
