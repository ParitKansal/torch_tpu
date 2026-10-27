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

"""Provides logic to resolve benchmark cases."""

import collections
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
import hashlib
import os
import random
from typing import Any
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
from perflab import common
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import known_failures
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import target as target_lib

CaseParams = tuple[registry_lib.BenchmarkSpec, common.RunMode]

_MAX_ABSLTEST_SEED = 4294967295


def _case_order_key(name: str, seed: str) -> str:
  """Returns a deterministic hash key for ordering benchmark cases."""
  payload = f"{seed}:{name}" if seed else name
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_shard_hash(name: str) -> int:
  """Returns a hash of `name` that never changes across commits or processes."""
  digest = hashlib.sha256(name.encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big")


def _get_order_seed(env: Mapping[str, str] | None = None) -> str:
  """Resolves the benchmark ordering seed from environment variables."""
  if env is None:
    env = os.environ  # PY_ENVIRON_OK=benchmark
  seed = env.get("TORCH_TPU_BENCHMARK_ORDER_SEED", "").strip()
  if seed:
    return seed
  fallback = env.get("TEST_RANDOMIZE_ORDERING_SEED", "").strip()
  if fallback and fallback not in ("0", "random"):
    return fallback
  return ""


def seed_to_int(seed: str) -> int:
  """Converts an arbitrary seed string into a valid positive absltest seed."""
  cleaned = seed.strip()
  if not cleaned or cleaned == "0":
    return 0
  if cleaned.isascii() and cleaned.isdigit():
    value = int(cleaned)
    if 1 <= value <= _MAX_ABSLTEST_SEED:
      return value
  digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
  return 1 + (int(digest, 16) % _MAX_ABSLTEST_SEED)


def case_params(
    cases_iter: Iterable[
        tuple[str, registry_lib.BenchmarkSpec, common.RunMode]
    ],
    method_name: str = "test_benchmark",
) -> Mapping[str, CaseParams]:
  """Maps the names parameterized.named_parameters generates to (spec, mode)."""
  return {
      f"{method_name}_{name}": (spec, mode) for name, spec, mode in cases_iter
  }


def normalize_randomize_ordering_seed(
    env: MutableMapping[str, str] | None = None,
) -> int:
  """Normalizes test ordering seed env vars into a valid integer for absltest.

  `absltest` raises `ValueError` if `TEST_RANDOMIZE_ORDERING_SEED` is set to a
  non-decimal string (such as a Git commit SHA) or an integer outside
  `[1, 4294967295]`. This function converts any non-empty string seed in
  `TEST_RANDOMIZE_ORDERING_SEED` (or falls back to
  `TORCH_TPU_BENCHMARK_ORDER_SEED` when `TEST_RANDOMIZE_ORDERING_SEED` is
  unset) into a deterministic positive 32-bit integer in `env`.

  Args:
    env: Environment mapping to normalize in place; defaults to `os.environ`.

  Returns:
    The normalized integer seed (0 if randomization is disabled or 'random').
  """
  flag_seed = ""
  if env is None:
    env = os.environ  # PY_ENVIRON_OK=benchmark
    if (
        flags.FLAGS.is_parsed()
        and "test_randomize_ordering_seed" in flags.FLAGS
        and flags.FLAGS["test_randomize_ordering_seed"].present
    ):
      flag_seed = str(flags.FLAGS.test_randomize_ordering_seed).strip()
      if flag_seed and flag_seed not in ("0", "random"):
        flags.FLAGS.test_randomize_ordering_seed = str(seed_to_int(flag_seed))
  raw_seed = env.get("TEST_RANDOMIZE_ORDERING_SEED", "").strip()
  if not raw_seed:
    raw_seed = env.get("TORCH_TPU_BENCHMARK_ORDER_SEED", "").strip()
  if not raw_seed and flag_seed:
    raw_seed = flag_seed
  if not raw_seed or raw_seed == "0":
    if "TEST_RANDOMIZE_ORDERING_SEED" in env and raw_seed == "0":
      env["TEST_RANDOMIZE_ORDERING_SEED"] = "0"
    return 0
  if raw_seed == "random":
    return 0
  normalized = seed_to_int(raw_seed)
  env["TEST_RANDOMIZE_ORDERING_SEED"] = str(normalized)
  return normalized


class BenchmarkTestLoader(absltest.TestLoader):
  """TestLoader with deterministic hash ordering and stable name-hash sharding."""

  def __init__(self, *args: Any, **kwargs: Any):
    self._seed_initialized: bool = False
    self._randomize_ordering_seed: int | None = None
    self._random: random.Random | None = None
    normalize_randomize_ordering_seed()
    super().__init__(*args, **kwargs)
    self._ensure_randomize_ordering_seed()

  def _ensure_randomize_ordering_seed(self) -> None:
    """Lazily resolves and normalizes randomize ordering seed after flag parsing."""
    if self._seed_initialized and (
        not flags.FLAGS.is_parsed()
        or not flags.FLAGS["test_randomize_ordering_seed"].present
    ):
      return
    raw_test_seed = os.environ.get(  # PY_ENVIRON_OK=benchmark
        "TEST_RANDOMIZE_ORDERING_SEED", ""
    ).strip()
    raw_bench_seed = os.environ.get(  # PY_ENVIRON_OK=benchmark
        "TORCH_TPU_BENCHMARK_ORDER_SEED", ""
    ).strip()
    flag_seed = ""
    if (
        flags.FLAGS.is_parsed()
        and "test_randomize_ordering_seed" in flags.FLAGS
        and flags.FLAGS["test_randomize_ordering_seed"].present
    ):
      flag_seed = str(flags.FLAGS.test_randomize_ordering_seed).strip()
    normalized = normalize_randomize_ordering_seed()
    if raw_test_seed == "0" or (
        flag_seed == "0" and not raw_test_seed and not raw_bench_seed
    ):
      self._randomize_ordering_seed = None
      self._random = None
    elif normalized > 0 and (raw_test_seed or raw_bench_seed or flag_seed):
      self._randomize_ordering_seed = normalized
      self._random = random.Random(normalized)
    else:
      self._randomize_ordering_seed = None
      self._random = None
    if flags.FLAGS.is_parsed():
      self._seed_initialized = True

  def getTestCaseNames(  # pylint: disable=invalid-name
      self,
      testCaseClass: type[unittest.TestCase],  # UNITTEST_OK=type annotation
  ) -> list[str]:
    """Returns test case names ordered by `case_params` before optional shuffle."""
    self._ensure_randomize_ordering_seed()
    names = list(super().getTestCaseNames(testCaseClass))
    class_case_params: Mapping[str, CaseParams] = getattr(
        testCaseClass, "case_params", {}
    )
    if class_case_params:
      if self._randomize_ordering_seed is None:
        case_rank = {
            name: idx for idx, name in enumerate(class_case_params.keys())
        }
        names.sort(
            key=lambda name: (1, case_rank[name])
            if name in case_rank
            else (0, name)
        )
      else:
        names.sort(key=lambda name: 1 if name in class_case_params else 0)
    return names

  def shardTestCaseNames(  # pylint: disable=invalid-name
      self,
      iterator: Iterator[int],
      ordered_names: Sequence[str],
      shard_index: int,
  ) -> Sequence[str]:
    """Partitions `ordered_names` across shards by a stable hash of each name.

    Auto-bisect reruns a single shard (e.g.
    `torch_benchmarks_shard_05_of_16_tpu-v7`) at each candidate commit, so a
    case must stay in the same shard across commits. A case's shard therefore
    depends only on its own name and the shard count, not on the ordering seed
    (the commit SHA in CI), estimated costs, known failures, the order of the
    names, or which other cases exist.

    Args:
      iterator: absltest's cyclic iterator over the shard indices.
      ordered_names: Test case names, in the order they should run.
      shard_index: The index of the current shard.

    Returns:
      The names in `ordered_names` assigned to `shard_index`, in order.
    """
    if not ordered_names:
      return []

    # TODO: b/578163611 - Balance shards by runtime without breaking bisect
    # stability, either with a checked-in shard map (falling back to the name
    # hash) or, once MLCompass can bisect a single test, by rebalancing on
    # every run. cl/998158472 has the previous cost-balanced loader.

    # absltest shares one `itertools.cycle(range(total_shards))` across all
    # test classes, so it may be mid-cycle; read one full cycle and sort it.
    seen_buckets: set[int] = set()
    for bucket in iterator:
      if bucket in seen_buckets:
        break
      seen_buckets.add(bucket)
    if shard_index not in seen_buckets:
      return []
    buckets = sorted(seen_buckets)

    return [
        name
        for name in ordered_names
        if buckets[_stable_shard_hash(name) % len(buckets)] == shard_index
    ]


def get_cases(
    platform: target_lib.Platform,
    framework: mode_lib.Framework,
    *,
    order_seed: str | None = None,
) -> Iterator[tuple[str, registry_lib.BenchmarkSpec, common.RunMode]]:
  """One case per (benchmark, applicable mode).

  The mode matrix is resolved here, at collection, so generated cases match the
  cell exactly: a CPU run never emits an eager_optimized case just to skip it.
  Ordering uses a deterministic hash (seeded by TORCH_TPU_BENCHMARK_ORDER_SEED
  when set) so all shards agree on the partition without alphabetical tail bias.

  Likewise, in presubmit scope (`--run_scope=presubmit`), benchmarks whose
  `PresubmitPolicy` keeps them out of this run emit no cases at all (see
  `get_presubmit_skip_reason`), rather than cases that `get_skip_reason` skips.
  Otherwise every presubmit-scope run would list hundreds of generated
  benchmarks, each in every run mode, as skipped cases. Collection usually runs
  at import, before flag parsing, so this reads `--run_scope` and
  `--fake_tensor_mode` with `flags_lib.collection_flag_values`, as
  `gen_benchmarks` does to register benchmarks. The benchmarks left out, and
  why, are logged once per call.

  Args:
    platform: Target hardware platform on which the benchmarks will run.
    framework: Execution framework (e.g. TORCH or TORCHAX) filtering modes.
    order_seed: Seed string used for deterministic benchmark ordering; falls
      back to the TORCH_TPU_BENCHMARK_ORDER_SEED environment variable if None.

  Yields:
    Tuples of (case_name, spec, run_mode) in deterministic hashed order.
  """
  if order_seed is None:
    order_seed = _get_order_seed()
  run_scope, fake_tensor_mode = flags_lib.collection_flag_values()
  target_kind = target_lib.make_target(platform).device_kind
  ordered_names = sorted(
      registry_lib.REGISTRY,
      key=lambda name: _case_order_key(name, order_seed),
  )
  # The names of the benchmarks left out, by presubmit policy and reason.
  excluded: dict[tuple[str, str | None], list[str]] = collections.defaultdict(
      list
  )
  for name in ordered_names:
    spec = registry_lib.REGISTRY[name]
    if (
        run_scope is context_lib.RunScope.PRESUBMIT
        and get_presubmit_skip_reason(spec, fake_tensor_mode=fake_tensor_mode)
        is not None
    ):
      excluded[(spec.presubmit.name, spec.presubmit_reason)].append(spec.name)
      continue
    for mode in mode_lib.modes_for(framework, target_kind):
      yield f"{spec.name}_{mode.value}", spec, mode
  if excluded:
    logging.info(
        "Left %d benchmarks out of this presubmit-scope run"
        " (--fake_tensor_mode=%s) for their presubmit policy:\n%s",
        sum(len(names) for names in excluded.values()),
        fake_tensor_mode,
        "\n".join(
            f"  {policy} ({reason}): {', '.join(sorted(names))}"
            for (policy, reason), names in sorted(excluded.items())
        ),
    )


def get_presubmit_skip_reason(
    spec: registry_lib.BenchmarkSpec, *, fake_tensor_mode: bool
) -> str | None:
  """Returns why `spec` does not run in this presubmit-scope run, or None.

  Args:
    spec: Benchmark specification.
    fake_tensor_mode: Whether this run is the FakeTensorMode CPU smoke.
  """
  if spec.presubmit is registry_lib.PresubmitPolicy.SKIP:
    return (
        f"Benchmark {spec.name} does not run in presubmit scope:"
        f" {spec.presubmit_reason}"
    )
  if (
      spec.presubmit is registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY
      and not fake_tensor_mode
  ):
    return (
        f"Benchmark {spec.name} runs in presubmit scope only in the"
        " FakeTensorMode CPU smoke (--fake_tensor_mode):"
        f" {spec.presubmit_reason}"
    )
  return None


def get_skip_reason(
    spec: registry_lib.BenchmarkSpec,
    mode: common.RunMode,
    *,
    framework: mode_lib.Framework,
    device_kind: target_lib.DeviceKind,
) -> str | None:
  """Returns why a case should be skipped, or None if it should run.

  A case is skipped if it matches an entry in `known_failures.KNOWN_FAILURES`,
  if `spec.skipped_run_modes` lists `mode`, or if, in presubmit scope, the
  benchmark's `PresubmitPolicy` keeps it out of this run.
  """
  failure_reason = known_failures.get_known_failure_reason(
      spec.name, framework=framework, device_kind=device_kind, run_mode=mode
  )
  if failure_reason is not None:
    return f"Benchmark {spec.name} is known to fail: {failure_reason}"
  if mode.value in spec.skipped_run_modes:
    return f"Benchmark {spec.name} explicitly skips run mode {mode.value}"
  run_scope, fake_tensor_mode = flags_lib.collection_flag_values()
  if run_scope is context_lib.RunScope.PRESUBMIT:
    return get_presubmit_skip_reason(spec, fake_tensor_mode=fake_tensor_mode)
  return None
