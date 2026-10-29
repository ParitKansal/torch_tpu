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

"""Common command-line flags for benchmark tests."""

import copy
import sys

from absl import flags
from perflab import context as context_lib
from perflab import target as target_lib

SKIP_BEHAVIOR = flags.DEFINE_enum(  # PYTHON_FLAG_OK=benchmark
    "skip_behavior",
    "skip",
    ["skip", "assert_raise", "ignore", "run_skipped"],
    "Behavior for run modes that are marked as skipped in the config."
    " Options:\n  skip: Skips the test if the benchmark is marked as skipped.\n"
    "  assert_raise: Runs the test and expects it to raise an exception if"
    " skipped.\n  ignore: Runs the test normally even if it is marked as"
    " skipped.\n  run_skipped: Only runs skipped tests (skips those not marked"
    " as skipped).",
)

ENABLE_XPROF = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "enable_xprof",
    False,
    "Whether to enable xprof profiling.",
    allow_override=True,
)

DRY_RUN = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "dry_run",
    False,
    "Whether to run in dry-run mode. If True, actual benchmark execution is"
    " skipped, and the list of targets that would have run are logged and"
    " written to the test outputs directory specified by environment variable"
    " TEST_UNDECLARED_OUTPUTS_DIR.",
)


RUN_IN_SUBPROCESS = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "run_in_subprocess",
    True,
    "Whether to run each benchmark case in its own child process. A case that"
    " crashes the interpreter (a CUDA device-side assert, a SIGSEGV in a"
    " kernel) or poisons the device context then fails alone instead of"
    " taking every later case in the shard with it. Disable to run cases"
    " in-process, e.g. to attach a debugger or use --pdb_post_mortem.",
)

FAKE_TENSOR_MODE = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "fake_tensor_mode",
    False,
    "Whether to construct models and execute benchmark steps inside PyTorch's"
    " FakeTensorMode(allow_non_fake_inputs=True). Only valid with"
    " BENCHMARK_PLATFORM=cpu, so it cannot skew real measurements; flag parsing"
    " fails otherwise. FakeTensorMode validates PyTorch op dispatch, tensor"
    " shapes, dtypes, and autograd graphs without allocating storage or"
    " executing CPU compute kernels, so it can also build models at their"
    " --run_scope=full sizes. Model configs are then read without the local"
    " weights cache directory. Building each case (not its steps) must also"
    " fit in the HBM of a v7 TensorCore; see perflab/fake_memory.py.",
)

COUNT_FLOPS = flags.DEFINE_bool(  # PYTHON_FLAG_OK=benchmark
    "count_flops",
    False,
    "Whether to count the model FLOPs of one step of each benchmark after it is"
    " measured, and export them with the achieved TFLOP/s and model FLOPs"
    " utilization (MFU). The FLOPs are counted on a second copy of the"
    " benchmark built for CPU under FakeTensorMode, so counting neither touches"
    " the measured device nor changes the measured times; see"
    " perflab/flops.py.",
)


def _check_fake_tensor_mode(fake_tensor_mode: bool) -> bool:
  """Checks that `--fake_tensor_mode` is only set in CPU runs."""
  if not fake_tensor_mode:
    return True
  platform = target_lib.platform_from_env()
  if platform is not target_lib.Platform.CPU:
    raise flags.ValidationError(
        f"--{FAKE_TENSOR_MODE.name} is only valid with"
        f" {target_lib.PLATFORM_ENV_VAR}=cpu; got"
        f" {target_lib.PLATFORM_ENV_VAR}={platform.value}"
    )
  return True


# Registered with the flag so that every binary that accepts it fails at flag
# parsing, before any benchmark is built or measured.
flags.register_validator(FAKE_TENSOR_MODE, _check_fake_tensor_mode)


def collection_flag_values() -> tuple[context_lib.RunScope, bool]:
  """Returns the values of `--run_scope` and `--fake_tensor_mode`.

  Benchmark binaries collect their test cases at import time, usually before
  `flags.FLAGS` is parsed. These two flags are then parsed from `sys.argv` by
  absl's parser, so `--`, `--noflag`, `=false` and `--flagfile` work as in the
  full parse, but into a scratch `FlagValues` that holds copies of just them.
  Parsing `flags.FLAGS` itself here would mark it parsed with only the flags
  defined so far, and the full parse in `app.run` would then append the values
  of multi-value flags a second time. An invalid `--fake_tensor_mode` still
  fails the full parse.
  """
  if flags.FLAGS.is_parsed():
    return context_lib.RUN_SCOPE.value, FAKE_TENSOR_MODE.value
  flag_values = flags.FlagValues()
  for flag in (context_lib.RUN_SCOPE, FAKE_TENSOR_MODE):
    flag_values[flag.name] = copy.deepcopy(flags.FLAGS[flag.name])
  flag_values(sys.argv, known_only=True)
  return (
      getattr(flag_values, context_lib.RUN_SCOPE.name),
      getattr(flag_values, FAKE_TENSOR_MODE.name),
  )
