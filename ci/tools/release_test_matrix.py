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

"""Builds the job matrix for release_wheel_full_matrix_tests.yml.

Each workflow input is a comma-separated list; an empty list means every
supported value. This validates the values against what the repository
supports (TPU generations known to CI, Python versions from
bazel/supported_python_versions.bzl, released PyTorch versions from
bazel/pytorch_versions.bzl) and prints the Cartesian product as the JSON
GitHub Actions expects for `strategy.matrix`.

Run from the repository root:

    $ python3 -m ci.tools.release_test_matrix \\
        --tpu_generations=v5,v7 --python_versions=3.14 \\
        --torch_versions=2.11.0,2.14.0 --dep_resolutions=oldest
"""

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import itertools
import json
import pathlib
import sys
from typing import Final

from bazel import pytorch_versions


@dataclasses.dataclass(frozen=True)
class TpuGeneration:
  name: str
  runner: str


# Runner labels match presubmit.yml and postsubmit.yml; the keys double as
# the suffix of the ci_tpu_v* and wheel_test_tpu_v* Bazel configs.
TPU_GENERATIONS: Final[Mapping[str, TpuGeneration]] = {
    "v5": TpuGeneration("TPU v5e", "linux-x86-ct5lp-224-8tpu"),
    "v6": TpuGeneration("TPU v6e", "linux-x86-ct6e-180-8tpu"),
    "v7": TpuGeneration("TPU v7x", "linux-x86-tpu7x-224-4tpu"),
}

# `uv pip compile --resolution` values, keyed by the workflow input names.
DEP_RESOLUTIONS: Final[Mapping[str, str]] = {
    "oldest": "lowest-direct",
    "newest": "highest",
}

_SUPPORTED_PYTHON_VERSIONS_BZL: Final[pathlib.Path] = pathlib.Path(
    pytorch_versions.__file__
).with_name("supported_python_versions.bzl")


def supported_python_versions() -> Sequence[str]:
  return pytorch_versions.read_value(
      "SUPPORTED_PYTHON_VERSIONS", _SUPPORTED_PYTHON_VERSIONS_BZL
  )


def released_torch_versions() -> Sequence[str]:
  """Every released PyTorch version the wheel supports, oldest first."""
  versions = {
      *pytorch_versions.read_value("WHEEL_TORCH_VERSIONS"),
      *pytorch_versions.read_value("INTERMEDIATE_PATCH_VERSIONS"),
  }
  return sorted(versions, key=lambda v: tuple(map(int, v.split("."))))


def parse_csv(raw: str, allowed: Sequence[str], flag: str) -> Sequence[str]:
  """Splits a comma-separated flag value and checks it against `allowed`.

  Args:
    raw: the flag value as given on the command line.
    allowed: the values the flag accepts.
    flag: the flag name, for error messages.

  Returns:
    The values in the order given, with surrounding whitespace removed, or
    every allowed value when `raw` is empty.

  Raises:
    ValueError: if a value is not in `allowed`.
  """
  values = [item.strip() for item in raw.split(",") if item.strip()]
  if not values:
    return list(allowed)
  if unknown := [v for v in values if v not in allowed]:
    raise ValueError(
        f"{flag} has unsupported values {unknown}; allowed: {list(allowed)}"
    )
  return values


def build_matrix(
    tpu_generations: Sequence[str],
    python_versions: Sequence[str],
    torch_versions: Sequence[str],
    dep_resolutions: Sequence[str],
) -> Mapping[str, Sequence[Mapping[str, str]]]:
  """Returns the `strategy.matrix` value with one `entry` per combination."""
  entries = []
  for tpu, python, torch, deps in itertools.product(
      tpu_generations, python_versions, torch_versions, dep_resolutions
  ):
    generation = TPU_GENERATIONS[tpu]
    entries.append({
        "tpu_generation": tpu,
        "tpu_name": generation.name,
        "runner": generation.runner,
        "python_version": python,
        "torch_version": torch,
        "deps": deps,
        "uv_resolution": DEP_RESOLUTIONS[deps],
    })
  return {"entry": entries}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  for flag in (
      "--tpu_generations",
      "--python_versions",
      "--torch_versions",
      "--dep_resolutions",
  ):
    parser.add_argument(flag, default="")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    matrix = build_matrix(
        parse_csv(
            args.tpu_generations, list(TPU_GENERATIONS), "--tpu_generations"
        ),
        parse_csv(
            args.python_versions,
            supported_python_versions(),
            "--python_versions",
        ),
        parse_csv(
            args.torch_versions, released_torch_versions(), "--torch_versions"
        ),
        parse_csv(
            args.dep_resolutions, list(DEP_RESOLUTIONS), "--dep_resolutions"
        ),
    )
  except ValueError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    return 1
  print(json.dumps(matrix, separators=(",", ":")))
  return 0


if __name__ == "__main__":
  sys.exit(main())
