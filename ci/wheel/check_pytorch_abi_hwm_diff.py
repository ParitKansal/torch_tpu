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

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Verifies that a PR does not move the PyTorch ABI symbol high-water mark.

To isolate symbol changes caused by the PR itself from any pre-existing or
upstream drift on the base branch, this script:
  1. Starts from a working copy of the PR's checked-in high-water mark file
     (e.g. `ci/wheel/pytorch_symbols_2_15_0.txt`).
  2. Resolves the diffbase SHA via `ci/tools/resolve_base_sha.sh`, builds the
     versioned glue libraries on the diffbase, and runs the HWM updater
     (`pytorch_abi_symbols.py update`) against the diffbase.
  3. Runs the HWM updater against the child wheel built at `HEAD_SHA`.
  4. Fails if the high-water mark moved between the diffbase run and the child
     run, printing the newly introduced symbols and prompting the author to run
     the HWM updater in their PR.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Final

if __package__ in (None, ""):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ci.wheel import pytorch_abi_symbols  # pylint: disable=g-import-not-at-top

_DEFAULT_PYTHON_VERSION: Final[str] = "3.14"
_DEFAULT_PYTORCH_VERSION: Final[str] = "2.15.0"
_DEFAULT_WHEEL_DIR: Final[str] = "dist"


@dataclasses.dataclass(frozen=True)
class HwmDiffOptions:
  """Configuration for checking PyTorch ABI high-water mark movement."""

  repo_root: pathlib.Path
  head_sha: str
  base_sha: str
  python_version: str
  pytorch_version: str
  wheel_dir: pathlib.Path
  symbol_file: pathlib.Path
  base_wheel_dir: pathlib.Path | None = None


def _resolve_repo_path(
    repo_root: pathlib.Path, raw_path: str | os.PathLike[str]
) -> pathlib.Path:
  path = pathlib.Path(raw_path)
  return path if path.is_absolute() else repo_root / path


def parse_args(argv: Sequence[str] | None = None) -> HwmDiffOptions:
  """Parses CLI arguments and environment variable defaults."""
  default_repo_root = pathlib.Path(__file__).resolve().parents[2]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--repo-root",
      type=pathlib.Path,
      default=pathlib.Path(
          os.environ.get("GITHUB_WORKSPACE", str(default_repo_root))
      ),
      help="Path to the repository root directory.",
  )
  parser.add_argument(
      "--head-sha",
      default=os.environ.get("HEAD_SHA", "HEAD"),
      help="Target commit SHA being evaluated (default: $HEAD_SHA or HEAD).",
  )
  parser.add_argument(
      "--base-sha",
      default=os.environ.get("BASE_SHA", ""),
      help="Optional override base commit SHA for the PR diff.",
  )
  parser.add_argument(
      "--python-version",
      default=os.environ.get("PYTHON_VERSION", _DEFAULT_PYTHON_VERSION),
      help=f"Python version used to build the wheel (default: {_DEFAULT_PYTHON_VERSION}).",
  )
  parser.add_argument(
      "--pytorch-version",
      default=os.environ.get("PYTORCH_VERSION", _DEFAULT_PYTORCH_VERSION),
      help=f"PyTorch ABI version to check (default: {_DEFAULT_PYTORCH_VERSION}).",
  )
  parser.add_argument(
      "--wheel-dir",
      default=os.environ.get("WHEEL_DIR", _DEFAULT_WHEEL_DIR),
      help="Directory containing the built child torch_tpu-*.whl (default: dist).",
  )
  parser.add_argument(
      "--symbol-file",
      default=os.environ.get("SYMBOL_FILE", ""),
      help=(
          "Path to the checked-in high-water mark symbol file (default:"
          " ci/wheel/pytorch_symbols_<version>.txt)."
      ),
  )
  parser.add_argument(
      "--base-wheel-dir",
      default=os.environ.get("BASE_WHEEL_DIR", ""),
      help=(
          "Optional directory of pre-built diffbase glue .so files (used by"
          " tests to avoid invoking Bazel)."
      ),
  )
  ns = parser.parse_args(argv)
  repo_root = ns.repo_root.resolve()
  version_suffix = ns.pytorch_version.strip().replace(".", "_")
  raw_symbol_file = (
      ns.symbol_file or f"ci/wheel/pytorch_symbols_{version_suffix}.txt"
  )
  base_wheel_dir = (
      _resolve_repo_path(repo_root, ns.base_wheel_dir)
      if ns.base_wheel_dir
      else None
  )
  return HwmDiffOptions(
      repo_root=repo_root,
      head_sha=ns.head_sha or "HEAD",
      base_sha=ns.base_sha,
      python_version=ns.python_version,
      pytorch_version=ns.pytorch_version,
      wheel_dir=_resolve_repo_path(repo_root, ns.wheel_dir),
      symbol_file=_resolve_repo_path(repo_root, raw_symbol_file),
      base_wheel_dir=base_wheel_dir,
  )


def _find_child_wheel(wheel_dir: pathlib.Path) -> pathlib.Path:
  """Locates the single `torch_tpu-*.whl` file in `wheel_dir`."""
  wheels = tuple(sorted(wheel_dir.glob("torch_tpu-*.whl")))
  if len(wheels) != 1:
    raise FileNotFoundError(
        f"Expected exactly one torch_tpu-*.whl in '{wheel_dir}', found"
        f" {len(wheels)}: {[str(w) for w in wheels]}"
    )
  return wheels[0]


def _git_rev_parse(repo_root: pathlib.Path, rev: str) -> str:
  """Resolves `rev` to a full commit SHA in `repo_root`."""
  return subprocess.run(
      ["git", "rev-parse", rev],
      cwd=repo_root,
      capture_output=True,
      text=True,
      check=True,
  ).stdout.strip()


def _resolve_base_sha(
    repo_root: pathlib.Path, head_sha: str, base_sha: str
) -> str:
  """Resolves the effective diffbase SHA for `head_sha`.

  1. If `base_sha` is provided, fetches it (if available on `origin`) and
     returns `git merge-base <base_sha> <head_sha>` (or `base_sha` if
     `merge-base` fails).
  2. Otherwise, if `head_sha` is a merge commit (e.g. GitHub Actions synthetic
     merge commit `refs/pull/N/merge`), returns `head_sha^1`.
  3. Otherwise, returns an empty string (signals non-PR event).
  """
  if base_sha:
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", base_sha],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    res = subprocess.run(
        ["git", "merge-base", base_sha, head_sha],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
      return res.stdout.strip()
    return base_sha

  verify_merge = subprocess.run(
      ["git", "rev-parse", "--verify", f"{head_sha}^2"],
      cwd=repo_root,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
  )
  if verify_merge.returncode == 0:
    return _git_rev_parse(repo_root, f"{head_sha}^1")
  return ""


def _bazel_glue_flags(python_version: str) -> tuple[str, ...]:
  """Returns the Bazel flags for building the versioned wheel glue target."""
  hermetic_py_ver = python_version.removesuffix("t")
  freethreaded = "yes" if python_version.endswith("t") else "no"
  return (
      "-c",
      "opt",
      "--config=wheel_common",
      f"--repo_env=HERMETIC_PYTHON_VERSION={hermetic_py_ver}",
      f"--define=PYTHON_VERSION={python_version}",
      f"--@rules_python//python/config_settings:py_freethreaded={freethreaded}",
      "--color=yes",
      "--curses=no",
  )


def _build_diffbase_glues(
    repo_root: pathlib.Path,
    pytorch_version: str,
    python_version: str,
    dest_dir: pathlib.Path,
) -> None:
  """Builds the versioned glue `.so` files at the current commit into `dest_dir`."""
  version_suffix = pytorch_version.strip().replace(".", "_")
  glue_target = (
      f"@torch_tpu_glues//torch_tpu/csrc/common/glue_{version_suffix}:binaries"
  )
  bazel_flags = _bazel_glue_flags(python_version)

  subprocess.run(
      [
          "bazel",
          "build",
          *bazel_flags,
          "--remote_download_toplevel",
          glue_target,
      ],
      cwd=repo_root,
      check=True,
  )

  cquery_proc = subprocess.run(
      ["bazel", "cquery", *bazel_flags, "--output=files", glue_target],
      cwd=repo_root,
      capture_output=True,
      text=True,
      check=True,
  )

  dest_dir.mkdir(parents=True, exist_ok=True)
  for raw_line in cquery_proc.stdout.splitlines():
    rel_path = raw_line.strip()
    if not rel_path.endswith(".so"):
      continue
    so_path = repo_root / rel_path
    if so_path.is_file():
      shutil.copy2(so_path, dest_dir / so_path.name)


def run_hwm_diff_check(opts: HwmDiffOptions) -> int:
  """Runs the HWM updater on diffbase and child, failing if the HWM moved."""
  if not opts.symbol_file.is_file():
    print(
        f"::error::High-water mark symbol file '{opts.symbol_file}' does not"
        " exist.",
        file=sys.stderr,
    )
    return 1

  try:
    child_wheel = _find_child_wheel(opts.wheel_dir)
  except FileNotFoundError as exc:
    print(f"::error::{exc}", file=sys.stderr)
    return 1

  orig_sha = _git_rev_parse(opts.repo_root, "HEAD")
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = pathlib.Path(tmp_dir)
    hwm_path = tmp_path / "hwm.txt"
    hwm_after_base_path = tmp_path / "hwm_after_base.txt"
    child_wheel_copy = tmp_path / child_wheel.name

    shutil.copy2(opts.symbol_file, hwm_path)
    shutil.copy2(child_wheel, child_wheel_copy)

    diff_base = _resolve_base_sha(opts.repo_root, opts.head_sha, opts.base_sha)
    if diff_base:
      print(f"===> Running HWM updater on diffbase ({diff_base})...")
      if opts.base_wheel_dir is not None:
        base_glues_dir = opts.base_wheel_dir
      else:
        base_glues_dir = tmp_path / "base_glues"
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", "origin", diff_base],
            cwd=opts.repo_root,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
          subprocess.run(
              ["git", "checkout", "--detach", diff_base],
              cwd=opts.repo_root,
              check=True,
          )
          _build_diffbase_glues(
              opts.repo_root,
              pytorch_version=opts.pytorch_version,
              python_version=opts.python_version,
              dest_dir=base_glues_dir,
          )
        finally:
          subprocess.run(
              ["git", "checkout", "--detach", orig_sha],
              cwd=opts.repo_root,
              check=False,
          )

      rc = pytorch_abi_symbols.main([
          "update",
          "--wheel-dir",
          str(base_glues_dir),
          "--version",
          opts.pytorch_version,
          "--output",
          str(hwm_path),
      ])
      if rc != 0:
        return rc
    else:
      print(
          "===> No diffbase SHA resolved; comparing child wheel directly"
          f" against {opts.symbol_file}."
      )

    shutil.copy2(hwm_path, hwm_after_base_path)

    print(f"===> Running HWM updater on child ({orig_sha})...")
    rc = pytorch_abi_symbols.main([
        "update",
        "--wheel",
        str(child_wheel_copy),
        "--version",
        opts.pytorch_version,
        "--output",
        str(hwm_path),
    ])
    if rc != 0:
      return rc

    symbols_after_base = pytorch_abi_symbols.load_symbols(hwm_after_base_path)
    symbols_after_child = pytorch_abi_symbols.load_symbols(hwm_path)
    if symbols_after_base != symbols_after_child:
      rel_symbol_file = (
          opts.symbol_file.relative_to(opts.repo_root)
          if opts.symbol_file.is_relative_to(opts.repo_root)
          else opts.symbol_file
      )
      result = pytorch_abi_symbols.check_symbols_compatible(
          symbols_after_base,
          symbols_after_child,
      )
      print(
          "\n"
          + result.format_report(
              version=opts.pytorch_version,
              symbol_file=rel_symbol_file,
          ),
          file=sys.stderr,
      )
      print(
          f"\n::error::The PyTorch {opts.pytorch_version} ABI high-water mark"
          " moved between the diffbase and this PR because your changes"
          " modify the PyTorch ABI symbols that torch_tpu uses. Please run"
          " the HWM updater in your PR and commit the updated"
          f" {rel_symbol_file}: python3 ci/wheel/pytorch_abi_symbols.py update"
          f" --wheel <path/to/torch_tpu.whl> --version {opts.pytorch_version}"
          f" --output {rel_symbol_file}",
          file=sys.stderr,
      )
      return 1

  print("===> High-water mark did not move between diffbase and child.")
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  opts = parse_args(argv)
  return run_hwm_diff_check(opts)


if __name__ == "__main__":
  sys.stdout.reconfigure(line_buffering=True)
  sys.exit(main())
