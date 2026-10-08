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

"""Extracts and validates PyTorch ABI symbols for versioned wheel libraries.

The multi-ABI `torch_tpu` wheel bundles versioned C++ extension glue `.so`
files (`*_<version>.so`) and a per-version common library
(`libpywrap_<version>_common.so`) for each supported PyTorch release, without
bundling `libtorch` or `libc10`. At runtime, those versioned `.so` files resolve
their undefined PyTorch symbols (`c10::`, `at::`, `torch::`, `c10d::`, `THP*`,
etc.) against the user's installed PyTorch distribution.

The checked-in symbol file (`ci/wheel/pytorch_symbols_2_15_0.txt`) records the
high-water mark of all undefined dynamic PyTorch symbols that `torch_tpu` has
referenced across builds targeting PyTorch 2.15.0:
  * Updating the file (`update --wheel ... --output ...`) unions newly extracted
    symbols with the existing symbols in the file so previously used symbols are
    preserved.
  * Checking a `torch_tpu` wheel (`check --wheel ... <symbol_file>`) verifies
    that every undefined PyTorch symbol in the wheel's versioned glues is
    recorded in the high-water mark (`wheel_symbols <= high_water_mark`).
  * Checking a PyTorch distribution (`check` with `--pytorch-wheel`,
    `--pytorch-lib-dir`, or `--pull-latest-pytorch`) verifies that every symbol
    in the high-water mark remains defined and exported by PyTorch's shared
    libraries (`high_water_mark <= pytorch_defined_symbols`).
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence, Set
import dataclasses
import hashlib
import html.parser
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Final, Self
import urllib.parse
import urllib.request
import zipfile

_DEFAULT_PYTORCH_VERSION: Final[str] = "2.15.0"
_DEFAULT_PYTHON_ABI_TAG: Final[str] = "cp314"
_DEFAULT_PLATFORM_ARCH: Final[str] = "x86_64"

# PyTorch wheel index directories checked by `--pull-latest-pytorch`. The release
# index is checked first (for GA releases), falling back to the RC test index.
_DEFAULT_PYTORCH_INDEX_URLS: Final[Sequence[str]] = (
    "https://download.pytorch.org/whl/cpu/torch/",
    "https://download.pytorch.org/whl/test/cpu/torch/",
)

# Dynamic symbol table (`nm -D`) types that represent undefined references
# resolved from external shared libraries at runtime: strong undefined (`U`) and
# weak undefined (`w`).
_UNDEFINED_NM_TYPES: Final[Set[str]] = frozenset({"U", "w"})

# Matches Itanium C++ ABI mangled symbols (and C-linkage THP / ParameterClass
# symbols) owned by PyTorch's libraries (`libc10.so`, `libtorch_cpu.so`,
# `libtorch_python.so`).
#
# Specifically matches:
#   * Nested names (`_ZN...`, `_ZNK...`, vtables `_ZTVN...`, typeinfo
#     `_ZTIN...` / `_ZTSN...`, VTT `_ZTTN...`, guard variables `_ZGVN...`,
#     thread-local wrappers `_ZTWN...`, and covariant/non-virtual thunks
#     `_ZTh..._N...` / `_ZTv..._N...`) rooted in the top-level PyTorch
#     namespaces `c10`, `at`, `torch`, `c10d`, `caffe2`, or `libkineto`, or
#     `pybind11::detail::type_caster<at::...>`. Compiler-synthesized optional
#     `_ZTH` thread-local init hooks (emitted as weak references even when the
#     underlying `thread_local` variable uses constant initialization) are
#     intentionally excluded; the underlying `_ZN...` / `_ZTW...` symbols are
#     still matched.
#   * `libtorch_python` C/C++ symbols prefixed with `THP` (e.g. `THPVariable_*`)
#     and `ParameterClass`.
_MANGLED_PYTORCH_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(
    r"^("
    r"_Z(?:TV|TI|TS|TT|GV|TW|Th[^_]*_|Tv[^_]*_)?N[KRVrO]*"
    r"(?:3c10|2at|5torch|4c10d|6caffe2|9libkineto"
    r"|8pybind116detail11type_casterIN2at)"
    r"|_Z\d*THP"
    r"|THP"
    r"|ParameterClass$"
    r")"
)

_PEP440_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<micro>\d+)"
    r"(?:(?P<pre_type>a|b|rc)(?P<pre_num>\d+))?"
    r"(?:\.post(?P<post_num>\d+))?"
    r"(?:\.dev(?P<dev_num>\d+))?"
    r"(?:\+(?P<local>[A-Za-z0-9._]+))?$"
)

_PRE_RELEASE_ORDER: Final[Mapping[str, int]] = {
    "a": 1,
    "b": 2,
    "rc": 3,
}

_FILE_HEADER_TEMPLATE: Final[str] = """\
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
#
# High-water mark of undefined dynamic PyTorch symbols referenced by the
# versioned {version} glue libraries in the torch_tpu wheel.
#
# To update the high-water mark from a built torch_tpu wheel:
#   python3 ci/wheel/pytorch_abi_symbols.py update \\
#       --wheel path/to/torch_tpu-*.whl \\
#       --version {version} \\
#       --output {symbol_file}
#
# To verify that the latest PyTorch {version} wheel still exports all
# symbols in this high-water mark:
#   python3 ci/wheel/pytorch_abi_symbols.py check \\
#       --pull-latest-pytorch \\
#       --version {version} \\
#       {symbol_file}"""


def _normalize_version_suffix(version: str) -> str:
  """Normalizes a version string like '2.15.0' or '2_15_0' to '2_15_0'."""
  cleaned = version.strip().replace(".", "_")
  if not re.fullmatch(r"\d+_\d+_\d+", cleaned):
    raise ValueError(
        "Expected a 3-component version like '2.15.0' or '2_15_0', got"
        f" {version!r}."
    )
  return cleaned


def _normalize_dotted_version(version: str) -> str:
  """Normalizes a version string like '2.15.0' or '2_15_0' to '2.15.0'."""
  return _normalize_version_suffix(version).replace("_", ".")


def _is_versioned_so_for_version(
    path: str | os.PathLike[str], version: str
) -> bool:
  """Returns True if `path` is a glue or common `.so` for `version`."""
  suffix = _normalize_version_suffix(version)
  filename = pathlib.PurePosixPath(os.fspath(path)).name
  return bool(re.search(rf"_{re.escape(suffix)}(?:_common)?\.so$", filename))


def _is_pytorch_lib_so(path: str | os.PathLike[str]) -> bool:
  """Returns True if `path` is a shared library in `torch/lib/lib*.so*`."""
  posix_path = pathlib.PurePosixPath(os.fspath(path))
  name = posix_path.name
  if not (name.startswith("lib") and (".so" in name)):
    return False
  parts = posix_path.parts
  if len(parts) >= 2 and parts[-3:-1] == ("torch", "lib"):
    return True
  return len(parts) == 1


def _is_pytorch_symbol(mangled_name: str) -> bool:
  """Returns True if `mangled_name` is a PyTorch symbol from libtorch/libc10."""
  base_symbol = mangled_name.split("@", 1)[0]
  return bool(_MANGLED_PYTORCH_SYMBOL_RE.search(base_symbol))


def _parse_nm_undefined_pytorch_symbols(nm_output: str) -> Set[str]:
  """Parses `nm -D --undefined-only` output and returns PyTorch symbols."""
  symbols: set[str] = set()
  for line in nm_output.splitlines():
    parts = line.split()
    if len(parts) < 2:
      continue
    sym_type = parts[-2]
    sym_name = parts[-1].split("@", 1)[0]
    if sym_type in _UNDEFINED_NM_TYPES and _is_pytorch_symbol(sym_name):
      symbols.add(sym_name)
  return frozenset(symbols)


def _parse_nm_defined_symbols(nm_output: str) -> Set[str]:
  """Parses `nm -D --defined-only` output and returns defined symbol names."""
  symbols: set[str] = set()
  for line in nm_output.splitlines():
    parts = line.split()
    if len(parts) < 2:
      continue
    sym_type = parts[-2]
    sym_name = parts[-1].split("@", 1)[0]
    if sym_type not in _UNDEFINED_NM_TYPES and len(sym_type) == 1:
      symbols.add(sym_name)
  return frozenset(symbols)


def _extract_so_pytorch_symbols(
    so_paths: Sequence[pathlib.Path],
) -> Set[str]:
  """Extracts undefined dynamic PyTorch symbols from `.so` files via `nm -D`."""
  if not so_paths:
    raise ValueError("so_paths must not be empty.")
  proc = subprocess.run(
      ["nm", "-D", "--undefined-only", *(str(p) for p in so_paths)],
      capture_output=True,
      text=True,
      check=True,
  )
  return _parse_nm_undefined_pytorch_symbols(proc.stdout)


def _extract_so_defined_symbols(
    so_paths: Sequence[pathlib.Path],
) -> Set[str]:
  """Extracts defined dynamic symbols from `.so` files via `nm -D`."""
  if not so_paths:
    raise ValueError("so_paths must not be empty.")
  proc = subprocess.run(
      ["nm", "-D", "--defined-only", *(str(p) for p in so_paths)],
      capture_output=True,
      text=True,
      check=True,
  )
  return _parse_nm_defined_symbols(proc.stdout)


def extract_dir_pytorch_symbols(
    wheel_root: pathlib.Path,
    version: str,
) -> Set[str]:
  """Extracts undefined PyTorch symbols for `version` from an unpacked wheel."""
  so_paths = sorted(
      p
      for p in wheel_root.rglob("*.so")
      if p.is_file()
      and _is_versioned_so_for_version(p.relative_to(wheel_root), version)
  )
  if not so_paths:
    raise FileNotFoundError(
        f"No versioned .so files for PyTorch {version} found under"
        f" {wheel_root}."
    )
  return _extract_so_pytorch_symbols(so_paths)


def extract_wheel_pytorch_symbols(
    wheel_path: pathlib.Path,
    version: str,
) -> Set[str]:
  """Extracts undefined PyTorch symbols for `version` from a `.whl` archive."""
  with (
      zipfile.ZipFile(wheel_path) as zf,
      tempfile.TemporaryDirectory() as tmp_dir,
  ):
    matching_members = sorted(
        name
        for name in zf.namelist()
        if _is_versioned_so_for_version(name, version)
    )
    if not matching_members:
      raise FileNotFoundError(
          f"No versioned .so files for PyTorch {version} found in {wheel_path}."
      )
    tmp_path = pathlib.Path(tmp_dir)
    extracted_paths = [
        pathlib.Path(zf.extract(member, tmp_path))
        for member in matching_members
    ]
    return _extract_so_pytorch_symbols(extracted_paths)


def extract_pytorch_lib_dir_defined_symbols(
    lib_dir: pathlib.Path,
) -> Set[str]:
  """Extracts defined dynamic symbols from a PyTorch `torch/lib` directory."""
  search_root = (
      lib_dir / "torch" / "lib"
      if (lib_dir / "torch" / "lib").is_dir()
      else lib_dir
  )
  so_paths = sorted(
      p
      for p in search_root.iterdir()
      if p.is_file() and p.name.startswith("lib") and ".so" in p.name
  )
  if not so_paths:
    raise FileNotFoundError(
        f"No PyTorch shared libraries (lib*.so*) found under {lib_dir}."
    )
  return _extract_so_defined_symbols(so_paths)


def extract_pytorch_wheel_defined_symbols(
    pytorch_wheel_path: pathlib.Path,
) -> Set[str]:
  """Extracts defined dynamic symbols from a PyTorch (`torch-*.whl`) archive."""
  with (
      zipfile.ZipFile(pytorch_wheel_path) as zf,
      tempfile.TemporaryDirectory() as tmp_dir,
  ):
    matching_members = sorted(
        name for name in zf.namelist() if _is_pytorch_lib_so(name)
    )
    if not matching_members:
      raise FileNotFoundError(
          "No PyTorch shared libraries (torch/lib/lib*.so*) found in"
          f" {pytorch_wheel_path}."
      )
    tmp_path = pathlib.Path(tmp_dir)
    extracted_paths = [
        pathlib.Path(zf.extract(member, tmp_path))
        for member in matching_members
    ]
    return _extract_so_defined_symbols(extracted_paths)


class _SimpleIndexHTMLParser(html.parser.HTMLParser):
  """PEP 503 simple repository API HTML parser extracting `<a href>` links."""

  def __init__(self) -> None:
    super().__init__()
    self.hrefs: list[str] = []

  def handle_starttag(
      self, tag: str, attrs: list[tuple[str, str | None]]
  ) -> None:
    if tag != "a":
      return
    for attr_name, attr_value in attrs:
      if attr_name == "href" and attr_value:
        self.hrefs.append(attr_value)


@dataclasses.dataclass(frozen=True)
class _WheelFilename:
  """Parsed PEP 427 wheel filename (`{dist}-{ver}(-{build})?-{py}-{abi}-{plat}.whl`)."""

  distribution: str
  release_triple: tuple[int, int, int]
  version_sort_key: tuple[int, int, int, int, int, str]
  build_sort_key: tuple[int, str]
  python_tags: Set[str]
  abi_tags: Set[str]
  platform_tags: Set[str]

  @classmethod
  def parse(cls, filename: str) -> Self | None:
    """Parses `filename` according to PEP 427, PEP 425, and PEP 440."""
    if not filename.endswith(".whl"):
      return None
    parts = filename.removesuffix(".whl").split("-")
    if len(parts) == 5:
      dist_raw, ver_raw, py_raw, abi_raw, plat_raw = parts
      build_raw: str | None = None
    elif len(parts) == 6:
      dist_raw, ver_raw, build_raw, py_raw, abi_raw, plat_raw = parts
    else:
      return None

    # PEP 503 project name normalization.
    distribution = re.sub(r"[-_.]+", "-", dist_raw).lower()

    ver_match = _PEP440_VERSION_RE.match(ver_raw)
    if ver_match is None:
      return None
    release_triple = (
        int(ver_match.group("major")),
        int(ver_match.group("minor")),
        int(ver_match.group("micro")),
    )
    # PEP 440 ordering: dev < pre-release (a < b < rc) < final < post-release.
    dev_num = ver_match.group("dev_num")
    pre_type = ver_match.group("pre_type")
    pre_num = ver_match.group("pre_num")
    post_num = ver_match.group("post_num")

    dev_rank = int(dev_num) if dev_num is not None else sys.maxsize
    if pre_type is not None and pre_num is not None:
      pre_rank = _PRE_RELEASE_ORDER[pre_type]
      pre_num_val = int(pre_num)
    else:
      pre_rank = 4
      pre_num_val = 0
    post_val = int(post_num) if post_num is not None else -1
    local_val = ver_match.group("local") or ""

    build_sort_key = (0, "")
    if build_raw is not None:
      build_match = re.match(r"^(\d+)(.*)$", build_raw)
      if build_match is None:
        return None
      build_sort_key = (int(build_match.group(1)), build_match.group(2))

    return cls(
        distribution=distribution,
        release_triple=release_triple,
        version_sort_key=(
            pre_rank,
            pre_num_val,
            post_val,
            dev_rank,
            1 if local_val else 0,
            local_val,
        ),
        build_sort_key=build_sort_key,
        python_tags=frozenset(py_raw.split(".")),
        abi_tags=frozenset(abi_raw.split(".")),
        platform_tags=frozenset(plat_raw.split(".")),
    )

  def matches(
      self,
      release_triple: tuple[int, int, int],
      python_tag: str,
      platform_arch: str,
  ) -> bool:
    """Returns True if this wheel matches the requested target tags."""
    if self.distribution != "torch" or self.release_triple != release_triple:
      return False
    if python_tag not in self.python_tags or python_tag not in self.abi_tags:
      return False
    return any(
        tag == f"linux_{platform_arch}"
        or (
            tag.startswith(("manylinux", "linux_"))
            and tag.endswith(f"_{platform_arch}")
        )
        for tag in self.platform_tags
    )


def _find_pytorch_wheel_url_in_index_html(
    html_text: str,
    base_url: str,
    version: str,
    python_tag: str = _DEFAULT_PYTHON_ABI_TAG,
    platform_arch: str = _DEFAULT_PLATFORM_ARCH,
) -> str | None:
  """Finds the highest-version matching PyTorch wheel URL in a PEP 503 index."""
  dotted = _normalize_dotted_version(version)
  major_str, minor_str, micro_str = dotted.split(".")
  target_triple = (int(major_str), int(minor_str), int(micro_str))

  parser = _SimpleIndexHTMLParser()
  parser.feed(html_text)

  best_key: (
      tuple[
          tuple[int, int, int, int, int, str],
          tuple[int, str],
          tuple[str, ...],
      ]
      | None
  ) = None
  best_url: str | None = None

  for href in parser.hrefs:
    parsed_href = urllib.parse.urlsplit(href)
    filename = pathlib.PurePosixPath(
        urllib.parse.unquote(parsed_href.path)
    ).name
    wheel = _WheelFilename.parse(filename)
    if wheel is None or not wheel.matches(
        target_triple, python_tag=python_tag, platform_arch=platform_arch
    ):
      continue
    candidate_key = (
        wheel.version_sort_key,
        wheel.build_sort_key,
        tuple(sorted(wheel.platform_tags)),
    )
    if best_key is None or candidate_key >= best_key:
      best_key = candidate_key
      best_url = urllib.parse.urljoin(base_url, href)

  return best_url


def _validate_sha256(
    actual_sha256: str, fragment: str, download_url: str
) -> None:
  """Validates `actual_sha256` against a `#sha256=<hex>` PEP 503 fragment."""
  if not fragment.startswith("sha256="):
    raise ValueError(
        f"Wheel URL {download_url!r} in index is missing a '#sha256=...' digest"
        " fragment."
    )
  expected_sha256 = fragment.removeprefix("sha256=").strip().lower()
  if not expected_sha256:
    raise ValueError(
        f"Wheel URL {download_url!r} in index has an empty '#sha256=' fragment."
    )
  if actual_sha256.lower() != expected_sha256:
    raise ValueError(
        f"SHA-256 mismatch for downloaded wheel {download_url}: expected"
        f" {expected_sha256}, got {actual_sha256.lower()}."
    )


def download_latest_pytorch_wheel(
    dest_dir: pathlib.Path,
    version: str,
    python_tag: str = _DEFAULT_PYTHON_ABI_TAG,
    platform_arch: str = _DEFAULT_PLATFORM_ARCH,
    index_urls: Sequence[str] = _DEFAULT_PYTORCH_INDEX_URLS,
) -> pathlib.Path:
  """Downloads the latest matching PyTorch wheel from `index_urls`."""
  wheel_url: str | None = None
  for index_url in index_urls:
    normalized_index = index_url if index_url.endswith("/") else f"{index_url}/"
    req = urllib.request.Request(
        normalized_index, headers={"User-Agent": "pytorch_abi_symbols/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
      html_text = resp.read().decode("utf-8", errors="replace")
    wheel_url = _find_pytorch_wheel_url_in_index_html(
        html_text,
        normalized_index,
        version=version,
        python_tag=python_tag,
        platform_arch=platform_arch,
    )
    if wheel_url is not None:
      break

  if wheel_url is None:
    raise FileNotFoundError(
        f"Could not find a PyTorch {version} wheel for {python_tag} /"
        f" {platform_arch} in indexes: {list(index_urls)}"
    )

  parsed = urllib.parse.urlsplit(wheel_url)
  download_url = urllib.parse.urlunsplit(parsed._replace(fragment=""))

  filename = pathlib.PurePosixPath(urllib.parse.unquote(parsed.path)).name
  dest_dir.mkdir(parents=True, exist_ok=True)
  dest_path = dest_dir / filename

  req = urllib.request.Request(
      download_url, headers={"User-Agent": "pytorch_abi_symbols/1.0"}
  )
  hasher = hashlib.sha256()
  with (
      urllib.request.urlopen(req, timeout=120) as resp,
      dest_path.open("wb") as out_file,
  ):
    while chunk := resp.read(1024 * 1024):
      out_file.write(chunk)
      hasher.update(chunk)

  _validate_sha256(hasher.hexdigest(), parsed.fragment, download_url)
  return dest_path


def demangle_symbols(mangled_names: Sequence[str]) -> Mapping[str, str]:
  """Demangles `mangled_names` with `c++filt`."""
  if not mangled_names:
    return {}
  proc = subprocess.run(
      ["c++filt"],
      input="\n".join(mangled_names),
      capture_output=True,
      text=True,
      check=True,
  )
  demangled_lines = proc.stdout.splitlines()
  return dict(zip(mangled_names, demangled_lines, strict=True))


def load_symbols(path: pathlib.Path) -> Set[str]:
  """Loads a newline-delimited symbol file, ignoring blank and `#` lines."""
  symbols: set[str] = set()
  for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    symbols.add(line)
  return frozenset(symbols)


def _format_symbols(
    symbols: Iterable[str],
    *,
    version: str = _DEFAULT_PYTORCH_VERSION,
    symbol_file: str | os.PathLike[str] | None = None,
) -> str:
  """Formats `symbols` into a canonical sorted text file with license header."""
  dotted = _normalize_dotted_version(version)
  suffix = _normalize_version_suffix(version)
  target_file = (
      os.fspath(symbol_file)
      if symbol_file is not None
      else f"ci/wheel/pytorch_symbols_{suffix}.txt"
  )
  header = _FILE_HEADER_TEMPLATE.format(
      version=dotted,
      symbol_file=target_file,
  )
  lines = [header, *sorted(set(symbols))]
  return "\n".join(lines) + "\n"


def update_high_water_mark(
    output_path: pathlib.Path,
    new_symbols: Iterable[str],
    *,
    version: str = _DEFAULT_PYTORCH_VERSION,
    reset: bool = False,
) -> Set[str]:
  """Updates `output_path` with `new_symbols`, preserving existing ones.

  Unless `reset=True` is passed, any symbols already recorded in `output_path`
  are unioned with `new_symbols` so that `output_path` acts as a monotonic
  high-water mark of all PyTorch symbols `torch_tpu` has used.

  Args:
    output_path: Target symbol file path.
    new_symbols: Newly extracted undefined PyTorch symbols from a `torch_tpu`
      wheel.
    version: PyTorch version string for the file header.
    reset: If True, overwrite `output_path` with only `new_symbols` instead of
      unioning with existing symbols.

  Returns:
    The resulting set of symbols written to `output_path`.
  """
  merged: set[str] = set(new_symbols)
  if not reset and output_path.exists():
    merged |= load_symbols(output_path)
  result = frozenset(merged)
  output_path.write_text(
      _format_symbols(result, version=version),
      encoding="utf-8",
  )
  return result


@dataclasses.dataclass(frozen=True)
class SymbolCompatibilityResult:
  """Result of checking PyTorch ABI symbols against a high-water mark list.

  Attributes:
    unexpected_in_wheel: Undefined PyTorch symbols present in the `torch_tpu`
      wheel's versioned glues but absent from the high-water mark file (e.g. a
      PyTorch function's signature/mangling changed in a newer release
      candidate, or `torch_tpu` added a new PyTorch C++ call that has not yet
      been added to the high-water mark).
    missing_from_pytorch: Symbols present in the high-water mark file that are
      not defined and exported by PyTorch's shared libraries (`libc10.so`,
      `libtorch_cpu.so`, `libtorch_python.so`, etc.), indicating an ABI break in
      PyTorch.
    unused_in_wheel: Historical symbols present in the high-water mark file that
      are not referenced by the current `torch_tpu` wheel. Allowed under
      high-water mark semantics (`is_compatible` remains True).
  """

  unexpected_in_wheel: Set[str] = frozenset()
  missing_from_pytorch: Set[str] = frozenset()
  unused_in_wheel: Set[str] = frozenset()

  @property
  def is_compatible(self) -> bool:
    """Returns True if no unexpected or missing symbols were detected."""
    return not self.unexpected_in_wheel and not self.missing_from_pytorch

  def format_report(
      self,
      *,
      version: str | None = None,
      symbol_file: str | os.PathLike[str] | None = None,
  ) -> str:
    """Formats a human-readable diff report with demangled symbol names."""
    if self.is_compatible:
      return "PyTorch ABI symbols are compatible with the high-water mark."

    dotted_version = (
        _normalize_dotted_version(version)
        if version is not None
        else "<version>"
    )
    suffix_version = (
        _normalize_version_suffix(version)
        if version is not None
        else "<version>"
    )
    target_symbol_file = (
        os.fspath(symbol_file)
        if symbol_file is not None
        else f"ci/wheel/pytorch_symbols_{suffix_version}.txt"
    )

    all_diff = sorted({*self.unexpected_in_wheel, *self.missing_from_pytorch})
    demangled = demangle_symbols(all_diff)
    lines = [
        (
            "PyTorch ABI symbol incompatibility detected against the high-water"
            " mark."
        ),
    ]
    if self.unexpected_in_wheel:
      lines.append(
          "\nUnexpected symbols in torch_tpu wheel"
          f" ({len(self.unexpected_in_wheel)} present in wheel but not in"
          " high-water mark; indicates a new PyTorch C++ call in torch_tpu or"
          " a PyTorch ABI signature change):"
      )
      for sym in sorted(self.unexpected_in_wheel):
        lines.append(f"  + {sym}  [{demangled.get(sym, sym)}]")
      lines.append(
          "\nIf this is an intentional torch_tpu C++ change that uses"
          " additional PyTorch symbols, update the high-water mark file with:\n"
          "  python3 ci/wheel/pytorch_abi_symbols.py update"
          f" --wheel <path/to/torch_tpu.whl> --version {dotted_version}"
          f" --output {target_symbol_file}"
      )
    if self.missing_from_pytorch:
      lines.append(
          "\nHigh-water mark symbols missing from PyTorch shared libraries"
          f" ({len(self.missing_from_pytorch)} present in high-water mark but"
          " not defined in PyTorch libtorch/libc10; indicates an incompatible"
          " PyTorch ABI change):"
      )
      for sym in sorted(self.missing_from_pytorch):
        lines.append(f"  - {sym}  [{demangled.get(sym, sym)}]")
    return "\n".join(lines)


def check_symbols_compatible(
    high_water_mark_symbols: Set[str],
    wheel_symbols: Set[str] | None = None,
    *,
    pytorch_defined_symbols: Set[str] | None = None,
) -> SymbolCompatibilityResult:
  """Checks `wheel_symbols` and/or `pytorch_defined_symbols` vs high-water mark.

  Args:
    high_water_mark_symbols: The checked-in high-water mark symbol set.
    wheel_symbols: Undefined PyTorch symbols extracted from a `torch_tpu`
      wheel's versioned glues. When provided, verifies `wheel_symbols <=
      high_water_mark_symbols`.
    pytorch_defined_symbols: Defined dynamic symbols extracted from PyTorch's
      shared libraries (`torch/lib/lib*.so*`). When provided, verifies
      `high_water_mark_symbols <= pytorch_defined_symbols`.

  Returns:
    A `SymbolCompatibilityResult` describing any incompatibilities.
  """
  if wheel_symbols is None and pytorch_defined_symbols is None:
    raise ValueError(
        "At least one of wheel_symbols or pytorch_defined_symbols must be"
        " provided."
    )
  unexpected_in_wheel: Set[str] = frozenset()
  unused_in_wheel: Set[str] = frozenset()
  if wheel_symbols is not None:
    unexpected_in_wheel = frozenset(wheel_symbols - high_water_mark_symbols)
    unused_in_wheel = frozenset(high_water_mark_symbols - wheel_symbols)

  missing_from_pytorch: Set[str] = frozenset()
  if pytorch_defined_symbols is not None:
    missing_from_pytorch = frozenset(
        high_water_mark_symbols - pytorch_defined_symbols
    )

  return SymbolCompatibilityResult(
      unexpected_in_wheel=unexpected_in_wheel,
      missing_from_pytorch=missing_from_pytorch,
      unused_in_wheel=unused_in_wheel,
  )


def _make_wheel_source_parent(*, required: bool) -> argparse.ArgumentParser:
  """Creates a shared parent parser for `--wheel` / `--wheel-dir`."""
  parent = argparse.ArgumentParser(add_help=False)
  wheel_group = parent.add_mutually_exclusive_group(required=required)
  wheel_group.add_argument(
      "--wheel",
      type=pathlib.Path,
      help="Path to a built torch_tpu .whl file.",
  )
  wheel_group.add_argument(
      "--wheel-dir",
      type=pathlib.Path,
      help="Path to an unpacked torch_tpu wheel directory.",
  )
  return parent


def _extract_wheel_symbols_from_args(
    args: argparse.Namespace,
) -> Set[str] | None:
  """Extracts `torch_tpu` wheel symbols from `--wheel` or `--wheel-dir`."""
  if args.wheel is not None:
    return extract_wheel_pytorch_symbols(args.wheel, args.version)
  if args.wheel_dir is not None:
    return extract_dir_pytorch_symbols(args.wheel_dir, args.version)
  return None


def main(argv: Sequence[str] | None = None) -> int:
  version_parent = argparse.ArgumentParser(add_help=False)
  version_parent.add_argument(
      "--version",
      default=_DEFAULT_PYTORCH_VERSION,
      help=(
          "PyTorch version to inspect (e.g. '2.15.0' or '2_15_0')."
          f" Defaults to {_DEFAULT_PYTORCH_VERSION!r}."
      ),
  )
  wheel_required_parent = _make_wheel_source_parent(required=True)
  wheel_optional_parent = _make_wheel_source_parent(required=False)

  parser = argparse.ArgumentParser(
      description=(
          "Dump, update, or verify the high-water mark of PyTorch ABI symbols"
          " used by versioned torch_tpu wheel libraries."
      )
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  subparsers.add_parser(
      "dump",
      parents=[version_parent, wheel_required_parent],
      help=(
          "Extract undefined PyTorch symbols from a torch_tpu wheel and write"
          " them to stdout."
      ),
  )

  update_parser = subparsers.add_parser(
      "update",
      parents=[version_parent, wheel_required_parent],
      help=(
          "Update a high-water mark symbol file from a torch_tpu wheel"
          " (unioning with existing symbols unless --reset is passed)."
      ),
  )
  update_parser.add_argument(
      "--output",
      type=pathlib.Path,
      required=True,
      help="Path to the high-water mark symbol file to update.",
  )
  update_parser.add_argument(
      "--reset",
      action="store_true",
      help=(
          "Overwrite the symbol file with only the given wheel's symbols"
          " instead of unioning with existing symbols."
      ),
  )

  check_parser = subparsers.add_parser(
      "check",
      parents=[version_parent, wheel_optional_parent],
      help="Check ABI compatibility against a high-water mark symbol file.",
  )
  pytorch_group = check_parser.add_mutually_exclusive_group()
  pytorch_group.add_argument(
      "--pytorch-wheel",
      type=pathlib.Path,
      help=(
          "Path to a PyTorch (torch-*.whl) wheel file to verify that all"
          " high-water mark symbols are defined in its shared libraries."
      ),
  )
  pytorch_group.add_argument(
      "--pytorch-lib-dir",
      type=pathlib.Path,
      help=(
          "Path to a PyTorch torch/lib directory (or unpacked torch wheel"
          " root) to verify that all high-water mark symbols are defined."
      ),
  )
  pytorch_group.add_argument(
      "--pull-latest-pytorch",
      action="store_true",
      help=(
          "Download the latest PyTorch wheel for --version from PyTorch's"
          " wheel index and verify that all high-water mark symbols are"
          " defined in it."
      ),
  )
  check_parser.add_argument(
      "--python-tag",
      default=_DEFAULT_PYTHON_ABI_TAG,
      help=(
          "Python ABI tag when downloading with --pull-latest-pytorch"
          f" (defaults to {_DEFAULT_PYTHON_ABI_TAG!r})."
      ),
  )
  check_parser.add_argument(
      "symbol_file",
      type=pathlib.Path,
      help="Path to the high-water mark symbol file to check against.",
  )

  args = parser.parse_args(argv)
  wheel_symbols = _extract_wheel_symbols_from_args(args)

  if args.command == "dump":
    assert wheel_symbols is not None
    sys.stdout.write(_format_symbols(wheel_symbols, version=args.version))
    return 0

  if args.command == "update":
    assert wheel_symbols is not None
    merged = update_high_water_mark(
        args.output,
        wheel_symbols,
        version=args.version,
        reset=args.reset,
    )
    print(
        f"Updated high-water mark in {args.output} ({len(merged)} total"
        f" PyTorch {args.version} ABI symbols; {len(wheel_symbols)} in current"
        " wheel)."
    )
    return 0

  pytorch_defined_symbols: Set[str] | None = None
  if args.pytorch_wheel is not None:
    pytorch_defined_symbols = extract_pytorch_wheel_defined_symbols(
        args.pytorch_wheel
    )
  elif args.pytorch_lib_dir is not None:
    pytorch_defined_symbols = extract_pytorch_lib_dir_defined_symbols(
        args.pytorch_lib_dir
    )
  elif args.pull_latest_pytorch:
    with tempfile.TemporaryDirectory() as tmp_dir:
      downloaded_whl = download_latest_pytorch_wheel(
          pathlib.Path(tmp_dir),
          version=args.version,
          python_tag=args.python_tag,
      )
      print(f"Downloaded {downloaded_whl.name}.")
      pytorch_defined_symbols = extract_pytorch_wheel_defined_symbols(
          downloaded_whl
      )

  if wheel_symbols is None and pytorch_defined_symbols is None:
    check_parser.error(
        "At least one source (--wheel, --wheel-dir, --pytorch-wheel,"
        " --pytorch-lib-dir, or --pull-latest-pytorch) is required."
    )

  high_water_mark = load_symbols(args.symbol_file)
  result = check_symbols_compatible(
      high_water_mark,
      wheel_symbols,
      pytorch_defined_symbols=pytorch_defined_symbols,
  )
  if not result.is_compatible:
    print(
        result.format_report(
            version=args.version,
            symbol_file=args.symbol_file,
        ),
        file=sys.stderr,
    )
    return 1
  if wheel_symbols is not None:
    print(
        f"Verified {len(wheel_symbols)} torch_tpu wheel symbols are within"
        f" the {len(high_water_mark)}-symbol high-water mark in"
        f" {args.symbol_file}."
    )
  if pytorch_defined_symbols is not None:
    print(
        f"Verified all {len(high_water_mark)} high-water mark symbols in"
        f" {args.symbol_file} are defined in PyTorch {args.version}"
        f" ({len(pytorch_defined_symbols)} exported symbols)."
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
