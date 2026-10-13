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

"""Unit tests for ci/wheel/pytorch_abi_symbols.py."""

from collections.abc import Iterator, Sequence
import contextlib
import functools
import hashlib
import http.server
import io
import os
import pathlib
import subprocess
import tempfile
import threading
import zipfile

from absl.testing import absltest
from ci.wheel import check_pytorch_abi_hwm_diff
from ci.wheel import pytorch_abi_symbols


def _compile_stub_so(
    out_path: pathlib.Path,
    undefined_symbols: Sequence[str] = (),
    defined_symbols: Sequence[str] = (),
) -> None:
  """Compiles a minimal real ELF shared library with the requested symbols."""
  lines = [f"extern void {sym}(void);" for sym in undefined_symbols]
  for sym in defined_symbols:
    lines.append(f"void {sym}(void) {{}}")
  lines.append("void _stub_entry(void) {")
  for sym in undefined_symbols:
    lines.append(f"  {sym}();")
  lines.append("}")

  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".c", encoding="utf-8", delete=True
  ) as src_file:
    src_file.write("\n".join(lines) + "\n")
    src_file.flush()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cc", "-shared", "-fPIC", src_file.name, "-o", str(out_path)],
        check=True,
    )


class _QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
  """SimpleHTTPRequestHandler that suppresses stderr request logs in tests."""

  def log_message(self, format: str, *args: object) -> None:  # pylint: disable=redefined-builtin
    del format, args


@contextlib.contextmanager
def _serve_directory(directory: pathlib.Path) -> Iterator[str]:
  """Serves `directory` over a local loopback HTTP server and yields its URL."""
  handler = functools.partial(
      _QuietHTTPRequestHandler, directory=str(directory)
  )
  with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
      host, port = server.server_address[:2]
      yield f"http://{host}:{port}/"
    finally:
      server.shutdown()
      thread.join()


class PytorchAbiSymbolsTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.work_dir = pathlib.Path(self.create_tempdir().full_path)

  def test_normalize_version_suffix_and_dotted(self) -> None:
    cases = [
        ("2.15.0", "2_15_0", "2.15.0"),
        (" 2_15_0 ", "2_15_0", "2.15.0"),
        ("2.16.1", "2_16_1", "2.16.1"),
    ]
    for raw, expected_suffix, expected_dotted in cases:
      with self.subTest(raw=raw):
        self.assertEqual(
            pytorch_abi_symbols._normalize_version_suffix(raw),
            expected_suffix,
        )
        self.assertEqual(
            pytorch_abi_symbols._normalize_dotted_version(raw),
            expected_dotted,
        )

    for invalid in ("2.15", "2.15.0.1", "abc"):
      with self.subTest(invalid=invalid):
        with self.assertRaises(ValueError):
          pytorch_abi_symbols._normalize_version_suffix(invalid)

  def test_is_versioned_so_for_version(self) -> None:
    cases: Sequence[tuple[str | pathlib.PurePosixPath, str, bool]] = [
        ("torch_tpu/_internal/env_2_15_0.so", "2.15.0", True),
        (
            pathlib.PurePosixPath(
                "torch_tpu/csrc/common/glue_2_15_0/libpywrap_2_15_0_common.so"
            ),
            "2.15.0",
            True,
        ),
        ("torch_tpu/_internal/env_2_14_0.so", "2.15.0", False),
        ("torch_tpu/csrc/common/libxla_base.so", "2.15.0", False),
    ]
    for path, version, expected in cases:
      with self.subTest(path=path, version=version):
        self.assertEqual(
            pytorch_abi_symbols._is_versioned_so_for_version(path, version),
            expected,
        )

  def test_is_pytorch_lib_so(self) -> None:
    cases: Sequence[tuple[str | pathlib.PurePosixPath, bool]] = [
        ("torch/lib/libc10.so", True),
        (pathlib.PurePosixPath("torch/lib/libtorch_cpu.so"), True),
        ("libtorch_python.so", True),
        ("torch/_C.cpython-314.so", False),
        ("torch/lib/libtorch.a", False),
    ]
    for path, expected in cases:
      with self.subTest(path=path):
        self.assertEqual(
            pytorch_abi_symbols._is_pytorch_lib_so(path),
            expected,
        )

  def test_is_pytorch_symbol_classifies_pytorch_and_non_pytorch_symbols(
      self,
  ) -> None:
    pytorch_syms = [
        "_ZN3c105ErrorC1ENS_14SourceLocationE",
        "_ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEE",
        "_ZN5torch8autograd13_wrap_outputsEv",
        "_ZN4c10d9FileStoreC1Ev",
        "_ZTVN3c1012AutogradMetaE",
        "_ZTIN5torch8autograd4NodeE",
        "_ZTWN3c104impl26raw_local_dispatch_key_setE",
        "_ZN3c104impl26raw_local_dispatch_key_setE",
        "_ZThn16_N3c1010TensorImplD1Ev",
        "_ZN8pybind116detail11type_casterIN2at6ScalarEvE4loadENS_6handleEb",
        "_Z16THPVariable_WrapN2at10TensorBaseE",
        "THPVariableClass",
        "ParameterClass",
    ]
    for sym in pytorch_syms:
      with self.subTest(sym=sym):
        self.assertTrue(pytorch_abi_symbols._is_pytorch_symbol(sym))

    non_pytorch_syms = [
        "PyObject_GetAttrString",
        "_Py_Dealloc",
        "_ZN3xla9ShapeUtil8MakeShapeEv",
        "_ZN9torch_tpu19GetBaseTensorBufferERKN2at6TensorE",
        "_ZN4absl12lts_202501276StatusC1Ev",
        "_ZSt20__throw_length_errorPKc",
        "memcpy@GLIBC_2.14",
        # Optional compiler-synthesized TLS init hook (_ZTH) is excluded.
        "_ZTHN3c104impl26raw_local_dispatch_key_setE",
    ]
    for sym in non_pytorch_syms:
      with self.subTest(sym=sym):
        self.assertFalse(pytorch_abi_symbols._is_pytorch_symbol(sym))

  def test_parse_nm_undefined_and_defined_symbols(self) -> None:
    nm_undefined = "\n".join([
        "glue_2_15_0.so:",
        "                 U _ZN3c105ErrorC1Ev",
        "                 w _ZTVN3c1012AutogradMetaE",
        "                 U PyObject_GetAttrString",
        "                 U _ZN3xla9ShapeUtil4MakeEv",
        "                 U _ZN9torch_tpu19GetBaseTensorBufferERKN2at6TensorE",
        "0000000000001000 T _ZN3c1014DefinedLocallyEv",
    ])
    self.assertEqual(
        pytorch_abi_symbols._parse_nm_undefined_pytorch_symbols(nm_undefined),
        frozenset({"_ZN3c105ErrorC1Ev", "_ZTVN3c1012AutogradMetaE"}),
    )

    nm_defined = "\n".join([
        "libc10.so:",
        "0000000000001000 T _ZN3c105ErrorC1Ev",
        "0000000000002000 V _ZTVN3c1012AutogradMetaE",
        "0000000000003000 B _ZN3c104impl26raw_local_dispatch_key_setE",
        "                 U memcpy@GLIBC_2.14",
    ])
    self.assertEqual(
        pytorch_abi_symbols._parse_nm_defined_symbols(nm_defined),
        frozenset({
            "_ZN3c105ErrorC1Ev",
            "_ZTVN3c1012AutogradMetaE",
            "_ZN3c104impl26raw_local_dispatch_key_setE",
        }),
    )

  def test_extract_dir_and_wheel_pytorch_symbols(self) -> None:
    wheel_root = self.work_dir / "unpacked"
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_15_0.so",
        undefined_symbols=[
            "_ZN3c105ErrorC1Ev",
            "PyObject_GetAttrString",
            "_ZN3xla9ShapeUtil4MakeEv",
        ],
    )
    _compile_stub_so(
        wheel_root
        / "torch_tpu/csrc/common/glue_2_15_0/libpywrap_2_15_0_common.so",
        undefined_symbols=[
            "_ZN2at4_ops5zeros4callEv",
            "THPVariableClass",
        ],
        defined_symbols=[
            "_ZN9torch_tpu19GetBaseTensorBufferEv",
        ],
    )
    # A 2.14.0 glue library whose symbols must be ignored when querying 2.15.0.
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_14_0.so",
        undefined_symbols=["_ZN3c1015report_overflowEPKc"],
    )

    expected = frozenset({
        "_ZN3c105ErrorC1Ev",
        "_ZN2at4_ops5zeros4callEv",
        "THPVariableClass",
    })
    self.assertEqual(
        pytorch_abi_symbols.extract_dir_pytorch_symbols(wheel_root, "2.15.0"),
        expected,
    )

    wheel_path = self.work_dir / "torch_tpu-0.1.2-cp314-cp314-linux_x86_64.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
      for file_path in wheel_root.rglob("*"):
        if file_path.is_file():
          zf.write(file_path, file_path.relative_to(wheel_root).as_posix())

    self.assertEqual(
        pytorch_abi_symbols.extract_wheel_pytorch_symbols(wheel_path, "2.15.0"),
        expected,
    )

  def test_extract_pytorch_lib_dir_and_wheel_defined_symbols(self) -> None:
    pytorch_root = self.work_dir / "pytorch_pkg"
    _compile_stub_so(
        pytorch_root / "torch/lib/libc10.so",
        defined_symbols=["_ZN3c105ErrorC1Ev"],
    )
    _compile_stub_so(
        pytorch_root / "torch/lib/libtorch_cpu.so",
        defined_symbols=["_ZN2at4_ops5zeros4callEv"],
    )

    defined_from_dir = (
        pytorch_abi_symbols.extract_pytorch_lib_dir_defined_symbols(
            pytorch_root
        )
    )
    self.assertContainsSubset(
        {"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"},
        defined_from_dir,
    )

    pytorch_whl = self.work_dir / "torch-2.15.0+cpu-cp314-cp314-linux.whl"
    with zipfile.ZipFile(pytorch_whl, "w") as zf:
      for file_path in pytorch_root.rglob("*"):
        if file_path.is_file():
          zf.write(file_path, file_path.relative_to(pytorch_root).as_posix())

    defined_from_whl = (
        pytorch_abi_symbols.extract_pytorch_wheel_defined_symbols(pytorch_whl)
    )
    self.assertContainsSubset(
        {"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"},
        defined_from_whl,
    )

  def test_high_water_mark_semantics_and_report(self) -> None:
    high_water_mark = frozenset({
        "_ZN3c105ErrorC1Ev",
        "_ZN2at4_ops5zeros4callEv",
        "_ZN2at4_ops3sum4callEv",
    })
    with self.assertRaises(ValueError):
      pytorch_abi_symbols.check_symbols_compatible(high_water_mark)

    # A wheel using a strict subset of the high-water mark is compatible.
    subset_result = pytorch_abi_symbols.check_symbols_compatible(
        high_water_mark,
        frozenset({"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}),
        pytorch_defined_symbols=high_water_mark | {"_ZN3c105ExtraEv"},
    )
    self.assertTrue(subset_result.is_compatible)
    self.assertEmpty(subset_result.unexpected_in_wheel)
    self.assertEmpty(subset_result.missing_from_pytorch)
    self.assertEqual(
        subset_result.unused_in_wheel,
        frozenset({"_ZN2at4_ops3sum4callEv"}),
    )

    # A wheel referencing a symbol outside the high-water mark fails.
    unexpected_result = pytorch_abi_symbols.check_symbols_compatible(
        high_water_mark,
        frozenset({"_ZN3c105ErrorC1Ev", "_ZN5torch7Library4_defEv"}),
    )
    self.assertFalse(unexpected_result.is_compatible)
    self.assertEqual(
        unexpected_result.unexpected_in_wheel,
        frozenset({"_ZN5torch7Library4_defEv"}),
    )
    report = unexpected_result.format_report(
        version="2.16.0",
        symbol_file="ci/wheel/pytorch_symbols_2_16_0.txt",
    )
    self.assertIn("torch::Library::_def()", report)
    self.assertIn("--version 2.16.0", report)
    self.assertIn("--output ci/wheel/pytorch_symbols_2_16_0.txt", report)

    # A PyTorch wheel missing a high-water mark symbol (even one no longer used
    # by the current torch_tpu wheel) fails.
    missing_pytorch_result = pytorch_abi_symbols.check_symbols_compatible(
        high_water_mark,
        frozenset({"_ZN3c105ErrorC1Ev"}),
        pytorch_defined_symbols=frozenset(
            {"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}
        ),
    )
    self.assertFalse(missing_pytorch_result.is_compatible)
    self.assertEqual(
        missing_pytorch_result.missing_from_pytorch,
        frozenset({"_ZN2at4_ops3sum4callEv"}),
    )
    self.assertIn(
        "at::_ops::sum::call()", missing_pytorch_result.format_report()
    )

  def test_find_pytorch_wheel_url_in_index_html(self) -> None:
    # Links are intentionally out of chronological order (final release listed
    # before rc1, plus cp313 and cp314t free-threaded wheels) to verify PEP 440
    # and PEP 427 parsing rather than HTML link order.
    html_text = """
    <!DOCTYPE html>
    <html><body>
      <a href="/whl/test/cpu/torch-2.15.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=aaa">torch-313</a>
      <a href="/whl/test/cpu/torch-2.15.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=bbb">torch-314-final</a>
      <a href="/whl/test/cpu/torch-2.15.0rc2%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=ddd">torch-314-rc2</a>
      <a href="/whl/test/cpu/torch-2.15.0%2Bcpu-cp314-cp314t-manylinux_2_28_x86_64.whl#sha256=ccc">torch-314t</a>
    </body></html>
    """
    url = pytorch_abi_symbols._find_pytorch_wheel_url_in_index_html(
        html_text,
        "https://download.pytorch.org/whl/test/cpu/torch/",
        version="2.15.0",
        python_tag="cp314",
        platform_arch="x86_64",
    )
    self.assertEqual(
        url,
        "https://download.pytorch.org/whl/test/cpu/torch-2.15.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=bbb",
    )

  def test_download_latest_pytorch_wheel_and_sha256_validation(self) -> None:
    index_dir = self.work_dir / "simple_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    wheel_filename = "torch-2.15.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl"
    wheel_bytes = b"fake-wheel-payload"
    wheel_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
    (index_dir / wheel_filename).write_bytes(wheel_bytes)

    with _serve_directory(index_dir) as base_url:
      # 1. Valid SHA-256 succeeds.
      (index_dir / "index.html").write_text(
          f'<html><body><a href="{wheel_filename}#sha256={wheel_sha256}">'
          f"{wheel_filename}</a></body></html>",
          encoding="utf-8",
      )
      dest_dir = self.work_dir / "downloaded"
      downloaded = pytorch_abi_symbols.download_latest_pytorch_wheel(
          dest_dir,
          version="2.15.0",
          python_tag="cp314",
          platform_arch="x86_64",
          index_urls=[base_url],
      )
      self.assertEqual(downloaded.read_bytes(), wheel_bytes)

      # 2. Mismatched SHA-256 raises ValueError.
      (index_dir / "index.html").write_text(
          f'<html><body><a href="{wheel_filename}#sha256={"0" * 64}">'
          f"{wheel_filename}</a></body></html>",
          encoding="utf-8",
      )
      with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
        pytorch_abi_symbols.download_latest_pytorch_wheel(
            dest_dir,
            version="2.15.0",
            python_tag="cp314",
            platform_arch="x86_64",
            index_urls=[base_url],
        )

      # 3. Missing #sha256= fragment raises ValueError.
      (index_dir / "index.html").write_text(
          f'<html><body><a href="{wheel_filename}">'
          f"{wheel_filename}</a></body></html>",
          encoding="utf-8",
      )
      with self.assertRaisesRegex(ValueError, "missing a '#sha256=...'"):
        pytorch_abi_symbols.download_latest_pytorch_wheel(
            dest_dir,
            version="2.15.0",
            python_tag="cp314",
            platform_arch="x86_64",
            index_urls=[base_url],
        )

      # 4. No matching wheel in index raises FileNotFoundError.
      with self.assertRaises(FileNotFoundError):
        pytorch_abi_symbols.download_latest_pytorch_wheel(
            dest_dir,
            version="2.16.0",
            python_tag="cp314",
            platform_arch="x86_64",
            index_urls=[base_url],
        )

  def test_cli_high_water_mark_update_reset_and_check(self) -> None:
    wheel_root = self.work_dir / "wheel_dir"
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_15_0.so",
        undefined_symbols=["_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"],
    )
    symbols_file = self.work_dir / "symbols.txt"

    # Initial update populates the high-water mark with 2 symbols.
    self.assertEqual(
        pytorch_abi_symbols.main([
            "update",
            "--wheel-dir",
            str(wheel_root),
            "--version",
            "2.15.0",
            "--output",
            str(symbols_file),
        ]),
        0,
    )
    self.assertEqual(
        pytorch_abi_symbols.load_symbols(symbols_file),
        frozenset({"_ZN3c105ErrorC1Ev", "_ZN2at4_ops5zeros4callEv"}),
    )

    # Second build drops `_ZN2at4_ops5zeros4callEv` and adds
    # `_ZN2at4_ops3sum4callEv`. Updating without `--reset` keeps the union (3).
    _compile_stub_so(
        wheel_root / "torch_tpu/_internal/env_2_15_0.so",
        undefined_symbols=["_ZN3c105ErrorC1Ev", "_ZN2at4_ops3sum4callEv"],
    )
    self.assertEqual(
        pytorch_abi_symbols.main([
            "update",
            "--wheel-dir",
            str(wheel_root),
            "--version",
            "2.15.0",
            "--output",
            str(symbols_file),
        ]),
        0,
    )
    self.assertEqual(
        pytorch_abi_symbols.load_symbols(symbols_file),
        frozenset({
            "_ZN3c105ErrorC1Ev",
            "_ZN2at4_ops5zeros4callEv",
            "_ZN2at4_ops3sum4callEv",
        }),
    )

    # Checking the 2-symbol wheel against the 3-symbol high-water mark passes.
    self.assertEqual(
        pytorch_abi_symbols.main([
            "check",
            "--wheel-dir",
            str(wheel_root),
            "--version",
            "2.15.0",
            str(symbols_file),
        ]),
        0,
    )

    # Checking against a PyTorch lib dir that defines all 3 high-water mark
    # symbols passes.
    pytorch_lib_dir = self.work_dir / "torch_lib"
    _compile_stub_so(
        pytorch_lib_dir / "libtorch_cpu.so",
        defined_symbols=[
            "_ZN3c105ErrorC1Ev",
            "_ZN2at4_ops5zeros4callEv",
            "_ZN2at4_ops3sum4callEv",
        ],
    )
    self.assertEqual(
        pytorch_abi_symbols.main([
            "check",
            "--pytorch-lib-dir",
            str(pytorch_lib_dir),
            "--version",
            "2.15.0",
            str(symbols_file),
        ]),
        0,
    )

    # If PyTorch drops `_ZN2at4_ops5zeros4callEv` (a historical high-water mark
    # symbol no longer in the current torch_tpu wheel), `check` fails!
    _compile_stub_so(
        pytorch_lib_dir / "libtorch_cpu.so",
        defined_symbols=[
            "_ZN3c105ErrorC1Ev",
            "_ZN2at4_ops3sum4callEv",
        ],
    )
    self.assertEqual(
        pytorch_abi_symbols.main([
            "check",
            "--pytorch-lib-dir",
            str(pytorch_lib_dir),
            "--version",
            "2.15.0",
            str(symbols_file),
        ]),
        1,
    )

    # Updating with `--reset` overwrites the file with only the current wheel's
    # 2 symbols.
    self.assertEqual(
        pytorch_abi_symbols.main([
            "update",
            "--wheel-dir",
            str(wheel_root),
            "--version",
            "2.15.0",
            "--reset",
            "--output",
            str(symbols_file),
        ]),
        0,
    )
    self.assertEqual(
        pytorch_abi_symbols.load_symbols(symbols_file),
        frozenset({"_ZN3c105ErrorC1Ev", "_ZN2at4_ops3sum4callEv"}),
    )

  def test_check_pytorch_abi_hwm_diff_script(self) -> None:
    test_repo = self.work_dir / "git_repo"
    (test_repo / "ci" / "wheel").mkdir(parents=True, exist_ok=True)

    git_env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "LC_ALL": "C",
    }

    def run_git(*args: str) -> str:
      return subprocess.run(
          ["git", *args],
          cwd=test_repo,
          env=git_env,
          capture_output=True,
          text=True,
          check=True,
      ).stdout.strip()

    run_git("init", "-b", "release/0.2.0")
    run_git("config", "user.name", "Test User")
    run_git("config", "user.email", "test@example.com")
    run_git("config", "commit.gpgsign", "false")

    symbol_file = test_repo / "ci" / "wheel" / "pytorch_symbols_2_15_0.txt"
    pytorch_abi_symbols.update_high_water_mark(
        symbol_file,
        ["_ZN3c105ErrorC1Ev"],
        version="2.15.0",
        reset=True,
    )
    run_git("add", ".")
    run_git("commit", "-m", "Base commit")
    base_sha = run_git("rev-parse", "HEAD")

    (test_repo / "change.txt").write_text("pr change\n", encoding="utf-8")
    run_git("add", "change.txt")
    run_git("commit", "-m", "PR commit")
    head_sha = run_git("rev-parse", "HEAD")

    base_glues_dir = self.work_dir / "base_glues"
    child_unpacked_dir = self.work_dir / "child_unpacked"
    dist_dir = test_repo / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    child_whl = dist_dir / "torch_tpu-0.2.0-cp314-cp314-linux_x86_64.whl"

    def build_wheels(
        base_syms: Sequence[str], child_syms: Sequence[str]
    ) -> None:
      _compile_stub_so(
          base_glues_dir / "torch_tpu/_internal/env_2_15_0.so",
          undefined_symbols=base_syms,
      )
      _compile_stub_so(
          child_unpacked_dir / "torch_tpu/_internal/env_2_15_0.so",
          undefined_symbols=child_syms,
      )
      with zipfile.ZipFile(child_whl, "w") as zf:
        for p in child_unpacked_dir.rglob("*"):
          if p.is_file():
            zf.write(p, p.relative_to(child_unpacked_dir).as_posix())

    def run_hwm_diff(
        *extra_args: str,
    ) -> tuple[int, str, str]:
      stdout_buf = io.StringIO()
      stderr_buf = io.StringIO()
      with (
          contextlib.redirect_stdout(stdout_buf),
          contextlib.redirect_stderr(stderr_buf),
      ):
        rc = check_pytorch_abi_hwm_diff.main([
            f"--repo-root={test_repo}",
            f"--base-sha={base_sha}",
            f"--head-sha={head_sha}",
            f"--base-wheel-dir={base_glues_dir}",
            f"--wheel-dir={dist_dir}",
            *extra_args,
        ])
      return rc, stdout_buf.getvalue(), stderr_buf.getvalue()

    # 1. Missing wheel in wheel_dir fails cleanly.
    rc, _, err = run_hwm_diff()
    self.assertEqual(rc, 1)
    self.assertIn("Expected exactly one torch_tpu-*.whl", err)

    # 2. Missing symbol file fails cleanly.
    build_wheels(["_ZN3c105ErrorC1Ev"], ["_ZN3c105ErrorC1Ev"])
    rc, _, err = run_hwm_diff("--symbol-file=ci/wheel/nonexistent.txt")
    self.assertEqual(rc, 1)
    self.assertIn("does not exist", err)

    # 3. Unchanged symbols between diffbase and child succeeds.
    rc, _, err = run_hwm_diff()
    self.assertEqual(rc, 0, msg=err)

    # 4. Pre-existing / upstream drift already present on the diffbase (not
    # introduced by the child) does not move the HWM between diffbase and child,
    # so the check succeeds.
    build_wheels(
        ["_ZN3c105ErrorC1Ev", "_ZN2at4_ops3sum4callEv"],
        ["_ZN3c105ErrorC1Ev", "_ZN2at4_ops3sum4callEv"],
    )
    rc, _, err = run_hwm_diff()
    self.assertEqual(rc, 0, msg=err)

    # 5. Child introduces a new PyTorch symbol without updating the HWM file ->
    # fails and prompts the user to run the HWM updater in their PR.
    build_wheels(
        ["_ZN3c105ErrorC1Ev"],
        ["_ZN3c105ErrorC1Ev", "_ZN5torch7Library4_defEv"],
    )
    rc, out, err = run_hwm_diff()
    self.assertEqual(rc, 1)
    combined_out = out + "\n" + err
    self.assertIn("torch::Library::_def()", combined_out)
    self.assertIn(
        "python3 ci/wheel/pytorch_abi_symbols.py update", combined_out
    )

    # 6. Once the user runs the HWM updater in their PR, the check succeeds.
    pytorch_abi_symbols.update_high_water_mark(
        symbol_file,
        ["_ZN3c105ErrorC1Ev", "_ZN5torch7Library4_defEv"],
        version="2.15.0",
    )
    run_git("add", "ci/wheel/pytorch_symbols_2_15_0.txt")
    run_git("commit", "-m", "Update HWM")
    head_sha = run_git("rev-parse", "HEAD")
    rc, _, err = run_hwm_diff()
    self.assertEqual(rc, 0, msg=err)


if __name__ == "__main__":
  absltest.main()
