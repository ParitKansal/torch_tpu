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

"""Unit tests for ci/tools/release_test_matrix.py."""

import io
import json
import re
import unittest
from unittest import mock

from bazel import pytorch_versions
from ci.tools import release_test_matrix


class ParseCsvTest(unittest.TestCase):

  def test_keeps_order_and_strips_whitespace(self):
    self.assertEqual(
        release_test_matrix.parse_csv(" v7, v5 ,", ["v5", "v6", "v7"], "--f"),
        ["v7", "v5"],
    )

  def test_empty_means_every_allowed_value(self):
    self.assertEqual(
        release_test_matrix.parse_csv(" , ", ["v5", "v6"], "--f"),
        ["v5", "v6"],
    )

  def test_rejects_unknown_values(self):
    with self.assertRaisesRegex(ValueError, r"\['v4', 'v8'\]"):
      release_test_matrix.parse_csv("v4,v5,v8", ["v5"], "--f")


class RepositoryVersionsTest(unittest.TestCase):

  def test_supported_python_versions_come_from_bzl(self):
    versions = release_test_matrix.supported_python_versions()
    self.assertIn("3.14", versions)
    for version in versions:
      self.assertRegex(version, r"^\d+\.\d+$")

  def test_released_torch_versions_are_sorted_and_unique(self):
    versions = release_test_matrix.released_torch_versions()
    self.assertEqual(len(versions), len(set(versions)))
    self.assertEqual(
        list(versions),
        sorted(versions, key=lambda v: tuple(map(int, v.split(".")))),
    )
    self.assertIn(
        pytorch_versions.read_value("DEFAULT_TORCH_VERSION")[0], versions
    )
    for intermediate in pytorch_versions.read_value(
        "INTERMEDIATE_PATCH_VERSIONS"
    ):
      self.assertIn(intermediate, versions)


class BuildMatrixTest(unittest.TestCase):

  def test_entry_carries_runner_and_resolution(self):
    matrix = release_test_matrix.build_matrix(
        ["v7"], ["3.12"], ["2.13.0"], ["newest"]
    )
    self.assertEqual(
        matrix,
        {
            "entry": [{
                "tpu_generation": "v7",
                "tpu_name": "TPU v7x",
                "runner": "linux-x86-tpu7x-224-4tpu",
                "python_version": "3.12",
                "torch_version": "2.13.0",
                "deps": "newest",
                "uv_resolution": "highest",
            }]
        },
    )

  def test_product_order_is_tpu_python_torch_deps(self):
    matrix = release_test_matrix.build_matrix(
        ["v5", "v6"], ["3.11", "3.14"], ["2.11.0"], ["oldest", "newest"]
    )
    entries = matrix["entry"]
    self.assertEqual(len(entries), 2 * 2 * 1 * 2)
    keys = [
        (e["tpu_generation"], e["python_version"], e["deps"]) for e in entries
    ]
    self.assertEqual(
        keys[:4],
        [
            ("v5", "3.11", "oldest"),
            ("v5", "3.11", "newest"),
            ("v5", "3.14", "oldest"),
            ("v5", "3.14", "newest"),
        ],
    )
    self.assertTrue(all(k[0] == "v6" for k in keys[4:]))


class MainTest(unittest.TestCase):

  def test_prints_compact_json(self):
    stdout = io.StringIO()
    with mock.patch("sys.stdout", stdout):
      status = release_test_matrix.main([
          "--tpu_generations=v5",
          "--python_versions=3.14",
          "--torch_versions=2.11.0,2.12.1",
          "--dep_resolutions=oldest",
      ])
    self.assertEqual(status, 0)
    output = stdout.getvalue()
    self.assertEqual(output.count("\n"), 1)
    self.assertNotRegex(output, re.compile(r":\s|,\s"))
    matrix = json.loads(output)
    self.assertEqual(
        [e["torch_version"] for e in matrix["entry"]], ["2.11.0", "2.12.1"]
    )

  def test_no_flags_covers_the_whole_product(self):
    stdout = io.StringIO()
    with mock.patch("sys.stdout", stdout):
      status = release_test_matrix.main([])
    self.assertEqual(status, 0)
    matrix = json.loads(stdout.getvalue())
    expected = (
        len(release_test_matrix.TPU_GENERATIONS)
        * len(release_test_matrix.supported_python_versions())
        * len(release_test_matrix.released_torch_versions())
        * len(release_test_matrix.DEP_RESOLUTIONS)
    )
    self.assertEqual(len(matrix["entry"]), expected)

  def test_unsupported_value_fails_with_message(self):
    stderr = io.StringIO()
    with mock.patch("sys.stderr", stderr):
      status = release_test_matrix.main([
          "--tpu_generations=v5",
          "--python_versions=3.14",
          "--torch_versions=nightly",
          "--dep_resolutions=oldest",
      ])
    self.assertEqual(status, 1)
    self.assertIn("--torch_versions has unsupported values ['nightly']",
                  stderr.getvalue())


if __name__ == "__main__":
  unittest.main()
