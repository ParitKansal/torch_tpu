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

"""Tests for torch_device_ops."""

import sys
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from perflab import target as target_lib
from perflab import torch_device_ops
from tests import seed_test_utils


class TorchDeviceOpsTest(seed_test_utils.RepeatableTest):

  def test_init_success(self):
    target = target_lib.make_target(
        platform=target_lib.Platform.V5E_1X1,
        dtype=target_lib.DType.BF16,
    )
    ops = torch_device_ops.TorchDeviceOps(target)
    self.assertEqual(ops.device.type, "tpu")
    self.assertEqual(ops.dtype, torch.bfloat16)

  @mock.patch("torch.accelerator")
  def test_init_device_count_mismatch(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 0
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        torch_device_ops.DeviceCountMismatch,
        r"expects 1 device\(s\) but the host has 0",
    ):
      torch_device_ops.TorchDeviceOps(target)

  @mock.patch("torch.accelerator")
  def test_await_result(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)

    ops.await_result(None)

    mock_accelerator.synchronize.assert_called_once()

  @mock.patch("torch.accelerator")
  def test_reset_peak_memory(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)

    ops.reset_peak_memory()

    mock_accelerator.reset_peak_memory_stats.assert_called_once()

  @mock.patch("torch.accelerator")
  def test_peak_memory_mb(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    mock_accelerator.max_memory_allocated.return_value = 5242880  # 5 * 2**20

    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    ops = torch_device_ops.TorchDeviceOps(target)
    peak_mb = ops.peak_memory_mb()

    self.assertEqual(peak_mb, 5.0)
    mock_accelerator.max_memory_allocated.assert_called_once()

  @mock.patch("torch.accelerator")
  def test_compile_count_dynamo(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)

    ops = torch_device_ops.TorchDeviceOps(target)
    with mock.patch.object(
        ops, "_dynamo_counters", return_value={"frames": {"ok": 42}}
    ):
      count = ops.compile_count()

    self.assertEqual(count, 42)

  def test_compile_count_dynamo_real_compile(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)

    ops = torch_device_ops.TorchDeviceOps(target)
    model = torch.nn.Linear(2, 2)
    compiled_model = torch.compile(model, backend="inductor")
    _ = compiled_model(torch.randn(2, 2))

    self.assertGreater(ops.compile_count(), 0)

  def test_compile_count_dynamo_none(self):
    target = target_lib.make_target(platform=target_lib.Platform.B200_1)

    ops = torch_device_ops.TorchDeviceOps(target)
    with mock.patch.object(ops, "_dynamo_counters", return_value=None):
      count = ops.compile_count()

    self.assertEqual(count, 0)

  @mock.patch("torch.accelerator")
  def test_compile_count_tpu(self, mock_accelerator):
    mock_accelerator.device_count.return_value = 1
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)

    ops = torch_device_ops.TorchDeviceOps(target)
    with mock.patch.object(torch, "tpu", create=True) as mock_tpu:
      mock_tpu._get_cache_misses.return_value = 7
      count = ops.compile_count()

    self.assertEqual(count, 7)


def _module(name: str, file: str) -> types.ModuleType:
  module = types.ModuleType(name)
  module.__file__ = file
  return module


def _without_torch_tpu(*modules: types.ModuleType) -> dict[str, Any]:
  """This process' `sys.modules` without TorchTPU, plus `modules`."""
  result = {
      name: module
      for name, module in sys.modules.items()
      if name.split(".")[0] != "torch_tpu"
  }
  result.update({module.__name__: module for module in modules})
  return result


class CheckTorchTpuNotLoadedTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      target_lib.Platform.CPU,
      target_lib.Platform.B200_1,
      target_lib.Platform.B200_8,
  )
  def test_raises_on_gpu_and_cpu_when_loader_is_imported(self, platform):
    loader = _module("torch_tpu._loader", "/torch_tpu/_loader.py")
    with mock.patch.dict(sys.modules, _without_torch_tpu(loader), clear=True):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
          RuntimeError,
          rf"torch_tpu is loaded on non-tpu platform '{platform.value}'"
          r" \(torch_tpu\._loader\)",
      ):
        torch_device_ops.check_torch_tpu_not_loaded(
            target_lib.make_target(platform)
        )

  def test_raises_when_an_extension_module_is_imported(self):
    extension = _module(
        "torch_tpu._internal.env", "/torch_tpu/_internal/env.so"
    )
    with mock.patch.dict(
        sys.modules, _without_torch_tpu(extension), clear=True
    ):
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
          RuntimeError, r"\(torch_tpu\._internal\.env\)"
      ):
        torch_device_ops.check_torch_tpu_not_loaded(
            target_lib.make_target(target_lib.Platform.B200_1)
        )

  def test_passes_with_only_pure_python_torch_tpu_modules(self):
    python_module = _module(
        "torch_tpu.perflab.cases", "/torch_tpu/perflab/cases.py"
    )
    with mock.patch.dict(
        sys.modules, _without_torch_tpu(python_module), clear=True
    ):
      torch_device_ops.check_torch_tpu_not_loaded(
          target_lib.make_target(target_lib.Platform.B200_1)
      )

  def test_passes_on_tpu_when_loaded(self):
    loader = _module("torch_tpu._loader", "/torch_tpu/_loader.py")
    with mock.patch.dict(sys.modules, _without_torch_tpu(loader), clear=True):
      torch_device_ops.check_torch_tpu_not_loaded(
          target_lib.make_target(target_lib.Platform.V7_1X1X1)
      )


if __name__ == "__main__":
  absltest.main()
