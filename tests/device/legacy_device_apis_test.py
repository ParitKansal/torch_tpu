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

"""Tests internal torch.tpu device APIs with TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS.

The BUILD target sets TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS=1, so the internal
device APIs aren't strict: calling them from user code warns once instead of
raising.
"""

import warnings

from absl.testing import absltest
import torch
import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]
from torch_tpu._internal.utils import annotations
from tests import seed_test_utils


class LegacyDeviceApisTest(seed_test_utils.RepeatableTest):

  def test_internal_device_count_warns_instead_of_raising(self):
    if annotations.get_torch_version() < (2, 11):
      self.skipTest("torch.tpu.device_count() is experimental.")
    self.assertEqual(
        getattr(torch.tpu.device_count, annotations.TT_API_STAGE), "Internal"
    )
    with self.assertWarnsRegex(UserWarning, "is internal to TorchTPU"):
      self.assertEqual(
          torch.tpu.device_count(), torch.accelerator.device_count()
      )
    # The warning is emitted only once per process.
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      self.assertEqual(
          torch.tpu.device_count(), torch.accelerator.device_count()
      )
    self.assertEmpty(w)


if __name__ == "__main__":
  absltest.main()
