# Copyright 2025 Google LLC
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

"""TorchTPU library.

Importing `torch` or `torch_tpu` loads TorchTPU automatically unless
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`. With `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, call
`torch.tpu.load()` after `import torch_tpu` to load TorchTPU explicitly.
"""

import os
import pathlib

from torch_tpu import _tcmalloc_check
from torch_tpu import _versioned_so_loader

# PyTorch's switch for loading out-of-tree device backends automatically.
_AUTOLOAD_ENV_VAR = "TORCH_DEVICE_BACKEND_AUTOLOAD"

# Install the version-dispatch import hook before any torch_tpu._internal C++
# extension is imported, so imports resolve to the glue built for the installed
# PyTorch. This is a no-op for an unversioned build, which has no bundled glue.
_versioned_so_loader.install_hook(pathlib.Path(__file__).parent)

try:
  import libtpu  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  libtpu.configure_library_path()
except ImportError:
  print("libtpu not found.")
  pass

# Warn if tcmalloc is not the allocator of this process. Checked last so the
# probe sees every library `import torch_tpu` pulls in, libtpu included.
_tcmalloc_check.check_tcmalloc()

# Load TorchTPU only when automatic loading is on, which is PyTorch's convention
# for TORCH_DEVICE_BACKEND_AUTOLOAD: with "0", neither `import torch` nor
# `import torch_tpu` loads it, and code that needs TorchTPU calls
# `torch.tpu.load()`. Once loaded, TorchTPU registers as PyTorch's PrivateUse1
# device, after which the first backward pass makes autograd try to initialize
# the TPU runtime, even on hosts without TPUs.
if os.getenv(_AUTOLOAD_ENV_VAR, "1") == "1":  # PY_ENVIRON_OK=pre-torch.
  from torch_tpu import _loader  # pylint: disable=g-import-not-at-top

  _loader.load()
else:
  import sys  # pylint: disable=g-import-not-at-top
  import types  # pylint: disable=g-import-not-at-top
  import torch  # pylint: disable=g-import-not-at-top

  def _load() -> None:
    """Loads TorchTPU's C++ extensions and registers PyTorch's `tpu` device."""
    from torch_tpu import _loader  # pylint: disable=g-import-not-at-top

    _loader.load()

  if "torch.tpu" not in sys.modules:
    _tpu_stub = types.ModuleType(
        "torch.tpu",
        "Unloaded TorchTPU module (`TORCH_DEVICE_BACKEND_AUTOLOAD=0`)."
        " Call `torch.tpu.load()` to load TorchTPU.",
    )
    _tpu_stub.load = _load  # type: ignore[attr-defined]
    sys.modules["torch.tpu"] = _tpu_stub
    setattr(torch, "tpu", _tpu_stub)

# TODO: b/477401982 - Introduce a new flag module to hold all TorchTPU flags.
