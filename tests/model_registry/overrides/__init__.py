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

"""Declarative ModelOverride registry.

A `ModelOverride` is a declarative per-model-family record that captures
model-specific behavior when generic rules are insufficient—for example,
overriding modality resolution or providing a custom `input_builder` when
`forward` inputs cannot be synthesized from signature reflection alone.

Overrides are grouped by domain (`audio`, `diffusion`, `multimodal`, `text`,
`vision`); importing this package imports every domain module below so each one
registers its overrides once `overrides/registry.py` lands.
"""

# Domain modules register overrides on import.
# pylint: disable=g-bad-import-order,unused-import
from tests.model_registry.overrides import audio  # noqa: F401
from tests.model_registry.overrides import diffusion  # noqa: F401
from tests.model_registry.overrides import multimodal  # noqa: F401
from tests.model_registry.overrides import text  # noqa: F401
from tests.model_registry.overrides import vision  # noqa: F401

# pylint: enable=g-bad-import-order,unused-import
