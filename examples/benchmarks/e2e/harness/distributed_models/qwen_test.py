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

"""Unit tests for Qwen 3.5 397B distributed model benchmark."""

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from fairscale.nn.model_parallel import layers
import torch
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness.distributed_models import qwen
from tests import module_registry
from tests import seed_test_utils
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe


class QwenBenchmarkTest(seed_test_utils.RepeatableTest):

  def test_qwen_benchmark_registered(self):
    self.assertIn("qwen3_5_397b_a17b_forward", registry_lib.REGISTRY)
    spec = registry_lib.REGISTRY["qwen3_5_397b_a17b_forward"]
    self.assertEqual(spec.name, "qwen3_5_397b_a17b_forward")

  @flagsaver.flagsaver(weights_base_path="")
  @mock.patch.object(modeling_qwen3_5_moe, "Qwen3_5MoeForCausalLM")
  @mock.patch.object(module_registry, "ModuleRegistry")
  def test_get_qwen_model_config_from_module_registry(
      self, mock_registry_cls, mock_model_cls
  ):
    mock_model = mock.MagicMock(spec=torch.nn.Module)
    mock_model.named_modules.return_value = []
    mock_model_cls.return_value = mock_model

    fake_base_config = mock.MagicMock()
    fake_text_config = mock.MagicMock()
    fake_text_config.num_hidden_layers = 4
    fake_text_config.layer_types = ["linear", "linear", "linear", "linear"]
    fake_base_config.get_text_config.return_value = fake_text_config

    def fake_get_module_spec(provider, name, load_weights, modify_config_hook):
      self.assertEqual(provider, "transformers")
      self.assertEqual(name, "Qwen/Qwen3.5-397B-A17B")
      self.assertFalse(load_weights)
      modified_config = modify_config_hook(fake_base_config)
      return mock.MagicMock(config=modified_config)

    mock_registry = mock_registry_cls.return_value
    mock_registry.get_module_spec.side_effect = fake_get_module_spec

    qwen.get_qwen3_5_397b_a17b_model(
        rank=0,
        world_size=1,
        device="cpu",
        torch_dtype=torch.bfloat16,
        use_ragged_dot_moe=False,
        num_hidden_layers=2,
    )

    mock_model_cls.assert_called_once()
    passed_config = mock_model_cls.call_args[0][0]
    self.assertEqual(passed_config.num_hidden_layers, 2)
    self.assertEqual(passed_config.layer_types, ["linear", "linear"])

  def test_supported_platforms_does_not_contain_gpu(self):
    self.assertNotIn("b200_8", [p.value for p in qwen._SUPPORTED_PLATFORMS])

  def test_apply_tensor_parallel_plan_colwise_gather_output(self):
    class DummySubmodule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(64, 64)
        self.out_proj = torch.nn.Linear(64, 64)

    class DummyModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear_attn = DummySubmodule()

    model = DummyModel()
    tp_plan = {
        r"linear_attn\.in_proj": "colwise_gather_output",
        r"linear_attn\.out_proj": "rowwise",
    }
    with mock.patch(
        "fairscale.nn.model_parallel.layers.get_model_parallel_world_size",
        return_value=1,
    ):
      with mock.patch(
          "fairscale.nn.model_parallel.layers.get_model_parallel_rank",
          return_value=0,
      ):
        qwen._apply_tensor_parallel_plan(
            model,
            tp_plan=tp_plan,
            world_size=1,
            rank=0,
        )

    self.assertIsInstance(
        model.linear_attn.in_proj, layers.ColumnParallelLinear
    )
    self.assertTrue(model.linear_attn.in_proj.gather_output)
    self.assertIsInstance(model.linear_attn.out_proj, layers.RowParallelLinear)

  def test_initialize_weights_and_biases_preserves_dtype(self):
    linear = torch.nn.Linear(16, 16, dtype=torch.float32)
    qwen._initialize_weights_and_biases(linear, "test_layer")
    self.assertEqual(linear.weight.dtype, torch.float32)


if __name__ == "__main__":
  absltest.main()
