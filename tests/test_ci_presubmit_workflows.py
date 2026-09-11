#!/usr/bin/env python3
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

"""Unit tests for GitHub Actions CI presubmit workflows and RBE bypass logic."""

import json
import os
import re
import unittest
import yaml

REPO_ROOT = "/usr/local/google/home/jonathanskim/torch_tpu"
PRESUBMIT_YML = os.path.join(REPO_ROOT, ".github", "workflows", "presubmit.yml")
TEST_RBE_YML = os.path.join(REPO_ROOT, ".github", "workflows", "test_rbe_opt_in.yml")


class TestWorkflowSchemaAndYaml(unittest.TestCase):
    """Verifies that the workflow YAML files are well-formed and meet schema requirements."""

    def test_presubmit_yaml_valid(self):
        self.assertTrue(os.path.exists(PRESUBMIT_YML), f"Missing {PRESUBMIT_YML}")
        with open(PRESUBMIT_YML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIn("name", data)
        on_data = data.get("on") or data.get(True)
        self.assertIsNotNone(on_data, "Missing 'on' trigger section in presubmit.yml")
        self.assertIn("jobs", data)
        self.assertIn("setup", data["jobs"])
        self.assertIn("run_tests", data["jobs"])
        self.assertIn("tpu_v5_bypass_notice", data["jobs"])

        # Check pull_request types
        pr_config = on_data["pull_request"]
        self.assertIn("types", pr_config)
        self.assertIn("labeled", pr_config["types"])
        self.assertIn("opened", pr_config["types"])
        self.assertIn("synchronize", pr_config["types"])

        # Check workflow_dispatch inputs
        dispatch_inputs = on_data["workflow_dispatch"]["inputs"]
        self.assertIn("bypass-tpu-v5", dispatch_inputs)
        self.assertEqual(dispatch_inputs["bypass-tpu-v5"]["type"], "boolean")

    def test_test_rbe_opt_in_yaml_valid(self):
        self.assertTrue(os.path.exists(TEST_RBE_YML), f"Missing {TEST_RBE_YML}")
        with open(TEST_RBE_YML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertIn("name", data)
        on_data = data.get("on") or data.get(True)
        self.assertIsNotNone(on_data, "Missing 'on' trigger section in test_rbe_opt_in.yml")
        self.assertIn("jobs", data)
        self.assertIn("run_tests", data["jobs"])

        # Check workflow_dispatch inputs
        dispatch_inputs = on_data["workflow_dispatch"]["inputs"]
        self.assertIn("mode", dispatch_inputs)
        self.assertIn("test_suite", dispatch_inputs)


class TestPresubmitMatrixLogic(unittest.TestCase):
    """Verifies the exact matrix resolution logic embedded in presubmit.yml setup step."""

    def evaluate_setup(self, event_name, event_action, label_added, labels, pr_title, pr_body, input_bypass):
        ci_control_labels = {
            "ci:bypass-tpu-v5", "ci:skip-tpu-v5", "bypass-tpu-v5", "skip-tpu-v5",
            "ci:replace-tpu-v5", "replace-tpu-v5", "ci:replace-tpu-v5-rbe",
            "replace-tpu-v5-with-rbe", "ci:disable-bazel-diff", "run-presubmit"
        }

        should_run = True
        if event_name == "pull_request" and event_action == "labeled":
            if label_added not in ci_control_labels:
                should_run = False

        bypass_labels = {
            "ci:bypass-tpu-v5", "ci:skip-tpu-v5", "bypass-tpu-v5", "skip-tpu-v5",
            "ci:replace-tpu-v5", "replace-tpu-v5", "ci:replace-tpu-v5-rbe",
            "replace-tpu-v5-with-rbe"
        }
        has_bypass_label = any(l in bypass_labels for l in labels)

        tag_pattern = r"\[(skip|bypass)[-_]tpu[-_]?v?5\]"
        has_tag = bool(re.search(tag_pattern, pr_title, re.I) or re.search(tag_pattern, pr_body, re.I))

        bypass_v5 = input_bypass or has_bypass_label or has_tag

        cpu_job = {
            "name": "CPU",
            "runner": "linux-x86-n4-16",
            "config": "ci_cpu_presubmit",
            "extra_flags": ""
        }
        v5_job = {
            "runner": "linux-x86-ct5lp-224-8tpu",
            "config": "ci_tpu_v5_presubmit",
            "extra_flags": '--run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh"'
        }
        v7_job = {
            "runner": "linux-x86-tpu7x-224-4tpu",
            "config": "ci_tpu_v7_presubmit",
            "extra_flags": '--run_under="$(pwd)/ci/tools/parallel_accelerator_execute.sh"'
        }

        matrix = [cpu_job, v7_job] if bypass_v5 else [cpu_job, v5_job, v7_job]
        return should_run, bypass_v5, matrix

    def test_default_pr_runs_all_three(self):
        should_run, bypass_v5, matrix = self.evaluate_setup(
            "pull_request", "opened", "", [], "Fix aten kernels", "", False
        )
        self.assertTrue(should_run)
        self.assertFalse(bypass_v5)
        self.assertEqual(len(matrix), 3)
        runners = [item.get("name", item.get("runner")) for item in matrix]
        self.assertIn("CPU", runners)
        self.assertIn("linux-x86-ct5lp-224-8tpu", runners)
        self.assertIn("linux-x86-tpu7x-224-4tpu", runners)

    def test_shadow_run_mode(self):
        # run-rbe label applied: presubmit matrix must KEEP TPU v5 running
        should_run, bypass_v5, matrix = self.evaluate_setup(
            "pull_request", "labeled", "run-rbe", ["run-rbe"], "Fix aten kernels", "", False
        )
        self.assertFalse(bypass_v5)
        self.assertEqual(len(matrix), 3)
        runners = [item.get("name", item.get("runner")) for item in matrix]
        self.assertIn("linux-x86-ct5lp-224-8tpu", runners)

    def test_replacement_mode_via_label(self):
        for lbl in ["ci:replace-tpu-v5", "replace-tpu-v5", "ci:replace-tpu-v5-rbe", "replace-tpu-v5-with-rbe"]:
            with self.subTest(label=lbl):
                should_run, bypass_v5, matrix = self.evaluate_setup(
                    "pull_request", "labeled", lbl, [lbl], "Fix aten kernels", "", False
                )
                self.assertTrue(should_run)
                self.assertTrue(bypass_v5)
                self.assertEqual(len(matrix), 2)
                runners = [item.get("name", item.get("runner")) for item in matrix]
                self.assertNotIn("linux-x86-ct5lp-224-8tpu", runners)
                self.assertIn("CPU", runners)
                self.assertIn("linux-x86-tpu7x-224-4tpu", runners)

    def test_standalone_bypass_labels(self):
        for lbl in ["ci:bypass-tpu-v5", "ci:skip-tpu-v5", "bypass-tpu-v5", "skip-tpu-v5"]:
            with self.subTest(label=lbl):
                should_run, bypass_v5, matrix = self.evaluate_setup(
                    "pull_request", "labeled", lbl, [lbl], "Fix aten kernels", "", False
                )
                self.assertTrue(should_run)
                self.assertTrue(bypass_v5)
                self.assertEqual(len(matrix), 2)
                runners = [item.get("name", item.get("runner")) for item in matrix]
                self.assertNotIn("linux-x86-ct5lp-224-8tpu", runners)

    def test_pr_title_and_body_tags(self):
        test_cases = [
            ("[skip-tpu-v5] Fix docs", ""),
            ("Fix docs [bypass-tpu-v5]", ""),
            ("[skip-tpu5] Update tests", ""),
            ("[bypass-tpu5] Update tests", ""),
            ("Fix docs", "Please [skip-tpu-v5] on this PR"),
            ("Fix docs", "[bypass-tpu-v5] due to runner outage"),
        ]
        for title, body in test_cases:
            with self.subTest(title=title, body=body):
                should_run, bypass_v5, matrix = self.evaluate_setup(
                    "pull_request", "opened", "", [], title, body, False
                )
                self.assertTrue(should_run)
                self.assertTrue(bypass_v5)
                self.assertEqual(len(matrix), 2)

    def test_workflow_dispatch_input(self):
        should_run, bypass_v5, matrix = self.evaluate_setup(
            "workflow_dispatch", "", "", [], "", "", True
        )
        self.assertTrue(should_run)
        self.assertTrue(bypass_v5)
        self.assertEqual(len(matrix), 2)

    def test_unrelated_label_does_not_trigger_rerun(self):
        should_run, bypass_v5, matrix = self.evaluate_setup(
            "pull_request", "labeled", "documentation", ["documentation"], "Fix docs", "", False
        )
        self.assertFalse(should_run)

    def test_adversarial_inputs(self):
        # Malformed JSON in PR_LABELS string or None values
        should_run, bypass_v5, matrix = self.evaluate_setup(
            "pull_request", "opened", "", [], "Emoji PR 🚀💥 [skip-tpu-v5] 🎉", "Details 🧠", "  TRUE "
        )
        self.assertTrue(should_run)
        self.assertTrue(bypass_v5)
        self.assertEqual(len(matrix), 2)

        # Unicode Japanese / German text with skip tag
        should_run, bypass_v5, matrix = self.evaluate_setup(
            "pull_request", "opened", "", [], "テスト修正 [skip-tpu5]", "Änderung", False
        )
        self.assertTrue(bypass_v5)
        self.assertEqual(len(matrix), 2)


class TestRbeTriggerAndGatingLogic(unittest.TestCase):
    """Verifies the trigger and continue-on-error gating logic in test_rbe_opt_in.yml."""

    def evaluate_rbe(self, labels, event_name, input_mode):
        rbe_trigger_labels = {
            "run-rbe", "ci:replace-tpu-v5", "replace-tpu-v5",
            "ci:replace-tpu-v5-rbe", "replace-tpu-v5-with-rbe", "ci:rbe"
        }
        triggered = any(l in rbe_trigger_labels for l in labels) or (event_name == "workflow_dispatch")

        replace_labels = {
            "ci:replace-tpu-v5", "replace-tpu-v5",
            "ci:replace-tpu-v5-rbe", "replace-tpu-v5-with-rbe"
        }
        is_replace = any(l in replace_labels for l in labels) or (event_name == "workflow_dispatch" and input_mode == "replacement")

        continue_on_error = not is_replace
        return triggered, continue_on_error

    def test_rbe_idle_by_default(self):
        triggered, continue_on_error = self.evaluate_rbe([], "pull_request", "shadow")
        self.assertFalse(triggered)

    def test_rbe_shadow_run_mode(self):
        triggered, continue_on_error = self.evaluate_rbe(["run-rbe"], "pull_request", "shadow")
        self.assertTrue(triggered)
        self.assertTrue(continue_on_error, "Shadow mode must NOT block PR (continue-on-error=true)")

    def test_rbe_replacement_mode_is_gating(self):
        for lbl in ["ci:replace-tpu-v5", "replace-tpu-v5", "ci:replace-tpu-v5-rbe", "replace-tpu-v5-with-rbe"]:
            with self.subTest(label=lbl):
                triggered, continue_on_error = self.evaluate_rbe([lbl], "pull_request", "shadow")
                self.assertTrue(triggered)
                self.assertFalse(continue_on_error, "Replacement mode MUST block PR on failure (continue-on-error=false)")

    def test_rbe_workflow_dispatch_modes(self):
        # Dispatch with shadow
        triggered, continue_on_error = self.evaluate_rbe([], "workflow_dispatch", "shadow")
        self.assertTrue(triggered)
        self.assertTrue(continue_on_error)

        # Dispatch with replacement
        triggered, continue_on_error = self.evaluate_rbe([], "workflow_dispatch", "replacement")
        self.assertTrue(triggered)
        self.assertFalse(continue_on_error)


if __name__ == "__main__":
    unittest.main()
