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

"""Tests for the CI presubmit workflows and the TPU v5 bypass path.

These run the real matrix script and read the real workflow YAML. Nothing here
re-implements the workflow logic, so a broken workflow fails the test.
"""

import json
import os
import re
import subprocess
import unittest

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_DIR = os.path.join(REPO_ROOT, ".github", "workflows")
PRESUBMIT_YML = os.path.join(WORKFLOW_DIR, "presubmit.yml")
TEST_RBE_YML = os.path.join(WORKFLOW_DIR, "test_rbe_opt_in.yml")
MATRIX_SCRIPT = os.path.join(
    REPO_ROOT, "ci", "tools", "presubmit_job_matrix.sh"
)
BAZELRC = os.path.join(REPO_ROOT, ".bazelrc")

TPU_V5_RUNNER = "linux-x86-ct5lp-224-8tpu"


def load_workflow(path):
  """Parses a workflow file, working around YAML reading `on:` as True."""
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if True in data:
    data["on"] = data.pop(True)
  return data


def run_matrix_script(bypass=None):
  """Runs the matrix script and returns (parsed matrix, raw stdout)."""
  env = dict(os.environ)
  env.pop("BYPASS_TPU_V5", None)
  if bypass is not None:
    env["BYPASS_TPU_V5"] = bypass
  result = subprocess.run(
      ["bash", MATRIX_SCRIPT],
      cwd=REPO_ROOT,
      env=env,
      capture_output=True,
      text=True,
      check=True,
  )
  return json.loads(result.stdout), result.stdout


def runners(matrix):
  return [entry["runner"] for entry in matrix]


def control_labels(expression):
  """Pulls the label list out of a `contains(fromJSON('[...]'), ...)` call."""
  match = re.search(r"fromJSON\('(\[.*?\])'\)", expression, re.DOTALL)
  if not match:
    return None
  return json.loads(match.group(1))


class TestPresubmitJobMatrixScript(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests workflow config.
):
  """Runs ci/tools/presubmit_job_matrix.sh and checks what it prints."""

  def test_default_matrix_covers_cpu_v5_and_v7(self):
    matrix, _ = run_matrix_script()
    self.assertEqual(
        runners(matrix),
        ["linux-x86-n4-16", TPU_V5_RUNNER, "linux-x86-tpu7x-224-4tpu"],
    )

  def test_bypass_drops_only_the_tpu_v5_entry(self):
    default_matrix, _ = run_matrix_script()
    bypassed, _ = run_matrix_script(bypass="true")
    self.assertNotIn(TPU_V5_RUNNER, runners(bypassed))
    self.assertEqual(
        bypassed,
        [e for e in default_matrix if e["runner"] != TPU_V5_RUNNER],
        "Bypass must remove the TPU v5 entry and leave the others untouched",
    )

  def test_only_the_literal_string_true_bypasses(self):
    for value in ["false", "", "TRUE", "1", "yes"]:
      with self.subTest(value=value):
        matrix, _ = run_matrix_script(bypass=value)
        self.assertIn(TPU_V5_RUNNER, runners(matrix))

  def test_output_is_a_single_line(self):
    # The workflow appends this to $GITHUB_OUTPUT as `job_matrix=<json>`, which
    # only reads back correctly if the JSON has no embedded newlines.
    _, raw = run_matrix_script()
    self.assertEqual(raw.count("\n"), 1)

  def test_accelerator_runners_use_the_chip_lease_wrapper(self):
    matrix, _ = run_matrix_script()
    for entry in matrix:
      if entry["runner"] == "linux-x86-n4-16":
        self.assertEqual(entry["extra_flags"], "")
      else:
        self.assertIn(
            "ci/tools/parallel_accelerator_execute.sh", entry["extra_flags"]
        )

  def test_default_matrix_matches_upstream_main(self):
    """The default matrix must stay identical to the one on main.

    Without this, a change here silently drops or renames a presubmit leg.
    """
    try:
      upstream = subprocess.run(
          ["git", "show", "origin/main:.github/workflows/presubmit.yml"],
          cwd=REPO_ROOT,
          capture_output=True,
          text=True,
          check=True,
      ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
      self.skipTest("origin/main is not available in this checkout")

    upstream_matrix = yaml.safe_load(upstream)["jobs"]["run_tests"]["strategy"][
        "matrix"
    ]["job_info"]
    generated, _ = run_matrix_script()
    self.assertEqual(generated, upstream_matrix)


class TestPresubmitWorkflow(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests workflow config.
):
  """Checks presubmit.yml wiring that the matrix script can't cover."""

  @classmethod
  def setUpClass(cls):
    cls.workflow = load_workflow(PRESUBMIT_YML)
    cls.jobs = cls.workflow["jobs"]

  def test_reacts_to_label_changes(self):
    types = self.workflow["on"]["pull_request"]["types"]
    for event_type in ["opened", "synchronize", "reopened", "labeled"]:
      self.assertIn(event_type, types)

  def test_bypass_is_also_a_manual_dispatch_input(self):
    dispatch = self.workflow["on"]["workflow_dispatch"]["inputs"]
    self.assertEqual(dispatch["bypass-tpu-v5"]["type"], "boolean")
    self.assertFalse(dispatch["bypass-tpu-v5"]["default"])

  def test_concurrency_and_setup_agree_on_control_labels(self):
    """Both sites list the labels that CI reacts to, so they must match.

    `concurrency` can't read `env`, so the list is written twice. If they drift,
    a control label either cancels a run it shouldn't or fails to start one.
    """
    from_concurrency = control_labels(self.workflow["concurrency"]["group"])
    from_setup = control_labels(self.jobs["setup"]["if"])
    self.assertIsNotNone(from_concurrency)
    self.assertIsNotNone(from_setup)
    self.assertEqual(from_concurrency, from_setup)

  def test_control_labels_include_both_bypass_labels(self):
    labels = control_labels(self.jobs["setup"]["if"])
    self.assertIn("ci:bypass-tpu-v5", labels)
    self.assertIn("ci:replace-tpu-v5", labels)

  def test_unrelated_labels_get_their_own_concurrency_group(self):
    group = self.workflow["concurrency"]["group"]
    self.assertIn("github.event.action == 'labeled'", group)
    self.assertIn("ignored-label", group)
    self.assertIn("github.run_id", group)

  def test_setup_gate_lets_non_label_events_through(self):
    self.assertIn("github.event.action != 'labeled'", self.jobs["setup"]["if"])

  def test_both_bypass_labels_set_bypass_tpu_v5(self):
    bypass_env = self.jobs["setup"]["steps"][-1]["env"]["BYPASS_TPU_V5"]
    self.assertIn("inputs.bypass-tpu-v5", bypass_env)
    self.assertIn("'ci:bypass-tpu-v5'", bypass_env)
    self.assertIn("'ci:replace-tpu-v5'", bypass_env)

  def test_run_tests_takes_its_matrix_from_setup(self):
    run_tests = self.jobs["run_tests"]
    self.assertEqual(run_tests["needs"], "setup")
    self.assertIn(
        "needs.setup.outputs.job_matrix",
        run_tests["strategy"]["matrix"]["job_info"],
    )

  def test_bypass_notice_claims_the_check_name_only_when_bypassed(self):
    """The notice stands in for the required check, so the names must match.

    GitHub publishes a check run for a skipped job too, so naming this job after
    the TPU v5 runner unconditionally would put two check runs under that name
    on every run that isn't bypassed, and the skipped one reads as a failure.
    """
    matrix, _ = run_matrix_script()
    v5_entry = next(e for e in matrix if e["runner"] == TPU_V5_RUNNER)
    expected = self.jobs["run_tests"]["name"].replace(
        "${{ matrix.job_info.name || matrix.job_info.runner }}",
        v5_entry.get("name", v5_entry["runner"]),
    )
    name = self.jobs["tpu_v5_bypass_notice"]["name"]
    self.assertIn("needs.setup.outputs.tpu_v5_bypassed == 'true'", name)
    self.assertIn(f"'{expected}'", name)
    self.assertNotIn(expected, name.split("||")[-1])

  def test_bypass_notice_only_runs_when_v5_is_bypassed(self):
    notice = self.jobs["tpu_v5_bypass_notice"]
    self.assertEqual(notice["needs"], "setup")
    self.assertIn("needs.setup.outputs.tpu_v5_bypassed == 'true'", notice["if"])

  def test_no_job_hardcodes_a_check_name_the_matrix_also_produces(self):
    """Two jobs sharing a check name make the skipped one look like a failure.

    GitHub publishes a check run for a skipped job, so a second job that always
    carries a matrix leg's name adds a `skipped` check run under that name on
    every run. Anything gating on the name then sees a result that never turns
    green. Take such a name only behind an expression.
    """
    matrix, _ = run_matrix_script()
    template = self.jobs["run_tests"]["name"]
    matrix_names = {
        template.replace(
            "${{ matrix.job_info.name || matrix.job_info.runner }}",
            entry.get("name", entry["runner"]),
        )
        for entry in matrix
    }

    for job_id, job in self.jobs.items():
      if job_id == "run_tests":
        continue
      name = job.get("name", "")
      if "${{" in name:
        continue
      self.assertNotIn(
          name,
          matrix_names,
          f"Job '{job_id}' always publishes the check name '{name}', which the"
          " run_tests matrix also produces",
      )


class TestRbeOptInWorkflow(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests workflow config.
):
  """Checks the trigger, gating, and auth wiring in test_rbe_opt_in.yml."""

  @classmethod
  def setUpClass(cls):
    cls.workflow = load_workflow(TEST_RBE_YML)
    cls.job = cls.workflow["jobs"]["run_tests"]
    cls.steps = {step["name"]: step for step in cls.job["steps"]}

  def test_runs_only_when_opted_in(self):
    condition = self.job["if"]
    self.assertIn("'run-rbe'", condition)
    self.assertIn("'ci:replace-tpu-v5'", condition)
    self.assertIn("github.event_name == 'workflow_dispatch'", condition)

  def test_replacement_mode_gates_the_pr(self):
    """continue-on-error must be false exactly when RBE stands in for TPU v5."""
    expression = self.job["continue-on-error"]
    self.assertIn("'ci:replace-tpu-v5'", expression)
    self.assertIn("inputs.mode == 'replacement'", expression)
    # The whole opt-in test is negated, so anything else stays advisory.
    self.assertTrue(expression.lstrip().startswith("${{ !("))

  def test_shadow_label_is_not_in_the_gating_expression(self):
    self.assertNotIn("run-rbe", self.job["continue-on-error"])

  def test_auth_uses_workload_identity_not_a_service_account_key(self):
    """rbe-tpu-oss blocks service account key creation, so WIF is the only way."""
    auth = self.steps["Authenticate to GCP RBE"]
    self.assertIn("workload_identity_provider", auth["with"])
    self.assertNotIn("credentials_json", auth["with"])
    self.assertEqual(self.workflow["permissions"]["id-token"], "write")

  def test_third_party_actions_are_pinned_to_a_commit(self):
    for name, step in self.steps.items():
      if "uses" not in step:
        continue
      with self.subTest(step=name):
        ref = step["uses"].split("@")[1]
        self.assertRegex(ref, r"^[0-9a-f]{40}$", "Pin actions to a full SHA")

  def test_missing_credentials_fail_a_gating_run(self):
    check = self.steps["Check RBE credentials are configured"]
    self.assertIn("'ci:replace-tpu-v5'", check["env"]["IS_GATING"])
    self.assertIn("exit 1", check["run"])

  def test_steps_needing_credentials_are_skipped_without_them(self):
    for name in ["Authenticate to GCP RBE", "Set up Bazel", "Run Test Suite"]:
      with self.subTest(step=name):
        self.assertIn(
            "steps.creds.outputs.configured == 'true'", self.steps[name]["if"]
        )

  def test_test_suite_input_selects_a_matrix_leg(self):
    suites = {
        entry["suite"] for entry in self.job["strategy"]["matrix"]["job_info"]
    }
    options = set(
        self.workflow["on"]["workflow_dispatch"]["inputs"]["test_suite"][
            "options"
        ]
    )
    self.assertEqual(options, suites | {"all"})

  def test_excluded_targets_are_negated_bazel_patterns(self):
    excluded = self.workflow["env"]["RBE_EXCLUDED_TARGETS"].split()
    self.assertTrue(excluded)
    for target in excluded:
      with self.subTest(target=target):
        self.assertTrue(target.startswith("-//"))


class TestTpuConfigsPinOnlyTestActions(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests workflow config.
):
  """TPU CI must send compilation to RBE and keep only tests on the runner.

  `--spawn_strategy` sets the default for every spawn, so
  `--spawn_strategy=standalone,local` also dragged every compile action onto
  the ct5lp host while its eight chips sat idle. A live A/B against
  projects/tensorflow-testing showed the same genrule reporting runner=local
  under `--spawn_strategy` and runner=remote under `--strategy=TestRunner`.
  """

  TPU_CONFIGS = (
      "ci_tpu_base",
      "ci_tpu_nightly",
      "ci_tpu_v5",
      "ci_tpu_v5_presubmit",
      "ci_tpu_v6",
  )

  def flags_for(self, config, seen=None):
    """Expands a --config the way bazel does, following --config= chains."""
    seen = set() if seen is None else seen
    if config in seen:
      return []
    seen.add(config)

    flags = []
    with open(BAZELRC, encoding="utf-8") as fh:
      for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
          continue
        head, _, rest = line.partition(" ")
        if head.partition(":")[2] != config:
          continue
        for flag in rest.split():
          if flag.startswith("--config="):
            flags.extend(self.flags_for(flag.split("=", 1)[1], seen))
          else:
            flags.append(flag)
    return flags

  def test_every_tpu_config_pins_the_test_mnemonic(self):
    for config in self.TPU_CONFIGS:
      with self.subTest(config=config):
        self.assertIn("--strategy=TestRunner=local", self.flags_for(config))

  def test_no_tpu_config_sets_a_default_spawn_strategy(self):
    for config in self.TPU_CONFIGS:
      with self.subTest(config=config):
        pins = [
            f
            for f in self.flags_for(config)
            if f.startswith("--spawn_strategy")
        ]
        self.assertEqual(pins, [], f"{config} pins every spawn locally: {pins}")

  def test_tpu_configs_still_reach_remote_execution(self):
    """A local pin is pointless if the config never had an executor."""
    for config in self.TPU_CONFIGS:
      with self.subTest(config=config):
        flags = self.flags_for(config)
        self.assertTrue(
            any(f.startswith("--remote_executor=") for f in flags),
            f"{config} has no remote executor to offload to",
        )

  def test_the_benchmark_config_keeps_its_local_pin(self):
    """Benchmarks want stable timing, not throughput. Leave that one alone."""
    self.assertIn("--spawn_strategy=standalone,local", self.flags_for("bench"))


class TestWorkflowsOnlyNameConfigsThatExist(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests workflow config.
):
  """Bazel exits immediately on an undefined --config, before running anything.

  test_rbe_opt_in.yml shipped naming `ci_cpu_presubmit_rbe` and
  `ci_tpu_v5_full_rbe`; neither was defined, so both matrix jobs could only
  fail at startup. The workflow is opt-in and had never been run, so nothing
  caught it.
  """

  def defined_configs(self):
    names = set()
    with open(BAZELRC, encoding="utf-8") as fh:
      for raw in fh:
        line = raw.strip()
        if line and not line.startswith("#"):
          names.add(line.partition(" ")[0].partition(":")[2])
    names.discard("")
    return names

  def referenced_configs(self, path):
    """Matrix entries plus any --config= written out literally."""
    with open(path, encoding="utf-8") as fh:
      text = fh.read()

    names = set(re.findall(r"--config=([A-Za-z0-9_]+)", text))
    for job in yaml.safe_load(text).get("jobs", {}).values():
      for entry in (
          job.get("strategy", {}).get("matrix", {}).get("job_info", [])
      ):
        if isinstance(entry, dict) and entry.get("config"):
          names.add(entry["config"])
    return names

  def test_every_config_a_workflow_names_is_defined(self):
    defined = self.defined_configs()
    for path in (PRESUBMIT_YML, TEST_RBE_YML):
      for config in sorted(self.referenced_configs(path)):
        with self.subTest(workflow=os.path.basename(path), config=config):
          self.assertIn(config, defined, f"{config} is not defined in .bazelrc")


class TestOssShardCountsStaySized(
    unittest.TestCase  # UNITTEST_OK=No RNG; tests workflow config.
):
  """Guards the OSS shard counts on the heavy ops test targets.

  OSS caps samples per op/dtype, so these shards finish in seconds while each
  one still pays a fixed startup: about 79s for the ops_test.py targets that
  load golden data, 14-31s for the rest. Sharding past that point only adds
  machine time to the presubmit without testing anything extra, and the counts
  below were sized from per-action timings measured on v5e.
  """

  BUILD_FILE = os.path.join(REPO_ROOT, "tests", "BUILD")

  # target -> most OSS shards that still earn their startup cost
  SHARD_CEILINGS = {
      "ops_test": 16,
      "ops_test_compiled": 16,
      "foreach_ops_test": 20,
      "ops_unit_test": 16,
      "ops_unit_test_tier3_cache": 16,
      "ops_unit_test_tier3_cache_no_backup": 16,
      "ops_test_grad_vs_cpu": 20,
      "ops_test_dynamic_vs_cpu": 12,
  }

  def shard_counts(self):
    """Maps target name to the raw shard_count expression in tests/BUILD."""
    counts = {}
    name = None
    with open(self.BUILD_FILE, encoding="utf-8") as build_file:
      for line in build_file:
        match = re.match(r'\s+name = "([^"]+)"', line)
        if match:
          name = match.group(1)
        match = re.match(r"\s+shard_count = (.+),\s*$", line)
        if match and name is not None:
          counts[name] = match.group(1)
    return counts

  def test_heavy_targets_split_oss_and_internal_shard_counts(self):
    counts = self.shard_counts()
    for target in self.SHARD_CEILINGS:
      with self.subTest(target=target):
        self.assertIn(target, counts, f"{target} has no shard_count")
        self.assertRegex(
            counts[target],
            r"^if_oss\(\d+, \d+\)$",
            f"{target} should set OSS and internal shard counts separately",
        )

  def test_oss_shard_counts_stay_under_their_ceilings(self):
    counts = self.shard_counts()
    for target, ceiling in self.SHARD_CEILINGS.items():
      with self.subTest(target=target):
        match = re.match(r"^if_oss\((\d+), \d+\)$", counts[target])
        self.assertIsNotNone(match, f"{target} does not use if_oss")
        self.assertLessEqual(
            int(match.group(1)),
            ceiling,
            f"each extra {target} shard repeats its startup cost",
        )


if __name__ == "__main__":
  unittest.main()
