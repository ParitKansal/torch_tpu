#!/usr/bin/env python3
"""generate_presubmit_report.py

Parses test logs, JUnit XML artifacts, and Bazel execution metadata
to produce presubmit_summary.json and presubmit_report.md.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET


def strip_ansi(text: str) -> str:
  ansi_regex = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\([a-zA-Z]")
  return ansi_regex.sub("", text)


def extract_diagnostics(log_path: str, max_lines: int = 50) -> str:
  if not log_path or not os.path.isfile(log_path):
    return "Log file not found."

  try:
    with open(log_path, "r", errors="replace") as f:
      raw_text = f.read()
  except OSError as e:
    return f"Failed to read log file: {e}"

  lines = strip_ansi(raw_text).splitlines()
  if not lines:
    return "Empty log file."

  # 1. Search for Python Traceback
  tb_indices = [
      i
      for i, line in enumerate(lines)
      if "Traceback (most recent call last):" in line
  ]
  if tb_indices:
    start = tb_indices[-1]
    return "\n".join(lines[start : start + max_lines])

  # 2. Search for C++ Fatal Errors, SIGSEGV, or Check failed
  fatal_patterns = [
      "Check failed:",
      "SIGSEGV",
      "SIGABRT",
      "SIGBUS",
      "SIGFPE",
      "Segmentation fault",
      "Fatal error",
      "terminate called",
      "Assertion failed",
  ]
  fatal_indices = [
      i
      for i, line in enumerate(lines)
      if any(pat in line for pat in fatal_patterns)
  ]
  if fatal_indices:
    start = max(0, fatal_indices[-1] - 3)
    return "\n".join(lines[start : start + max_lines])

  # 3. Search for TPU device or runtime errors
  tpu_patterns = [
      "InitializePjrtPlugin failed",
      "The TPU is already in use",
      "No such device or address",
      "TPU_VISIBLE_DEVICES",
      "Failed to open /dev/vfio",
      "ERROR: TPU",
  ]
  tpu_indices = [
      i
      for i, line in enumerate(lines)
      if any(pat in line for pat in tpu_patterns)
  ]
  if tpu_indices:
    start = max(0, tpu_indices[0] - 2)
    return "\n".join(lines[start : start + max_lines])

  # 4. Search for Relay runner errors or preemption
  relay_patterns = [
      "ERROR [relay_test_runner]",
      "ERROR [remote_tpu_executor]",
      "Project boundary violation",
      "Preemption detected",
      "ControlMaster connection failed",
  ]
  relay_indices = [
      i
      for i, line in enumerate(lines)
      if any(pat in line for pat in relay_patterns)
  ]
  if relay_indices:
    start = relay_indices[0]
    return "\n".join(lines[start : start + max_lines])

  # 5. Fallback: last 40 lines
  return "\n".join(lines[-40:] if len(lines) > 40 else lines)


def target_to_testlog_relpath(target: str) -> str:
  clean = target.lstrip("/")
  if ":" in clean:
    pkg, name = clean.split(":", 1)
    return os.path.join(pkg, name)
  return clean


def testlog_relpath_to_target(rel_path: str) -> str:
  parts = rel_path.strip("/").split(os.sep)
  if len(parts) == 1:
    return f"//:{parts[0]}"
  pkg = "/".join(parts[:-1])
  name = parts[-1]
  return f"//{pkg}:{name}"


def _safe_int(val, default=0):
  if val is None:
    return default
  try:
    return int(float(str(val).strip()))
  except (ValueError, TypeError, OverflowError):
    return default


def _safe_float(val, default=0.0):
  if val is None:
    return default
  try:
    f = float(str(val).strip())
    if math.isnan(f) or math.isinf(f):
      return default
    return f
  except (ValueError, TypeError):
    return default


def parse_xml_report(xml_path: str):
  if not os.path.isfile(xml_path):
    return None

  try:
    with open(xml_path, "r", errors="replace") as f:
      content = f.read()
    clean_content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
    root = ET.fromstring(clean_content)
  except (ET.ParseError, OSError):
    return None

  testsuites = []
  if root.tag == "testsuite":
    testsuites.append(root)
  else:
    testsuites.extend(root.findall(".//testsuite"))

  if not testsuites:
    if root.findall(".//testcase") or root.tag == "testsuites":
      testsuites.append(root)
    else:
      return None

  total_tests = 0
  total_failures = 0
  total_errors = 0
  total_skipped = 0
  total_time = 0.0
  error_msg = ""

  for suite in testsuites:
    testcases = suite.findall(".//testcase")
    tc_count = len(testcases)

    tc_failures = sum(1 for tc in testcases if tc.find("failure") is not None)
    tc_errors = sum(1 for tc in testcases if tc.find("error") is not None)
    tc_skipped = sum(
        1
        for tc in testcases
        if tc.find("skipped") is not None or tc.find("disabled") is not None
    )
    tc_time = sum(_safe_float(tc.attrib.get("time"), 0.0) for tc in testcases)

    attr_tests = _safe_int(suite.attrib.get("tests"), default=None)
    if attr_tests is not None and attr_tests >= 0:
      suite_tests = (
          tc_count if (attr_tests == 0 and tc_count > 0) else attr_tests
      )
    else:
      suite_tests = tc_count if tc_count > 0 else 1

    attr_failures = _safe_int(suite.attrib.get("failures"), default=None)
    if attr_failures is not None and attr_failures >= 0:
      suite_failures = max(attr_failures, tc_failures)
    else:
      suite_failures = tc_failures
    if suite_failures == 0 and suite.find(".//failure") is not None:
      suite_failures = 1

    attr_errors = _safe_int(suite.attrib.get("errors"), default=None)
    if attr_errors is not None and attr_errors >= 0:
      suite_errors = max(attr_errors, tc_errors)
    else:
      suite_errors = tc_errors
    if suite_errors == 0 and suite.find(".//error") is not None:
      suite_errors = 1

    attr_skipped = _safe_int(suite.attrib.get("skipped"), default=None)
    if attr_skipped is None:
      attr_skipped = _safe_int(suite.attrib.get("disabled"), default=None)
    if attr_skipped is not None and attr_skipped >= 0:
      suite_skipped = max(attr_skipped, tc_skipped)
    else:
      suite_skipped = tc_skipped

    attr_time = _safe_float(suite.attrib.get("time"), default=None)
    if attr_time is not None and attr_time >= 0.0:
      suite_time = attr_time
    else:
      suite_time = tc_time

    total_tests += suite_tests
    total_failures += suite_failures
    total_errors += suite_errors
    total_skipped += suite_skipped
    total_time += suite_time

    if not error_msg:
      fail_el = suite.find(".//failure")
      if fail_el is not None:
        error_msg = fail_el.attrib.get("message", "").strip()
        if not error_msg and fail_el.text:
          lines = fail_el.text.strip().splitlines()
          if lines:
            error_msg = lines[0].strip()
      if not error_msg:
        err_el = suite.find(".//error")
        if err_el is not None:
          error_msg = err_el.attrib.get("message", "").strip()
          if not error_msg and err_el.text:
            lines = err_el.text.strip().splitlines()
            if lines:
              error_msg = lines[0].strip()

  if (total_failures > 0 or total_errors > 0) and not error_msg:
    error_msg = "Test assertion or execution failure"

  return {
      "tests": total_tests,
      "failures": total_failures,
      "errors": total_errors,
      "skipped": total_skipped,
      "time": total_time,
      "error_message": error_msg,
  }


def parse_session_env(session_env_path: str) -> dict:
  res = {
      "TPU_NAME": "unknown",
      "TPU_ZONE": "europe-west4-b",
      "TPU_PROJECT": "rbe-tpu-oss",
  }
  if not session_env_path or not os.path.isfile(session_env_path):
    return res

  try:
    with open(session_env_path, "r", errors="replace") as f:
      for line in f:
        line = line.strip()
        if line.startswith("export "):
          line = line[7:]
        if "=" in line:
          k, v = line.split("=", 1)
          k = k.strip()
          v = v.strip().strip("\"'")
          if k in res and v:
            res[k] = v
  except OSError:
    pass
  return res


def discover_targets_from_testlogs(testlogs_dir: str) -> list[str]:
  targets = []
  if not os.path.isdir(testlogs_dir):
    return targets

  for root, _, files in os.walk(testlogs_dir):
    if "test.xml" in files or "test.log" in files:
      rel = os.path.relpath(root, testlogs_dir)
      targets.append(testlog_relpath_to_target(rel))

  targets.sort()
  return targets


def discover_targets_from_bazel_log(bazel_log_path: str) -> list[str]:
  targets = []
  if not bazel_log_path or not os.path.isfile(bazel_log_path):
    return targets

  target_regex = re.compile(
      r"^\s*(//[a-zA-Z0-9_\-./]+:[a-zA-Z0-9_\-.]+)\s+(PASSED|FAILED|FLAKY|TIMEOUT|TIMED OUT)"
  )
  try:
    with open(bazel_log_path, "r", errors="replace") as f:
      for line in f:
        m = target_regex.search(line)
        if m:
          t = m.group(1)
          if t not in targets:
            targets.append(t)
  except OSError:
    pass
  return targets


def parse_args():
  parser = argparse.ArgumentParser(
      description="Generate presubmit summary JSON and Markdown reports"
  )
  parser.add_argument(
      "--output-dir",
      required=True,
      help="Directory to deposit presubmit_summary.json and presubmit_report.md",
  )
  parser.add_argument(
      "--bazel-log",
      "--bazel-output",
      dest="bazel_log",
      default="",
      help="Captured Bazel console output log",
  )
  parser.add_argument(
      "--workspace-root",
      default=os.getcwd(),
      help="Root directory of the workspace",
  )
  parser.add_argument(
      "--testlogs-dir",
      default="",
      help="Path to bazel-testlogs directory (default: <workspace-root>/bazel-testlogs)",
  )
  parser.add_argument(
      "--targets-file",
      default="",
      help="File containing list of executed test targets",
  )
  parser.add_argument(
      "--targets",
      default="",
      help="Comma- or space-separated list of targets",
  )
  parser.add_argument(
      "--session-env",
      default="/tmp/tpu_active_session.env",
      help="Path to active TPU session file",
  )
  parser.add_argument(
      "--duration",
      type=float,
      default=0.0,
      help="Execution duration in seconds",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Dry run mode; generate mock/simulated report",
  )
  parser.add_argument(
      "--bazel-exit-code",
      type=int,
      default=0,
      help="Exit code returned by Bazel process",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  session_info = parse_session_env(args.session_env)
  if args.dry_run and session_info["TPU_NAME"] == "unknown":
    session_info["TPU_NAME"] = "simulated-dry-run-vm"

  testlogs_dir = args.testlogs_dir
  if not testlogs_dir:
    testlogs_dir = os.path.join(args.workspace_root, "bazel-testlogs")

  # Resolve target list
  targets = []
  if args.targets_file and os.path.isfile(args.targets_file):
    with open(args.targets_file, "r") as f:
      targets = [line.strip() for line in f if line.strip()]
  elif args.targets:
    raw = args.targets.replace(",", " ")
    targets = [t.strip() for t in raw.split() if t.strip()]
  else:
    targets = discover_targets_from_bazel_log(args.bazel_log)
    if not targets:
      targets = discover_targets_from_testlogs(testlogs_dir)

  target_records = []
  total_targets = len(targets)
  passed_targets = 0
  failed_targets = 0
  skipped_targets = 0

  for target in targets:
    rel_path = target_to_testlog_relpath(target)
    xml_path = os.path.join(testlogs_dir, rel_path, "test.xml")
    log_path = os.path.join(testlogs_dir, rel_path, "test.log")

    if args.dry_run:
      target_records.append({
          "target": target,
          "status": "DRY_RUN",
          "duration_seconds": 0.0,
          "exit_code": 0,
          "tests_count": 1,
          "failures_count": 0,
          "errors_count": 0,
          "skipped_count": 0,
          "log_file": log_path,
          "xml_file": xml_path,
          "error_message": "",
          "failure_details": "",
      })
      passed_targets += 1
      continue

    xml_data = parse_xml_report(xml_path)
    if xml_data is not None:
      has_fail = xml_data["failures"] > 0 or xml_data["errors"] > 0
      status = "FAILED" if has_fail else "PASSED"
      exit_code = 1 if has_fail else 0
      err_msg = xml_data["error_message"]
      diag = extract_diagnostics(log_path) if has_fail else ""

      if has_fail:
        failed_targets += 1
      else:
        passed_targets += 1
      skipped_targets += xml_data["skipped"]

      target_records.append({
          "target": target,
          "status": status,
          "duration_seconds": round(xml_data["time"], 2),
          "exit_code": exit_code,
          "tests_count": xml_data["tests"],
          "failures_count": xml_data["failures"],
          "errors_count": xml_data["errors"],
          "skipped_count": xml_data["skipped"],
          "log_file": log_path,
          "xml_file": xml_path,
          "error_message": err_msg,
          "failure_details": diag,
      })
    else:
      passed_in_log = False
      if args.bazel_log and os.path.isfile(args.bazel_log):
        try:
          with open(args.bazel_log, "r", errors="replace") as bf:
            log_content = bf.read()
            if f"{target} PASSED" in log_content or f"{target} (cached) PASSED" in log_content:
              passed_in_log = True
        except OSError:
          pass

      if passed_in_log or (args.bazel_exit_code == 0):
        passed_targets += 1
        target_records.append({
            "target": target,
            "status": "PASSED",
            "duration_seconds": 0.0,
            "exit_code": 0,
            "tests_count": 1,
            "failures_count": 0,
            "errors_count": 0,
            "skipped_count": 0,
            "log_file": log_path,
            "xml_file": xml_path,
            "error_message": "",
            "failure_details": "",
        })
      else:
        failed_targets += 1
        diag = ""
        if os.path.isfile(log_path):
          diag = extract_diagnostics(log_path)
        elif args.bazel_log and os.path.isfile(args.bazel_log):
          diag = extract_diagnostics(args.bazel_log)

        target_records.append({
            "target": target,
            "status": "BUILD_FAILED",
            "duration_seconds": 0.0,
            "exit_code": 1,
            "tests_count": 1,
            "failures_count": 1,
            "errors_count": 0,
            "skipped_count": 0,
            "log_file": log_path,
            "xml_file": xml_path,
            "error_message": "Action failed before generating test.xml",
            "failure_details": diag,
        })

  overall_status = "PASSED"
  if args.dry_run:
    overall_status = "DRY_RUN"
  elif failed_targets > 0 or args.bazel_exit_code != 0:
    overall_status = "FAILED"

  iso_timestamp = (
      datetime.now(timezone.utc).replace(microsecond=0).isoformat()
  )

  summary = {
      "schema_version": "1.0.0",
      "timestamp": iso_timestamp,
      "duration_seconds": round(args.duration, 2),
      "execution_time": round(args.duration, 2),
      "project": session_info["TPU_PROJECT"],
      "tpu_zone": session_info["TPU_ZONE"],
      "tpu_vm": session_info["TPU_NAME"],
      "status": overall_status,
      "total_targets": total_targets,
      "targets_total": total_targets,
      "passed_targets": passed_targets,
      "targets_passed": passed_targets,
      "failed_targets": failed_targets,
      "targets_failed": failed_targets,
      "skipped_targets": skipped_targets,
      "targets_skipped": skipped_targets,
      "targets": target_records,
  }

  json_path = os.path.join(args.output_dir, "presubmit_summary.json")
  with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

  pass_rate = (passed_targets / total_targets * 100.0) if total_targets else 0.0
  status_badge = "🟢 PASSED"
  if overall_status == "FAILED":
    status_badge = "🔴 FAILED"
  elif overall_status == "DRY_RUN":
    status_badge = "⚪ DRY_RUN"

  md_lines = [
      "# Torch TPU Presubmit Report: v5e Spot Relay",
      "",
      f"**Status**: {status_badge}",
      f"**Timestamp**: {iso_timestamp}",
      f"**Total Duration**: {round(args.duration, 1)}s",
      f"**Project**: `{session_info['TPU_PROJECT']}`",
      f"**Zone**: `{session_info['TPU_ZONE']}`",
      f"**TPU VM**: `{session_info['TPU_NAME']}`",
      "",
      "## Summary Metrics",
      "",
      "| Metric | Value |",
      "| :--- | :--- |",
      f"| Total Targets | {total_targets} |",
      f"| Passed Targets | {passed_targets} |",
      f"| Failed Targets | {failed_targets} |",
      f"| Pass Rate | {pass_rate:.1f}% |",
      f"| Total Execution Time | {round(args.duration, 1)}s |",
      "",
      "## Target Execution Results",
      "",
      "| # | Target | Status | Duration | Tests | Failures | Logs |",
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
  ]

  for idx, rec in enumerate(target_records, 1):
    dur_str = f"{rec['duration_seconds']:.1f}s"
    log_name = os.path.basename(rec["log_file"])
    log_link = f"[{log_name}]({rec['log_file']})"
    md_lines.append(
        f"| {idx} | `{rec['target']}` | {rec['status']} | {dur_str} |"
        f" {rec['tests_count']} | {rec['failures_count']} | {log_link} |"
    )

  failures_exist = any(
      r["status"] not in ["PASSED", "DRY_RUN"] for r in target_records
  )
  if failures_exist:
    md_lines.extend(["", "## Failure Diagnostics", ""])
    for rec in target_records:
      if rec["status"] in ["PASSED", "DRY_RUN"]:
        continue
      md_lines.append(f"### ❌ `{rec['target']}`")
      md_lines.append(f"- **Status**: {rec['status']} (exit code {rec['exit_code']})")
      md_lines.append(f"- **Duration**: {rec['duration_seconds']:.1f}s")
      if rec["error_message"]:
        md_lines.append(f"- **Error Summary**: `{rec['error_message']}`")
      md_lines.append(f"- **Log Path**: `{rec['log_file']}`")
      md_lines.append("")
      md_lines.append("<details open>")
      md_lines.append("<summary>Diagnostic Excerpt</summary>")
      md_lines.append("")
      md_lines.append("```")
      md_lines.append(
          rec["failure_details"] or "No detailed diagnostic excerpt available."
      )
      md_lines.append("```")
      md_lines.append("</details>")
      md_lines.append("")

  md_path = os.path.join(args.output_dir, "presubmit_report.md")
  with open(md_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")

  return 0 if overall_status in ["PASSED", "DRY_RUN"] else 1


if __name__ == "__main__":
  sys.exit(main())
