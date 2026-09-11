"""
Test runner script to run all unit tests in tests/ across any environment.
Compatible with pytest or standard python execution.
Usage:
    python -m scripts.run_tests
"""
from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_all_tests() -> int:
    tests_dir = ROOT / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))

    total_passed = 0
    total_failed = 0
    failures = []

    print(f"Discovered {len(test_files)} test files in {tests_dir}...\n")

    for tf in test_files:
        module_name = f"tests.{tf.stem}"
        try:
            mod = __import__(module_name, fromlist=["*"])
        except Exception as e:
            print(f"[ERROR] Could not import {module_name}: {e}")
            total_failed += 1
            failures.append((module_name, "import", traceback.format_exc()))
            continue

        test_funcs = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        for name, func in test_funcs:
            try:
                func()
                print(f"  PASS: {module_name}.{name}")
                total_passed += 1
            except Exception as e:
                print(f"  FAIL: {module_name}.{name} -> {e}")
                total_failed += 1
                failures.append((module_name, name, traceback.format_exc()))

    print(f"\n==========================================")
    print(f"TEST SUMMARY: {total_passed} passed, {total_failed} failed (Total: {total_passed + total_failed})")
    print(f"==========================================")

    if failures:
        print("\nFailures detail:")
        for mod, name, tb in failures:
            print(f"\n--- {mod}.{name} ---")
            print(tb)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
