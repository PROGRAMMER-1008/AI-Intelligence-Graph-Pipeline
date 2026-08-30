"""
Runs every test module in tests/ and prints a consolidated summary.
Not pytest-based deliberately: see comment in test_http_client.py —
this keeps the test suite runnable with zero extra dependencies
beyond what the pipeline itself already needs, which matters given
this sandbox's restricted pip/network access during development. A
real deployment should run these via pytest (see README.md) for
proper fixtures, parametrization, and CI integration; this runner is
the fast-path equivalent for local iteration.
"""

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_MODULES = [
    "tests.test_http_client",
    "tests.test_ssl_fallback",
    "tests.test_date_normalizer",
    "tests.test_entity_resolver",
    "tests.test_llm_orchestrator",
    "tests.test_groq_model_resolution",
    "tests.test_arxiv_papers",
    "tests.test_news",
    "tests.test_jobs",
    "tests.test_yc_startups",
    "tests.test_csv_export",
    "tests.test_pipeline_integration",
]


def main():
    total_passed = 0
    total_failed = 0
    failures = []

    for module_name in TEST_MODULES:
        print(f"\n{'=' * 60}")
        print(f"  {module_name}")
        print("=" * 60)
        module = importlib.import_module(module_name)
        test_fns = [
            (name, fn) for name, fn in vars(module).items() if name.startswith("test_")
        ]
        for name, fn in test_fns:
            try:
                fn()
                print(f"  PASS: {name}")
                total_passed += 1
            except Exception as e:
                print(f"  FAIL: {name} -> {e}")
                traceback.print_exc()
                total_failed += 1
                failures.append(f"{module_name}.{name}")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    main()
