from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    failures: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            failures.append(f"{path}: could not load")
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
                print(f"PASS {path.name}::{name}")
            except Exception:
                failures.append(f"{path.name}::{name}\n{traceback.format_exc()}")
                print(f"FAIL {path.name}::{name}")
    if failures:
        print("\n".join(failures))
        return 1
    print("All tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
