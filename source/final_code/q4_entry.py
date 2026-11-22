"""Q4最终入口：默认仅离线验锁，不会进入模拟器。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "code" / "Q4" / "python" / "q4_free_tasks_v1" / "selected_entry.py"
MODULE_NAME = "q4_final_selected_entry"
spec = importlib.util.spec_from_file_location(MODULE_NAME, SOURCE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载Q4最终入口：{SOURCE_PATH}")
_implementation = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = _implementation
spec.loader.exec_module(_implementation)


VERSION = _implementation.VERSION
run_search = _implementation.run_search


def validate() -> dict:
    return _implementation.validate()


def main() -> None:
    validate()
    print(json.dumps({"status": "PASS", "question": "Q4", "version": VERSION}, ensure_ascii=False))


if __name__ == "__main__":
    main()
