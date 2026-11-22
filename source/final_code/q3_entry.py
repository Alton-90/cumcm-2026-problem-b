"""Q3最终入口：默认仅离线验锁，不会进入模拟器。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q3_DIR = ROOT / "code" / "Q3" / "python"
SHARED_DIR = ROOT / "code" / "shared"
sys.path[:0] = [str(Q3_DIR), str(SHARED_DIR)]

import q3_final_main as _implementation


VERSION = _implementation.VERSION
METHOD_ID = _implementation.METHOD_ID
run_search = _implementation.run_search


def validate() -> dict:
    return _implementation.validate_code_lock()


def main() -> None:
    validate()
    print(json.dumps({"status": "PASS", "question": "Q3", "version": VERSION}, ensure_ascii=False))


if __name__ == "__main__":
    main()
