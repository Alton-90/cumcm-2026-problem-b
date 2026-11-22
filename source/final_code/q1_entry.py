"""Q1评委入口：透传到冻结目录中的几何求解命令行。"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "code" / "Q1" / "python" / "q1_geometry.py"


def main() -> None:
    runpy.run_path(str(SOURCE_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
