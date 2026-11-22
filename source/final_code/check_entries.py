"""离线检查四问评委入口及其实际依赖。"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "final_code"),
    str(ROOT / "code" / "Q1" / "python"),
    str(ROOT / "code" / "Q2" / "python"),
]

q1 = importlib.import_module("q1_geometry")
q2 = importlib.import_module("select_second_detection_point")
q3 = importlib.import_module("q3_entry")
q4 = importlib.import_module("q4_entry")

q1_result = q1.solve(q1.bearing_halfplanes([(-500, 0, 0), (0, -500, 90)]))
if abs(q1_result["diameter_interval_m"][0] - 24.69271291186713) >= 1e-9:
    raise AssertionError("Q1入口算例不一致")

q2_polygon = q2.initial_outer_polygon((0, 0), 0, circle_sides=360)
if not q2.correlated_reception_certificate((400, 400), q2_polygon, (0, 0))["feasible"]:
    raise AssertionError("Q2入口接收证书失败")

q3.validate()
q4.validate()

print(
    json.dumps(
        {
            "status": "PASS",
            "entries": {
                "Q1": "q1_entry.py",
                "Q2": "q2_entry.py",
                "Q3": {"entry": "q3_entry.py", "version": q3.VERSION},
                "Q4": {"entry": "q4_entry.py", "version": q4.VERSION},
            },
        },
        ensure_ascii=False,
        indent=2,
    )
)
