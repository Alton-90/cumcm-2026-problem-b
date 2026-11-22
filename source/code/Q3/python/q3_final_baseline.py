"""Unchanged combined policy exposed as the final same-case control."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/shared'))
from search_policy_q3_combined import run_search, VERSION

METHOD_ID = 'q3_final_combined_control'
