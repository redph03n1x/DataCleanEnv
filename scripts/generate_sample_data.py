"""
scripts/generate_sample_data.py
================================
Preview generated (dirty, clean) pairs for all 3 tasks.
Run from project root:
  python scripts/generate_sample_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from env.dataset_generator import DatasetGenerator
from env.tasks.task_monday_morning  import TASK_CONFIG as T1
from env.tasks.task_warehouse_merge import TASK_CONFIG as T2
from env.tasks.task_data_lake       import TASK_CONFIG as T3

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)

SEED = 42

for cfg in [T1, T2, T3]:
    print(f"\n{'='*60}")
    print(f"TASK: {cfg.display_name}  (seed={SEED})")
    print(f"{'='*60}")

    gen = DatasetGenerator(cfg, seed=SEED)
    dirty, clean, secondary = gen.generate()

    print(f"\nDIRTY  — shape: {dirty.shape}")
    print(f"  Null counts:\n{dirty.isna().sum().to_string()}")
    print(f"  Duplicates: {dirty.duplicated().sum()}")
    print(f"\nCLEAN  — shape: {clean.shape}")
    print(f"  Null counts:\n{clean.isna().sum().to_string()}")
    print(f"  Duplicates: {clean.duplicated().sum()}")

    if secondary:
        print(f"\nSecondary tables: {list(secondary.keys())}")
        for name, tbl in secondary.items():
            print(f"  {name}: {tbl.shape}")

    print(f"\nFirst 3 dirty rows:")
    print(dirty.head(3).to_string())

print("\n✅ All 3 task datasets generated successfully.")