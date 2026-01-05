#!/usr/bin/env python3
"""
Process remaining products: Early, Fraud, Home, Hospital, Maid, PersonalAccident
"""

import subprocess
import sys
from datetime import datetime

# Skip Travel and Car - already done
products = ["Early", "Fraud", "Home", "Hospital", "Maid", "PersonalAccident"]

print("=" * 60)
print(f"Processing {len(products)} Remaining Products")
print("Already completed: Travel, Car")
print("=" * 60)
print()

for i, product in enumerate(products, 1):
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(products)}] Processing: {product}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "embedding_agent.py", "--product", product, "--auto-proceed"],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n[OK] {product} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {product} failed with exit code: {e.returncode}")
    except Exception as e:
        print(f"\n[ERROR] {product} error: {e}")

print("\n" + "=" * 60)
print("All Remaining Products Processed!")
print("=" * 60)

