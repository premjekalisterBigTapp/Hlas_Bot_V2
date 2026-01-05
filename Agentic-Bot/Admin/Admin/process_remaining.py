#!/usr/bin/env python3
"""
Process remaining products after Travel is complete.
This script processes: Car, Early, Fraud, Home, Hospital, Maid, PersonalAccident
"""

import subprocess
import sys
from datetime import datetime

products = ["Car", "Early", "Fraud", "Home", "Hospital", "Maid", "PersonalAccident"]

print("=" * 60)
print(f"Processing {len(products)} Products")
print("=" * 60)
print()

results = {}

for product in products:
    print(f"\n{'=' * 60}")
    print(f"Starting: {product}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}")
    
    try:
        result = subprocess.run(
            [sys.executable, "embedding_agent.py", "--product", product, "--auto-proceed"],
            capture_output=False,
            text=True,
            check=True
        )
        results[product] = "SUCCESS"
        print(f"\n[OK] Completed: {product}")
    except subprocess.CalledProcessError as e:
        results[product] = f"FAILED (Exit Code: {e.returncode})"
        print(f"\n[FAIL] Failed: {product} (Exit Code: {e.returncode})")
    except Exception as e:
        results[product] = f"ERROR: {str(e)}"
        print(f"\n[ERROR] Error: {product} - {e}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for product, status in results.items():
    symbol = "[OK]" if status == "SUCCESS" else "[FAIL]"
    print(f"{symbol} {product}: {status}")

print("\n" + "=" * 60)
print("All Products Processed!")
print("=" * 60)

