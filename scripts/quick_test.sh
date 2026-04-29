#!/bin/bash
# Quick validation: should complete in < 5 minutes

set -e

echo "=== Step 1: Validate framework with Strassen ==="
python -u scripts/strassen_baseline.py

echo ""
echo "=== Step 2: Quick search on <2,2,3> rank 11 (known achievable) ==="
python -u -m src.pipeline --case 2,2,3 --rank 11 --quick --output-dir results/quick

echo ""
echo "=== Step 3: Quick search on <2,2,3> rank 10 (should fail) ==="
python -u -m src.pipeline --case 2,2,3 --rank 10 --quick --output-dir results/quick

echo ""
echo "Done. Check results/quick/ for output."