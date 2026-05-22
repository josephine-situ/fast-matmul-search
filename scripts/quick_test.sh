#!/bin/bash
# Quick validation: should complete in < 30 minutes

set -e

echo "=== Step 1: Validate framework with Strassen ==="
python -u scripts/strassen_baseline.py

echo ""
echo "=== Step 2: Quick batch run (reduced restarts/steps) ==="
python -u -m src.run_experiments --quick

echo ""
echo "Done. Check quick_results/ for output."
