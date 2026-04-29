#!/bin/bash
# Full overnight run. Expect 8-24 hours depending on hardware.

set -e

echo "Starting full pipeline at $(date)"
echo "This will take many hours. Results saved incrementally."

# Run the batch
python -u -m src.run_experiments --output-dir results/full_$(date +%Y%m%d)

echo "Completed at $(date)"