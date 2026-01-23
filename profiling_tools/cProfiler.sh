#!/bin/bash
# This script profiles the SHORT training script using cProfile.

# The output file for the profiling data
PROFILE_OUTPUT="training_profile.prof"

echo "Starting profiling of the SHORT training script..."

# Run the training script with cProfile
uv run python -m cProfile -o $PROFILE_OUTPUT profiling_tools/train_short.py "$@"

echo "Profiling complete."
echo "Profiling data saved to $PROFILE_OUTPUT"
echo "To view the stats, use the pstats module with uv run python view_stats.py"
