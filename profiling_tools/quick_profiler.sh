#!/bin/bash
# This script profiles the main training script using cProfile.

# The output file for the profiling data
PROFILE_OUTPUT="training_profile.prof"

echo "Starting profiling of the training script..."

# Run the training script with cProfile
# We use 'uv run' to ensure we are using the project's environment
uv run python -m cProfile -o $PROFILE_OUTPUT src/eurosat_classifier/train.py "$@"

echo "Profiling complete."
echo "Profiling data saved to $PROFILE_OUTPUT"
echo "To view the stats, use the pstats module with uv run python view_stats.py"
