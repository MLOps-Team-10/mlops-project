#!/bin/bash
# This script runs the PyTorch profiler on the training script.

echo "Starting PyTorch profiling..."
echo "Traces will be saved in the 'logs/profiler' directory."

# Run the profiled training script, passing all script arguments to it.
# This allows overriding hydra configs from the command line.
# Example: ./run_pytorch_profiler.sh training.epochs=1
uv run python src/eurosat_classifier/train_profiled.py "$@"

echo "Profiling complete."
echo "To view the results, launch TensorBoard and point it to the 'logs/profiler' directory."
echo "Example command: uv run tensorboard --logdir logs/profiler"
