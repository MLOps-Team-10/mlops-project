import pstats
import sys

# Default to 'training_profile.prof' if no file is specified
profile_file = "training_profile.prof"
if len(sys.argv) > 1:
    profile_file = sys.argv[1]

try:
    p = pstats.Stats(profile_file)
    print(f"Successfully loaded stats from '{profile_file}'")

    print("\n" + "="*40)
    print("Top 20 functions by cumulative time")
    print("="*40)
    p.sort_stats("cumulative").print_stats(20)

    print("\n" + "="*40)
    print("Top 20 functions by internal time")
    print("="*40)
    p.sort_stats("time").print_stats(20)

except FileNotFoundError:
    print(f"Error: The profile data file '{profile_file}' was not found.")
    print("Please run the profiling script first to generate it (e.g., './profile.sh').")

except Exception as e:
    print(f"An error occurred: {e}")

