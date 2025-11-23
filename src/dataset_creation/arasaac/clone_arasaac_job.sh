#!/bin/bash
#SBATCH --job-name=clone_arasaac
#SBATCH --output=clone_arasaac.out
#SBATCH --error=clone_arasaac.err
#SBATCH --partition=inf-train
#SBATCH --time=12:00:00
#SBATCH --account=dslab_jobs

# Parse command line arguments for output directory
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sbatch $0 [--output-dir <directory>]"
            exit 1
            ;;
    esac
done

echo "Starting clone_arasaac job on $(hostname)"

# Pass output directory to clone_arasaac.sh if specified
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
    bash clone_arasaac.sh --output-dir "$OUTPUT_DIR"
else
    bash clone_arasaac.sh
fi

echo "✅ clone_arasaac.sh completed"
