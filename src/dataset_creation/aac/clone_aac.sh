#!/bin/bash

# Default output directory
OUTPUT_DIR="../../../data/aac"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output-dir <directory>]"
            exit 1
            ;;
    esac
done

# Run scrape.py with optional output directory
if [ -n "$OUTPUT_DIR" ]; then
    python scrape.py --output-dir "$OUTPUT_DIR"
else
    python scrape.py
fi
