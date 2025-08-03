#!/bin/bash

# Script to run all R scripts from 01 to 08 and save logs
# Usage: ./run_all_scripts.sh

echo "=== RUNNING ALL R SCRIPTS WITH LOGGING ==="
echo "Start time: $(date)"
echo ""

# Array of script files
scripts=("01_data_loading.R" "02_data_cleaning.R" "03_eda.R" "04_knn_model.R" "05_decision_tree.R" "06_random_forest.R" "07_model_comparison.R" "08_final_summary.R")

# Run each script and save log
for script in "${scripts[@]}"; do
    echo "Running $script..."
    log_file="${script%.R}_log.txt"
    
    # Run script and capture both stdout and stderr
    if Rscript "$script" > "$log_file" 2>&1; then
        echo "✓ $script completed successfully - log saved to $log_file"
    else
        echo "✗ $script failed - check $log_file for errors"
    fi
    echo ""
done

echo "=== ALL SCRIPTS COMPLETED ==="
echo "End time: $(date)"
echo ""
echo "Generated log files:"
ls -la *_log.txt