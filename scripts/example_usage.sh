#!/bin/bash
#
# Example Usage of SAbDab Antibody Processing Pipeline
#
# This script demonstrates various ways to use process_sabdab_antibodies.py
#

set -e  # Exit on error

# Configuration
OUTPUT_DIR="$HOME/antibody_analysis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/process_sabdab_antibodies.py"

# Ensure conda environment is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install conda first."
    exit 1
fi

# ============================================================================
# Example 1: Full Pipeline (All Steps)
# ============================================================================
example_full_pipeline() {
    echo "============================================================================"
    echo "Example 1: Running Full Pipeline"
    echo "============================================================================"
    echo "This will download all structures, organize, cluster, and compute embeddings."
    echo "Expected time: 8-48 hours (GPU: 2-8 hours)"
    echo ""

    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR"
}

# ============================================================================
# Example 2: Resume After Interruption
# ============================================================================
example_resume() {
    echo "============================================================================"
    echo "Example 2: Resuming Pipeline After Interruption"
    echo "============================================================================"
    echo "The script automatically detects the state file and resumes."
    echo ""

    # Just run the same command - it will automatically resume
    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR"
}

# ============================================================================
# Example 3: Skip Structure Download (Use Existing)
# ============================================================================
example_skip_download() {
    echo "============================================================================"
    echo "Example 3: Skip Downloading Structures"
    echo "============================================================================"
    echo "Use this if structures are already downloaded."
    echo ""

    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --skip-download
}

# ============================================================================
# Example 4: Process Specific Categories Only
# ============================================================================
example_specific_categories() {
    echo "============================================================================"
    echo "Example 4: Process Only Human Heavy Kinked Antibodies"
    echo "============================================================================"
    echo "This processes only a subset of categories for faster testing."
    echo ""

    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --categories human_heavy_kinked
}

# ============================================================================
# Example 5: Run Everything Except Embeddings
# ============================================================================
example_no_embeddings() {
    echo "============================================================================"
    echo "Example 5: Data Processing Without Embeddings"
    echo "============================================================================"
    echo "Useful for testing the pipeline or if you only need clustered structures."
    echo "Expected time: 2-8 hours"
    echo ""

    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --skip-embeddings
}

# ============================================================================
# Example 6: Process Multiple Specific Categories
# ============================================================================
example_multiple_categories() {
    echo "============================================================================"
    echo "Example 6: Process Human Heavy and Light Chains"
    echo "============================================================================"
    echo "Process multiple categories in one run."
    echo ""

    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --categories human_heavy_kinked,human_heavy_extended,human_light_all
}

# ============================================================================
# Example 7: Custom Clustering Threshold
# ============================================================================
example_custom_clustering() {
    echo "============================================================================"
    echo "Example 7: Custom Sequence Identity Threshold (90%)"
    echo "============================================================================"
    echo "Use a different clustering threshold (default is 70%)."
    echo ""

    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --min-seq-id 0.9
}

# ============================================================================
# Example 8: More Frequent Embedding Saves
# ============================================================================
example_frequent_saves() {
    echo "============================================================================"
    echo "Example 8: Save Embeddings More Frequently"
    echo "============================================================================"
    echo "Save after every structure instead of every 10 (safer but slower)."
    echo ""

    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        --save-interval 1
}

# ============================================================================
# Example 9: Background Execution with Logging
# ============================================================================
example_background() {
    echo "============================================================================"
    echo "Example 9: Run in Background with Logging"
    echo "============================================================================"
    echo "Useful for long-running jobs on remote servers."
    echo ""

    LOG_FILE="$OUTPUT_DIR/pipeline.log"

    nohup conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "Pipeline running in background (PID: $PID)"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Monitor with: tail -f $LOG_FILE"
    echo "Check status with: ps -p $PID"
    echo "Kill with: kill $PID"
}

# ============================================================================
# Example 10: Test Run (Small Subset)
# ============================================================================
example_test_run() {
    echo "============================================================================"
    echo "Example 10: Test Run (Download Only Summary and First Category)"
    echo "============================================================================"
    echo "Quick test to ensure everything is working."
    echo ""

    TEST_DIR="$OUTPUT_DIR/test_run"

    # Run with skip options and just one category
    conda run -n softalign_env python "$SCRIPT" \
        --output-dir "$TEST_DIR" \
        --categories llama_alpaca_light_all \
        --skip-embeddings

    echo ""
    echo "Test complete! Check $TEST_DIR for results."
}

# ============================================================================
# Main Menu
# ============================================================================

show_menu() {
    echo ""
    echo "============================================================================"
    echo "SAbDab Antibody Processing Pipeline - Example Usage"
    echo "============================================================================"
    echo ""
    echo "Choose an example to run:"
    echo ""
    echo "  1)  Full pipeline (all steps)"
    echo "  2)  Resume after interruption"
    echo "  3)  Skip structure download (use existing)"
    echo "  4)  Process specific category (human heavy kinked)"
    echo "  5)  Run without embeddings"
    echo "  6)  Process multiple categories"
    echo "  7)  Custom clustering threshold (90%)"
    echo "  8)  More frequent saves (every structure)"
    echo "  9)  Background execution with logging"
    echo "  10) Test run (small subset)"
    echo ""
    echo "  0)  Exit"
    echo ""
}

# Run interactive menu if no arguments
if [ $# -eq 0 ]; then
    while true; do
        show_menu
        read -p "Enter choice [0-10]: " choice

        case $choice in
            1) example_full_pipeline ;;
            2) example_resume ;;
            3) example_skip_download ;;
            4) example_specific_categories ;;
            5) example_no_embeddings ;;
            6) example_multiple_categories ;;
            7) example_custom_clustering ;;
            8) example_frequent_saves ;;
            9) example_background ;;
            10) example_test_run ;;
            0) echo "Exiting."; exit 0 ;;
            *) echo "Invalid choice. Please try again." ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Run specific example from command line
    case "$1" in
        full) example_full_pipeline ;;
        resume) example_resume ;;
        skip-download) example_skip_download ;;
        specific) example_specific_categories ;;
        no-embeddings) example_no_embeddings ;;
        multiple) example_multiple_categories ;;
        custom-cluster) example_custom_clustering ;;
        frequent-saves) example_frequent_saves ;;
        background) example_background ;;
        test) example_test_run ;;
        *)
            echo "Usage: $0 [example_name]"
            echo ""
            echo "Available examples:"
            echo "  full, resume, skip-download, specific, no-embeddings,"
            echo "  multiple, custom-cluster, frequent-saves, background, test"
            echo ""
            echo "Or run without arguments for interactive menu."
            exit 1
            ;;
    esac
fi
