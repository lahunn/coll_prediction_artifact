#!/bin/bash
"""
Batch run accuracy analysis experiments

Test prediction accuracy changes with training data size under different parameter configurations
"""

# Parameter configurations
THRESHOLDS=(0.1)
SAMPLE_RATES=(1)
QNONCOLL_MULTIPLIERS=(4)
DATA_FOLDER="../trace_files/scene_benchmarks/bit_collision_data"
BASENAME="iiwa_7"
NUM_BENCHMARKS=1
ROBOT_NAME="iiwa"

# Create result directories
mkdir -p result_files
mkdir -p plots

echo "=== Starting Batch Accuracy Analysis Experiments ==="
echo "Number of test configurations: $(( ${#THRESHOLDS[@]} * ${#SAMPLE_RATES[@]} * ${#QNONCOLL_MULTIPLIERS[@]} ))"
echo "Number of benchmarks: $NUM_BENCHMARKS"
echo "Data folder: $DATA_FOLDER"
echo "================================="

rm -f result_files/sphere_results.csv
rm -f result_files/sphere_accuracy_curve.csv

# Counter
total_configs=$(( ${#THRESHOLDS[@]} * ${#SAMPLE_RATES[@]} * ${#QNONCOLL_MULTIPLIERS[@]} ))
current_config=0

# Run all configuration combinations
for threshold in "${THRESHOLDS[@]}"; do
    for sample_rate in "${SAMPLE_RATES[@]}"; do
        for qnoncoll_multiplier in "${QNONCOLL_MULTIPLIERS[@]}"; do
            current_config=$((current_config + 1))
            
            echo ""
            echo "[$current_config/$total_configs] Running configuration: Threshold=$threshold, Sample Rate=$sample_rate, Queue Multiplier=$qnoncoll_multiplier"
            
            # Run simulation
            python prediction_simulation_nDOF_sphere_accuracy_tracking.py \
                $threshold \
                $sample_rate \
                $qnoncoll_multiplier \
                $DATA_FOLDER \
                $BASENAME \
                $NUM_BENCHMARKS \
                $ROBOT_NAME
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Configuration completed"
            else
                echo "  ✗ Configuration failed"
            fi
        done
    done
done

echo ""
echo "=== All simulations completed ==="
echo "Generating analysis results..."

# Run analysis script
echo "Running accuracy analysis..."
python analyze_accuracy_vs_training_size.py \
    result_files/sphere_accuracy_curve.csv \
    plots/accuracy_analysis.png

if [ $? -eq 0 ]; then
    echo "✓ Analysis completed, results saved to plots/accuracy_analysis.png"
else
    echo "✗ Analysis failed"
fi

# Generate detailed visualization
echo "Generating detailed visualization..."
python plot_accuracy_learning_curve.py \
    result_files/sphere_accuracy_curve.csv \
    --output plots/detailed_learning_curves.png \
    --mode aggregated

if [ $? -eq 0 ]; then
    echo "✓ Detailed visualization completed, results saved to plots/detailed_learning_curves.png"
else
    echo "✗ Detailed visualization failed"
fi

echo ""
echo "=== Experiment completed ==="
echo "Result files:"
echo "  - Simulation results: result_files/sphere_results.csv"
echo "  - Accuracy curves: result_files/sphere_accuracy_curve.csv"
echo "  - Analysis charts: plots/accuracy_analysis.png"
echo "  - Detailed curves: plots/detailed_learning_curves.png"
echo ""
echo "Use the following command to view detailed configuration comparison:"
echo "python plot_accuracy_learning_curve.py result_files/sphere_accuracy_curve.csv --mode comparison --configs 0.5,0.1,8 1.0,0.1,8 2.0,0.1,8"