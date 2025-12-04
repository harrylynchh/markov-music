#!/bin/bash
# Comprehensive experiment script for results section
# Run this to generate all results for your paper

echo "=========================================="
echo "Running All Experiments for Results Section"
echo "=========================================="

# Create results directory
mkdir -p results
mkdir -p results/nottingham
mkdir -p results/pop909
mkdir -p results/comparisons

echo ""
echo "=========================================="
echo "PART 1: Nottingham Dataset Experiments"
echo "=========================================="

# Nottingham - Order 1, pitch only
echo "Nottingham - Order 1, Pitch Only..."
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order1_pitch_only

# Nottingham - Order 1, with rhythm
echo "Nottingham - Order 1, With Rhythm..."
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --include_rhythm \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order1_with_rhythm

# Nottingham - Order 3, pitch only
echo "Nottingham - Order 3, Pitch Only..."
python train_and_evaluate.py \
    --dataset nottingham \
    --order 3 \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order3_pitch_only

# Nottingham - Order 3, with rhythm
echo "Nottingham - Order 3, With Rhythm..."
python train_and_evaluate.py \
    --dataset nottingham \
    --order 3 \
    --include_rhythm \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order3_with_rhythm

# Nottingham - Order 5, pitch only
echo "Nottingham - Order 5, Pitch Only..."
python train_and_evaluate.py \
    --dataset nottingham \
    --order 5 \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order5_pitch_only

# Nottingham - Order 5, with rhythm
echo "Nottingham - Order 5, With Rhythm..."
python train_and_evaluate.py \
    --dataset nottingham \
    --order 5 \
    --include_rhythm \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order5_with_rhythm

echo ""
echo "=========================================="
echo "PART 2: POP909 Dataset Experiments"
echo "=========================================="

# POP909 - Order 1, pitch only
echo "POP909 - Order 1, Pitch Only..."
python train_and_evaluate.py \
    --dataset pop909 \
    --order 1 \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order1_pitch_only

# POP909 - Order 1, with rhythm
echo "POP909 - Order 1, With Rhythm..."
python train_and_evaluate.py \
    --dataset pop909 \
    --order 1 \
    --include_rhythm \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order1_with_rhythm

# POP909 - Order 3, pitch only
echo "POP909 - Order 3, Pitch Only..."
python train_and_evaluate.py \
    --dataset pop909 \
    --order 3 \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order3_pitch_only

# POP909 - Order 3, with rhythm
echo "POP909 - Order 3, With Rhythm..."
python train_and_evaluate.py \
    --dataset pop909 \
    --order 3 \
    --include_rhythm \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order3_with_rhythm

# POP909 - Order 5, pitch only
echo "POP909 - Order 5, Pitch Only..."
python train_and_evaluate.py \
    --dataset pop909 \
    --order 5 \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order5_pitch_only

# POP909 - Order 5, with rhythm
echo "POP909 - Order 5, With Rhythm..."
python train_and_evaluate.py \
    --dataset pop909 \
    --order 5 \
    --include_rhythm \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order5_with_rhythm

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved in results/ directory"
echo "=========================================="

