# Experiment Commands for Results Section

This document provides all the commands you need to run to generate results for your paper.

## Answer to Your Question

**Can the model use both datasets at once?**
- **Current implementation**: No, it loads one dataset at a time (either Nottingham or POP909)
- **You CAN combine them**: Load both separately and concatenate sequences, but the command-line interface doesn't do this automatically
- **For your paper**: It's better to train separate models on each dataset to compare their characteristics

---

## Quick Start: Run All Experiments

```bash
# Make the script executable
chmod +x run_experiments.sh

# Run all experiments (this will take a while!)
./run_experiments.sh
```

---

## Individual Commands for Each Section

### 1. Model Performance - Nottingham Dataset

#### Order 1, Pitch Only
```bash
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order1_pitch_only
```

#### Order 1, With Rhythm
```bash
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --include_rhythm \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order1_with_rhythm
```

#### Order 3, Pitch Only
```bash
python train_and_evaluate.py \
    --dataset nottingham \
    --order 3 \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order3_pitch_only
```

#### Order 3, With Rhythm
```bash
python train_and_evaluate.py \
    --dataset nottingham \
    --order 3 \
    --include_rhythm \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order3_with_rhythm
```

#### Order 5, Pitch Only
```bash
python train_and_evaluate.py \
    --dataset nottingham \
    --order 5 \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order5_pitch_only
```

#### Order 5, With Rhythm
```bash
python train_and_evaluate.py \
    --dataset nottingham \
    --order 5 \
    --include_rhythm \
    --max_files 200 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/nottingham/order5_with_rhythm
```

---

### 2. Model Performance - POP909 Dataset

#### Order 1, Pitch Only
```bash
python train_and_evaluate.py \
    --dataset pop909 \
    --order 1 \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order1_pitch_only
```

#### Order 1, With Rhythm
```bash
python train_and_evaluate.py \
    --dataset pop909 \
    --order 1 \
    --include_rhythm \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order1_with_rhythm
```

#### Order 3, Pitch Only
```bash
python train_and_evaluate.py \
    --dataset pop909 \
    --order 3 \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order3_pitch_only
```

#### Order 3, With Rhythm
```bash
python train_and_evaluate.py \
    --dataset pop909 \
    --order 3 \
    --include_rhythm \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order3_with_rhythm
```

#### Order 5, Pitch Only
```bash
python train_and_evaluate.py \
    --dataset pop909 \
    --order 5 \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order5_pitch_only
```

#### Order 5, With Rhythm
```bash
python train_and_evaluate.py \
    --dataset pop909 \
    --order 5 \
    --include_rhythm \
    --max_files 100 \
    --test_ratio 0.2 \
    --generate_samples 5 \
    --output_dir results/pop909/order5_with_rhythm
```

---

### 3. Comparisons - Order Comparison (Nottingham)

Generate samples with different orders for comparison:

```bash
# Order 1
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --include_rhythm \
    --max_files 200 \
    --generate_samples 3 \
    --output_dir results/comparisons/nottingham_order1

# Order 3
python train_and_evaluate.py \
    --dataset nottingham \
    --order 3 \
    --include_rhythm \
    --max_files 200 \
    --generate_samples 3 \
    --output_dir results/comparisons/nottingham_order3

# Order 5
python train_and_evaluate.py \
    --dataset nottingham \
    --order 5 \
    --include_rhythm \
    --max_files 200 \
    --generate_samples 3 \
    --output_dir results/comparisons/nottingham_order5
```

---

### 4. Comparisons - With/Without Rhythm (Nottingham)

```bash
# With Rhythm
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --include_rhythm \
    --max_files 200 \
    --generate_samples 3 \
    --output_dir results/comparisons/nottingham_with_rhythm

# Without Rhythm (Pitch Only)
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --max_files 200 \
    --generate_samples 3 \
    --output_dir results/comparisons/nottingham_pitch_only
```

---

### 5. Human Testing - Nottingham Generated vs Real

For human testing, you'll need:
- **Generated samples**: From your experiments above
- **Real samples**: From the original Nottingham dataset

```bash
# Generate samples for human testing (generate more samples)
python train_and_evaluate.py \
    --dataset nottingham \
    --order 1 \
    --include_rhythm \
    --max_files 200 \
    --generate_samples 10 \
    --output_dir results/human_testing/generated

# Copy some real Nottingham MIDI files for comparison
mkdir -p results/human_testing/real
# Manually copy 10 real MIDI files from data/nottingham_github/MIDI to results/human_testing/real
```

---

## Generating Results Tables

After running all experiments, collect the metrics and create tables:

```bash
# Generate table templates
python generate_results_tables.py
```

This creates `results/results_tables.txt` with LaTeX and Markdown table templates.

**You'll need to manually fill in the values** from the experiment outputs. Look for these lines in the console output:
```
Negative Log-Likelihood (NLL): XXX.XXXX
Average Log-Likelihood: -XXX.XXXX
State Coverage: XX.XX%
```

---

## Results Directory Structure

After running experiments, you'll have:

```
results/
├── nottingham/
│   ├── order1_pitch_only/
│   │   ├── nottingham_order1_sample1.mid
│   │   ├── nottingham_order1_sample2.mid
│   │   ├── ...
│   │   └── model_order1.pkl
│   ├── order1_with_rhythm/
│   ├── order2_pitch_only/
│   └── order2_with_rhythm/
├── pop909/
│   ├── order1_pitch_only/
│   ├── order1_with_rhythm/
│   ├── order2_pitch_only/
│   └── order2_with_rhythm/
├── comparisons/
│   ├── nottingham_order1/
│   ├── nottingham_order2/
│   ├── nottingham_with_rhythm/
│   └── nottingham_pitch_only/
└── human_testing/
    ├── generated/
    └── real/
```

---

## What to Record for Each Experiment

For your results tables, record:

1. **NLL** (Negative Log-Likelihood) - from console output
2. **Average Log-Likelihood** - from console output
3. **State Coverage** - from console output
4. **Number of Training Sequences** - from console output
5. **Number of Validation Sequences** - from console output
6. **Unique States** - from console output
7. **Total Transitions** - from console output

---

## Suggested Results Tables

### Table 1: Nottingham Dataset Performance

| Configuration | NLL | Avg Log-Likelihood | Coverage | Training Seq | Validation Seq |
|---------------|-----|-------------------|----------|--------------|-----------------|
| Order 1, Pitch Only | | | | | |
| Order 1, With Rhythm | | | | | |
| Order 3, Pitch Only | | | | | |
| Order 3, With Rhythm | | | | | |
| Order 5, Pitch Only | | | | | |
| Order 5, With Rhythm | | | | | |

### Table 2: POP909 Dataset Performance

| Configuration | NLL | Avg Log-Likelihood | Coverage | Training Seq | Validation Seq |
|---------------|-----|-------------------|----------|--------------|-----------------|
| Order 1, Pitch Only | | | | | |
| Order 1, With Rhythm | | | | | |
| Order 3, Pitch Only | | | | | |
| Order 3, With Rhythm | | | | | |
| Order 5, Pitch Only | | | | | |
| Order 5, With Rhythm | | | | | |

### Table 3: Comparison - Order Effect (Nottingham)

| Order | NLL | Avg Log-Likelihood | Coverage |
|-------|-----|-------------------|----------|
| 1 | | | |
| 3 | | | |
| 5 | | | |

### Table 4: Comparison - Rhythm Effect (Nottingham)

| Configuration | NLL | Avg Log-Likelihood | Coverage |
|---------------|-----|-------------------|----------|
| Pitch Only | | | |
| With Rhythm | | | |

---

## Tips

1. **Run experiments in batches**: They take time, so run overnight or in parallel if possible
2. **Save console output**: Redirect output to files for later reference:
   ```bash
   python train_and_evaluate.py ... > results/nottingham/order1_pitch_only/output.log 2>&1
   ```
3. **Use smaller max_files for testing**: Test with `--max_files 10` first to make sure everything works
4. **Human testing**: Create a survey with 10 generated + 10 real samples, randomly ordered

---

## Next Steps After Running Experiments

1. Collect all metrics into tables
2. Listen to generated samples and note observations
3. Set up human testing survey
4. Analyze results and write up findings

Good luck with your experiments!

