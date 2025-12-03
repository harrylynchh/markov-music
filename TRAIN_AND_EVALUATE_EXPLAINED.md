# Train and Evaluate Script: Complete Line-by-Line Explanation

This document provides an extremely detailed explanation of every line in `train_and_evaluate.py`, the main script that orchestrates the complete training and evaluation pipeline. This will help you understand how everything fits together and write about it in your paper.

---

## Table of Contents
1. [Imports and Setup](#imports-and-setup)
2. [Function 1: `train_test_split()`](#function-1-train_test_split)
3. [Function 2: `evaluate_model()`](#function-2-evaluate_model)
4. [Function 3: `print_metrics()`](#function-3-print_metrics)
5. [Function 4: `main()`](#function-4-main)
6. [Writing About This in Your Paper](#writing-about-this-in-your-paper)

---

## Imports and Setup

### Lines 1-13: Module Header

```python
"""
Main Training and Evaluation Script

This script ties everything together:
1. Loads MIDI data from datasets
2. Splits into training and validation sets
3. Trains Markov chain models (first-order and higher-order)
4. Evaluates models using metrics like NLL
5. Generates new music samples
"""
```

**What it is**: Documentation explaining the script's purpose.

**For your paper**: "The training and evaluation pipeline integrates data loading, model training, evaluation, and music generation into a single automated workflow."

---

### Line 15: Argparse Import

```python
import argparse
```

**What it does**: Imports Python's argument parser for command-line interfaces.

**Why we need it**: Allows users to specify options via command-line arguments (e.g., `--dataset nottingham --order 1`).

**For your paper**: "Command-line argument parsing enables flexible configuration of training parameters without code modification."

---

### Line 16: OS Import

```python
import os
```

**What it does**: Imports operating system interface for file/directory operations.

**Why we need it**: 
- `os.makedirs()`: Create directories
- `os.path.join()`: Build file paths

**For your paper**: "File system operations handle output directory creation and path management."

---

### Line 17: NumPy Import

```python
import numpy as np
```

**What it does**: Imports NumPy for numerical operations.

**Why we need it**:
- `np.random.seed()`: Set random seed for reproducibility
- `np.random.permutation()`: Shuffle sequences
- `np.mean()`: Calculate averages

**For your paper**: "Numerical operations including random number generation and statistical calculations use NumPy."

---

### Lines 19-21: Local Imports

```python
from midi_loader import load_dataset
from markov_chain import MarkovChain
from midi_generator import sequence_to_midi, sequences_to_midi
```

**What it does**: Imports functions and classes from other modules in the project.

**Breaking it down**:
- `load_dataset`: Loads and preprocesses MIDI files
- `MarkovChain`: The Markov chain model class
- `sequence_to_midi`: Converts generated sequences to MIDI files

**For your paper**: "The pipeline integrates modular components: data loading, model implementation, and MIDI generation."

---

### Line 22: Playback Import (Optional)

```python
from playback import playback_midi
```

**Note**: This import might not exist in your codebase. If you get an error, you can remove this line or create a simple playback module.

**What it would do**: Play MIDI files automatically (if implemented).

---

## Function 1: `train_test_split()`

### Function Signature (Lines 25-26)

```python
def train_test_split(sequences: List[List], test_ratio: float = 0.2, 
                     random_seed: int = 42) -> Tuple[List[List], List[List]]:
```

**What it does**: Splits sequences into training and validation sets.

**Parameters**:
- `sequences`: All sequences to split
- `test_ratio`: Proportion for validation (0.2 = 20%)
- `random_seed`: Random seed for reproducibility

**Returns**: `(train_sequences, val_sequences)`

**For your paper**: "We split data at the sequence level (not state level) to ensure validation pieces are completely unseen during training, providing a fair evaluation of generalization."

---

### Line 50: Set Random Seed

```python
np.random.seed(random_seed)
```

**What it does**: Sets the random number generator seed.

**Why**: Ensures the same split every time (reproducibility).

**For your paper**: "Random seed ensures deterministic train/validation splits for reproducible experiments."

---

### Line 53: Shuffle Indices

```python
indices = np.random.permutation(len(sequences))
```

**What it does**: Creates a shuffled array of indices.

**Breaking it down**:
- `len(sequences)`: Total number of sequences
- `np.random.permutation()`: Returns shuffled array [0, 1, 2, ..., N-1]
- Example: `[3, 0, 4, 1, 2]` (shuffled)

**Why shuffle**: Randomizes which sequences go to train vs validation.

**For your paper**: "Sequences are randomly shuffled before splitting to avoid systematic biases from dataset ordering."

---

### Line 54: Calculate Split Point

```python
split_idx = int(len(sequences) * (1 - test_ratio))
```

**What it does**: Calculates where to split the data.

**Breaking it down**:
- `1 - test_ratio`: Proportion for training (e.g., 1 - 0.2 = 0.8 = 80%)
- `len(sequences) * 0.8`: Number of sequences for training
- `int(...)`: Converts to integer (rounds down)

**Example**: 100 sequences, test_ratio=0.2
- `split_idx = int(100 * 0.8) = 80`
- First 80 → training, last 20 → validation

**For your paper**: "The split point is calculated to allocate the specified proportion of data to validation."

---

### Lines 56-57: Get Indices

```python
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]
```

**What it does**: Splits shuffled indices into train and validation.

**Breaking it down**:
- `indices[:split_idx]`: First `split_idx` elements (training)
- `indices[split_idx:]`: Remaining elements (validation)

**Example**:
```python
indices = [3, 0, 4, 1, 2]  # Shuffled
split_idx = 3
train_indices = [3, 0, 4]  # First 3
val_indices = [1, 2]       # Last 2
```

**For your paper**: "Shuffled indices are partitioned into training and validation sets."

---

### Lines 59-60: Extract Sequences

```python
train_sequences = [sequences[i] for i in train_indices]
val_sequences = [sequences[i] for i in val_indices]
```

**What it does**: Uses indices to extract actual sequences.

**Breaking it down**:
- List comprehension: `[sequences[i] for i in train_indices]`
- For each index in `train_indices`, get `sequences[i]`
- Creates new lists with the selected sequences

**For your paper**: "Sequences are extracted using the partitioned indices, creating separate training and validation datasets."

---

### Line 62: Return Split

```python
return train_sequences, val_sequences
```

**What it does**: Returns the two lists.

**For your paper**: "The function returns separate training and validation sequence lists ready for model training and evaluation."

---

## Function 2: `evaluate_model()`

### Function Signature (Lines 65-87)

```python
def evaluate_model(model: MarkovChain, val_sequences: List[List]) -> dict:
```

**What it does**: Evaluates a trained model on validation data.

**Returns**: Dictionary of evaluation metrics.

**For your paper**: "Model evaluation computes multiple metrics on validation data to assess generalization performance."

---

### Line 88: Print Status

```python
print("\nEvaluating model on validation set...")
```

**What it does**: Prints status message.

**For your paper**: "User feedback is provided during evaluation."

---

### Line 91: Calculate NLL

```python
nll = model.calculate_negative_log_likelihood(val_sequences)
```

**What it does**: Calls the model's NLL calculation method.

**What NLL measures**: How well the model predicts validation data (lower = better).

**For your paper**: "Negative Log-Likelihood (NLL) measures how surprised the model is by validation data, with lower values indicating better prediction."

---

### Lines 94-99: Calculate Per-Sequence Log-Likelihoods

```python
log_likelihoods = []
for seq in val_sequences:
    if len(seq) >= model.order + 1:
        ll = model.calculate_log_likelihood(seq)
        if ll != float('-inf'):
            log_likelihoods.append(ll)
```

**What it does**: Calculates log-likelihood for each sequence individually.

**Breaking it down**:
- `len(seq) >= model.order + 1`: Check if sequence is long enough
- `calculate_log_likelihood(seq)`: Get log-likelihood for this sequence
- `if ll != float('-inf')`: Skip invalid sequences (too short, etc.)
- `log_likelihoods.append(ll)`: Add to list

**Why**: To calculate average log-likelihood (separate from NLL).

**For your paper**: "Per-sequence log-likelihoods are computed to enable statistical analysis of model performance across individual pieces."

---

### Line 101: Calculate Average

```python
avg_log_likelihood = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
```

**What it does**: Calculates average log-likelihood.

**Breaking it down**:
- `np.mean(log_likelihoods)`: Average of all log-likelihoods
- `if log_likelihoods`: Check if list is not empty
- `else float('-inf')`: Fallback if no valid sequences

**For your paper**: "Average log-likelihood provides a summary statistic of model fit across validation sequences."

---

### Lines 104-109: Calculate Coverage

```python
val_states = set()
for seq in val_sequences:
    val_states.update(seq)

seen_states = model.all_states
coverage = len(val_states & seen_states) / len(val_states) if val_states else 0.0
```

**What it does**: Calculates what percentage of validation states were seen during training.

**Breaking it down**:
- `val_states = set()`: Create empty set
- `val_states.update(seq)`: Add all states from each validation sequence
- `seen_states = model.all_states`: States seen during training
- `val_states & seen_states`: Set intersection (states in both)
- `len(...) / len(val_states)`: Percentage

**Example**:
```python
val_states = {60, 62, 64, 66, 68}  # 5 states
seen_states = {60, 62, 64, 65, 67}  # 5 states
intersection = {60, 62, 64}  # 3 states
coverage = 3/5 = 0.6 = 60%
```

**Why it matters**: High coverage means model has seen most validation patterns.

**For your paper**: "State coverage measures the percentage of validation states encountered during training, indicating how well the training data represents the validation distribution."

---

### Lines 111-119: Build Metrics Dictionary

```python
metrics = {
    'nll': nll,
    'avg_log_likelihood': avg_log_likelihood,
    'coverage': coverage,
    'num_val_sequences': len(val_sequences),
    'num_valid_sequences': len(log_likelihoods)
}
```

**What it does**: Packages all metrics into a dictionary.

**For your paper**: "Evaluation metrics are aggregated into a dictionary for reporting and analysis."

---

### Line 119: Return Metrics

```python
return metrics
```

**What it does**: Returns the metrics dictionary.

**For your paper**: "The function returns a comprehensive set of evaluation metrics."

---

## Function 3: `print_metrics()`

### Function Signature (Lines 122-132)

```python
def print_metrics(metrics: dict, model_name: str = "Model"):
```

**What it does**: Pretty-prints evaluation metrics.

**For your paper**: "Metrics are formatted and displayed in a human-readable format."

---

### Line 124: Print Header

```python
print(f"\n{'='*50}")
```

**What it does**: Prints a line of 50 equals signs.

**Breaking it down**:
- `'='*50`: String repetition (50 equals signs)
- `f"..."`: f-string formatting

**For your paper**: "Visual formatting improves readability of evaluation results."

---

### Lines 127-131: Print Each Metric

```python
print(f"Negative Log-Likelihood (NLL): {metrics['nll']:.4f}")
print(f"Average Log-Likelihood: {metrics['avg_log_likelihood']:.4f}")
print(f"State Coverage: {metrics['coverage']*100:.2f}%")
print(f"Validation Sequences: {metrics['num_val_sequences']}")
print(f"Valid Sequences (for evaluation): {metrics['num_valid_sequences']}")
```

**What it does**: Prints each metric with formatting.

**Breaking it down**:
- `{metrics['nll']:.4f}`: Format as float with 4 decimal places
- `{metrics['coverage']*100:.2f}%`: Convert to percentage with 2 decimals

**For your paper**: "Metrics are displayed with appropriate precision and formatting."

---

## Function 4: `main()`

### Function Signature (Line 135)

```python
def main():
```

**What it does**: Main function that orchestrates the entire pipeline.

**For your paper**: "The main function coordinates all pipeline stages: argument parsing, data loading, training, evaluation, and generation."

---

### Lines 147-162: Argument Parser Setup

```python
parser = argparse.ArgumentParser(description='Train and evaluate Markov chain music models')
parser.add_argument('--dataset', type=str, default='nottingham',
                   choices=['nottingham', 'pop909'],
                   help='Dataset to use')
parser.add_argument('--order', type=int, default=1,
                   help='Markov chain order (1=first-order, 2=second-order, etc.)')
parser.add_argument('--max_files', type=int, default=None,
                   help='Maximum number of MIDI files to load (None = all)')
parser.add_argument('--include_rhythm', action='store_true',
                   help='Include rhythm (duration) in states')
parser.add_argument('--test_ratio', type=float, default=0.2,
                   help='Proportion of data for validation')
parser.add_argument('--generate_samples', type=int, default=5,
                   help='Number of sample pieces to generate')
parser.add_argument('--output_dir', type=str, default='output',
                   help='Directory for generated MIDI files')
```

**What it does**: Defines all command-line arguments.

**Breaking it down**:
- `argparse.ArgumentParser()`: Creates argument parser
- `add_argument()`: Adds each argument
  - `--dataset`: Dataset choice (nottingham or pop909)
  - `--order`: Markov chain order
  - `--max_files`: Limit number of files
  - `--include_rhythm`: Flag (if present, include rhythm)
  - `--test_ratio`: Validation proportion
  - `--generate_samples`: Number of pieces to generate
  - `--output_dir`: Where to save files

**For your paper**: "Command-line arguments enable flexible configuration of all training parameters without code modification."

---

### Line 164: Parse Arguments

```python
args = parser.parse_args()
```

**What it does**: Parses command-line arguments into an object.

**Usage**: Access via `args.dataset`, `args.order`, etc.

**For your paper**: "Arguments are parsed and stored for use throughout the pipeline."

---

### Lines 167-174: Determine Dataset Path

```python
if args.dataset == 'nottingham':
    data_dir = "data/nottingham_github/MIDI"
    track_name = None  # Use first track
elif args.dataset == 'pop909':
    data_dir = "data/pop909/POP909"
    track_name = "MELODY"  # POP909 has labeled tracks
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
```

**What it does**: Sets dataset-specific paths and track names.

**Why different**: Different datasets have different structures.

**For your paper**: "Dataset-specific paths and track selection are configured based on the selected dataset."

---

### Lines 176-183: Print Configuration

```python
print("="*60)
print("Markov Chain Music Generation - Training Pipeline")
print("="*60)
print(f"Dataset: {args.dataset}")
print(f"Order: {args.order}")
print(f"Include Rhythm: {args.include_rhythm}")
print(f"Max Files: {args.max_files or 'All'}")
print("="*60)
```

**What it does**: Prints configuration summary.

**For your paper**: "Configuration is displayed at pipeline start for transparency and debugging."

---

### Lines 186-199: Step 1 - Load Dataset

```python
print("\n[Step 1] Loading dataset...")
sequences = load_dataset(
    data_dir=data_dir,
    track_name=track_name,
    max_files=args.max_files,
    include_rhythm=args.include_rhythm
)

if not sequences:
    print("Error: No sequences loaded. Check dataset path.")
    return

print(f"Loaded {len(sequences)} sequences")
print(f"Average sequence length: {np.mean([len(s) for s in sequences]):.1f} states")
```

**What it does**: Loads and preprocesses MIDI files.

**Breaking it down**:
- `load_dataset()`: Calls the data loading function
- `if not sequences`: Error handling
- `np.mean([len(s) for s in sequences])`: Calculate average length

**For your paper**: "Dataset loading extracts state sequences from MIDI files, with error handling for empty datasets."

---

### Lines 202-209: Step 2 - Split Data

```python
print("\n[Step 2] Splitting into train/validation sets...")
train_sequences, val_sequences = train_test_split(
    sequences, 
    test_ratio=args.test_ratio,
    random_seed=42
)
print(f"Training sequences: {len(train_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")
```

**What it does**: Splits data into training and validation.

**For your paper**: "Data is split into training (80%) and validation (20%) sets using a fixed random seed for reproducibility."

---

### Lines 212-214: Step 3 - Train Model

```python
print(f"\n[Step 3] Training {args.order}-order Markov chain...")
model = MarkovChain(order=args.order)
model.train(train_sequences)
```

**What it does**: Creates and trains the model.

**Breaking it down**:
- `MarkovChain(order=args.order)`: Create model with specified order
- `model.train(train_sequences)`: Train on training data

**For your paper**: "The Markov chain model is instantiated and trained on the training sequences, learning transition probabilities from the data."

---

### Lines 217-219: Step 4 - Evaluate Model

```python
print("\n[Step 4] Evaluating model...")
metrics = evaluate_model(model, val_sequences)
print_metrics(metrics, f"{args.order}-order Markov Chain")
```

**What it does**: Evaluates model and prints metrics.

**For your paper**: "Model evaluation computes metrics on validation data and displays results."

---

### Lines 222-239: Step 5 - Generate Music

```python
print(f"\n[Step 5] Generating {args.generate_samples} sample pieces...")
os.makedirs(args.output_dir, exist_ok=True)

generated_sequences = []
for i in range(args.generate_samples):
    # Generate a sequence of similar length to training data
    avg_length = int(np.mean([len(s) for s in train_sequences]))
    generated = model.generate(length=avg_length, temperature=1.0)
    generated_sequences.append(generated)
    
    # Save individual MIDI file
    output_path = os.path.join(
        args.output_dir, 
        f"{args.dataset}_order{args.order}_sample{i+1}.mid"
    )
    sequence_to_midi(generated, output_path)
    print(f"  Generated sample {i+1}: {len(generated)} states -> {output_path}")
    playback_midi(output_path)
```

**What it does**: Generates multiple music samples and saves them.

**Breaking it down**:
- `os.makedirs(args.output_dir, exist_ok=True)`: Create output directory
- `for i in range(args.generate_samples)`: Loop for each sample
- `avg_length = int(np.mean([len(s) for s in train_sequences]))`: Calculate average length
- `model.generate(length=avg_length, temperature=1.0)`: Generate sequence
- `sequence_to_midi(generated, output_path)`: Convert to MIDI
- `playback_midi(output_path)`: Play MIDI (if implemented)

**For your paper**: "Generation creates multiple samples with lengths matching training data, saving each as a separate MIDI file."

---

### Lines 241-244: Save Model

```python
# Save model
model_path = os.path.join(args.output_dir, f"model_order{args.order}.pkl")
model.save(model_path)
print(f"\nModel saved to {model_path}")
```

**What it does**: Saves the trained model to disk.

**For your paper**: "The trained model is serialized to disk for later use without retraining."

---

### Lines 246-255: Print Summary

```python
print("\n" + "="*60)
print("Training and generation complete!")
print("="*60)
print(f"\nGenerated files are in: {args.output_dir}")
print(f"Model file: {model_path}")
print("\nYou can now:")
print("  1. Listen to the generated MIDI files")
print("  2. Compare different orders (run with --order 1, 2, 3)")
print("  3. Compare with/without rhythm (--include_rhythm flag)")
print("  4. Use the saved model to generate more music later")
```

**What it does**: Prints completion message and next steps.

**For your paper**: "Pipeline completion is reported with file locations and suggested next steps."

---

### Lines 258-259: Script Entry Point

```python
if __name__ == "__main__":
    main()
```

**What it does**: Runs `main()` when script is executed directly.

**Why**: Allows importing the file as a module without running main.

**For your paper**: "The script is designed to be both executable and importable as a module."

---

## Writing About This in Your Paper

### Methodology Section: Training and Evaluation Pipeline

**Suggested text:**

> "We implement a complete training and evaluation pipeline that automates the entire workflow from data loading through music generation. The pipeline consists of five main stages:
> 
> **1. Data Loading**: MIDI files are loaded and preprocessed into state sequences using the dataset loading module. Configuration options include dataset selection (Nottingham or POP909), maximum number of files, and whether to include rhythmic information.
> 
> **2. Train/Validation Split**: Sequences are randomly shuffled and split at the sequence level (80% training, 20% validation) using a fixed random seed (42) to ensure reproducibility. This ensures validation pieces are completely unseen during training.
> 
> **3. Model Training**: The Markov chain model is instantiated with the specified order parameter and trained on the training sequences, learning transition probabilities from the data.
> 
> **4. Model Evaluation**: The trained model is evaluated on validation sequences using multiple metrics: Negative Log-Likelihood (NLL), average log-likelihood, and state coverage (percentage of validation states seen during training).
> 
> **5. Music Generation**: Multiple music samples are generated with lengths matching the average training sequence length. Each sample is saved as a separate MIDI file, and the trained model is serialized for later use."

### Evaluation Metrics

> "We evaluate models using three complementary metrics:
> 
> - **Negative Log-Likelihood (NLL)**: Measures how well the model predicts validation sequences. Lower NLL indicates better fit. NLL is computed as the negative average log-probability of sequences under the model.
> 
> - **Average Log-Likelihood**: Provides a summary statistic of model fit across individual validation sequences, enabling analysis of performance variation.
> 
> - **State Coverage**: Measures the percentage of validation states that were encountered during training. High coverage (>90%) indicates the training data well represents the validation distribution, while low coverage suggests potential generalization issues."

### Reproducibility

> "All random operations use fixed seeds (random_seed=42) to ensure reproducible results across runs. This includes train/validation splitting and any random sampling during generation. File paths are sorted alphabetically to ensure deterministic processing order."

### Command-Line Interface

> "The pipeline is controlled via command-line arguments, enabling flexible configuration without code modification. Key parameters include dataset selection, Markov chain order, rhythm inclusion, and number of generated samples. This design facilitates systematic experimentation and comparison of different model configurations."

---

This completes the exhaustive explanation of the training and evaluation pipeline!

