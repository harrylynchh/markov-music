# Markov Chain Music Generation - Complete Guide

This guide explains the improved Markov chain model system for generating music. It covers the architecture, how each component works, and how to use it.

## Overview

The system consists of four main modules:

1. **`midi_loader.py`** - Loads and preprocesses MIDI files
2. **`markov_chain.py`** - Implements the Markov chain model
3. **`midi_generator.py`** - Converts generated sequences back to MIDI
4. **`train_and_evaluate.py`** - Main training and evaluation pipeline

## Architecture

### Data Flow

```
MIDI Files → Loader → State Sequences → Markov Chain → Generated Sequences → MIDI Files
```

### Key Concepts

**States**: A state represents a musical event. It can be:
- Just a pitch (integer 0-127): `60` (middle C)
- Pitch + rhythm: `(60, 1.0)` (C with quarter note duration)

**Context**: For an N-th order Markov chain, the context is the previous N states. The model predicts the next state given this context.

**Transition Matrix**: A dictionary mapping contexts to probability distributions over next states.

## Module-by-Module Explanation

### 1. MIDI Loader (`midi_loader.py`)

**Purpose**: Extract musical information from MIDI files and convert it into sequences suitable for training.

#### Key Functions:

**`load_midi_files(data_dir, max_files=None)`**
- Recursively finds all `.mid` files in a directory
- Returns a list of file paths
- `max_files` limits the number for quick testing

**`extract_notes_from_midi(midi_path, track_name=None, quantize=True)`**
- Loads a MIDI file using music21
- Extracts notes from a specific track (or first track)
- Converts each note to `(pitch, start_time, duration)`
- Quantizes durations to standard musical values (0.25, 0.5, 1.0, etc.)

**`preprocess_sequences(notes_data, include_rhythm=True)`**
- Converts temporal note data into a sequence of states
- If `include_rhythm=True`: states are `(pitch, duration)` tuples
- If `include_rhythm=False`: states are just pitch integers
- Sorts by time to ensure correct order

**`load_dataset(data_dir, ...)`**
- Main function to load an entire dataset
- Processes all MIDI files and returns a list of sequences
- Each sequence represents one musical piece

### 2. Markov Chain (`markov_chain.py`)

**Purpose**: Learn transition probabilities from training data and generate new sequences.

#### Class: `MarkovChain`

**Initialization: `MarkovChain(order=1)`**
- `order=1`: First-order (next state depends only on current state)
- `order=2`: Second-order (next state depends on previous 2 states)
- Higher order = more memory, but requires more training data

**Training: `train(sequences)`**
- Takes a list of training sequences
- For each position in each sequence:
  1. Extract context (previous N states)
  2. Count how often each next state follows this context
- Builds a transition matrix: `context → {next_state: count}`

**Generation: `generate(length, start_context=None, temperature=1.0)`**
- Generates a new sequence of specified length
- Process:
  1. Start with initial context (or random from training)
  2. Look up probability distribution for next state
  3. Sample from distribution (with temperature control)
  4. Add sampled state to sequence
  5. Update context and repeat
- `temperature`:
  - `1.0`: Normal probabilities
  - `< 1.0`: More deterministic (follows training data closely)
  - `> 1.0`: More random (more creative but less coherent)

**Evaluation: `calculate_negative_log_likelihood(sequences)`**
- Measures how well the model predicts validation data
- Lower NLL = better model
- Formula: `NLL = -1/N * sum(log P(sequence))`

### 3. MIDI Generator (`midi_generator.py`)

**Purpose**: Convert generated state sequences back into playable MIDI files.

**`sequence_to_midi(sequence, output_path, tempo_bpm=120)`**
- Takes a sequence of states (pitches or `(pitch, duration)` tuples)
- Creates a music21 Stream
- Converts each state to a Note with appropriate pitch and duration
- Writes MIDI file

### 4. Training Pipeline (`train_and_evaluate.py`)

**Purpose**: Complete pipeline from data loading to music generation.

**Main Steps:**
1. Load dataset (Nottingham or POP909)
2. Split into training (80%) and validation (20%)
3. Train Markov chain model
4. Evaluate on validation set (calculate NLL)
5. Generate sample music pieces
6. Save model and MIDI files

## Usage Examples

### Basic Usage

```python
from midi_loader import load_dataset
from markov_chain import MarkovChain
from midi_generator import sequence_to_midi

# Load data
sequences = load_dataset("data/nottingham_github/MIDI", max_files=100, include_rhythm=True)

# Split train/validation
train = sequences[:int(len(sequences) * 0.8)]
val = sequences[int(len(sequences) * 0.8):]

# Train model
model = MarkovChain(order=1)
model.train(train)

# Evaluate
nll = model.calculate_negative_log_likelihood(val)
print(f"NLL: {nll}")

# Generate
generated = model.generate(length=50)
sequence_to_midi(generated, "output.mid")
```

### Command-Line Usage

```bash
# Train first-order model on Nottingham dataset
python train_and_evaluate.py --dataset nottingham --order 1 --max_files 100

# Train second-order model with rhythm
python train_and_evaluate.py --dataset nottingham --order 2 --include_rhythm --max_files 200

# Train on POP909 dataset
python train_and_evaluate.py --dataset pop909 --order 1 --max_files 50

# Generate more samples
python train_and_evaluate.py --dataset nottingham --order 1 --generate_samples 10
```

### Running Examples

```bash
# Run all examples
python example_usage.py

# This will generate several MIDI files demonstrating:
# - Basic usage
# - First-order vs second-order comparison
# - Temperature effects
# - Pitch-only vs pitch+rhythm comparison
```

## Understanding the Code Line-by-Line

### Example: Training a First-Order Chain

```python
# Create a first-order Markov chain
# This means: P(next_state | current_state)
model = MarkovChain(order=1)

# Train on sequences
# For each sequence, the model learns:
# - What states appear
# - How often each state follows each other state
model.train(train_sequences)

# The transition matrix now contains:
# transition_matrix[(state_A,)] = {state_B: count_B, state_C: count_C, ...}
# This means: "When we see state_A, how often does state_B follow? state_C?"
```

### Example: Generation Process

```python
# Generate 50 states
generated = model.generate(length=50)

# Internally, this does:
# 1. Start with a random context from training
#    context = (some_state,)
# 
# 2. For each step:
#    a. Look up: transition_matrix[context]
#       This gives: {next_state_1: prob_1, next_state_2: prob_2, ...}
#    b. Sample from this distribution
#       next_state = random_choice(weighted by probabilities)
#    c. Add to sequence: generated.append(next_state)
#    d. Update context: context = (next_state,)
# 
# 3. Repeat until we have 50 states
```

### Example: Higher-Order Chains

```python
# Second-order chain
model = MarkovChain(order=2)

# Now context is 2 states: (previous_state, current_state)
# Transition matrix: transition_matrix[(state_A, state_B)] = {next_state: count}

# This captures patterns like:
# "After seeing C then D, E is very likely"
# "After seeing C then D, F is rare"
```

## Evaluation Metrics

### Negative Log-Likelihood (NLL)

**What it measures**: How surprised the model is by the validation data.

**Interpretation**:
- Lower NLL = model predicts validation data better
- Higher NLL = model is more surprised by validation data

**Calculation**:
```python
NLL = -1/N * sum(log P(sequence_i))
```

For each sequence, we calculate:
- `P(sequence) = P(state_1) * P(state_2|state_1) * P(state_3|state_2) * ...`
- Take log: `log P(sequence) = sum(log P(state_i | context_i))`
- Average over all sequences and negate

### Coverage

**What it measures**: Percentage of validation states that were seen during training.

**Interpretation**:
- High coverage (>90%): Model has seen most validation states
- Low coverage (<50%): Model encounters many new states in validation

## Tips for Better Results

1. **More Training Data**: More MIDI files = better transition probabilities
2. **Appropriate Order**: 
   - First-order: Good for simple patterns, works with less data
   - Second-order: Captures longer patterns, needs more data
   - Third-order+: Usually overfits unless you have thousands of pieces
3. **Include Rhythm**: Models with rhythm capture both melodic and rhythmic patterns
4. **Temperature Tuning**: 
   - Use temperature < 1.0 for more conservative, training-like music
   - Use temperature > 1.0 for more creative, surprising music
5. **Dataset Choice**:
   - Nottingham: Folk music, simpler structure, good for learning
   - POP909: Pop music, more complex, richer patterns

## Comparison with Your Original Model

### Improvements:

1. **MIDI Loading**: Now loads actual MIDI files instead of CSV chord sequences
2. **Rhythm Support**: Can model both pitch and rhythm, not just chords
3. **Higher-Order**: Supports second-order, third-order, etc. chains
4. **Train/Validation Split**: Proper evaluation methodology
5. **Evaluation Metrics**: NLL, coverage, etc.
6. **Multiple Datasets**: Works with Nottingham, POP909, and others
7. **Better Generation**: Temperature control, proper sampling

### Your Original Model:
- Worked with chord sequences from CSV
- First-order only
- Simple bigram approach

### New Model:
- Works with MIDI files (pitches, rhythms, durations)
- Supports any order
- Proper probability distributions
- Evaluation framework
- More flexible and extensible

## Next Steps

1. **Experiment with Different Orders**: Try order=1, 2, 3 and compare NLL
2. **Compare Datasets**: Train on Nottingham vs POP909 and compare results
3. **Hybrid Models**: Combine multiple models (e.g., one for pitch, one for rhythm)
4. **Evaluation**: Use the generated music in your survey to see if people can tell the difference
5. **Metrics**: Implement additional metrics from your proposal (tonal centroid, IOI variance, etc.)

## Troubleshooting

**"No sequences loaded"**
- Check that data directory path is correct
- Make sure MIDI files exist in that directory
- Check that music21 can parse the MIDI files

**"Model not trained yet"**
- Call `model.train()` before `model.generate()`

**"Very high NLL"**
- Model might not have enough training data
- Try lower order (order=1 instead of order=2)
- Check that validation data is similar to training data

**"Generated music sounds random"**
- Try lower temperature (< 1.0)
- Train on more data
- Check that training sequences are long enough

## File Structure

```
markov-music/
├── midi_loader.py          # Load and preprocess MIDI files
├── markov_chain.py          # Markov chain implementation
├── midi_generator.py        # Convert sequences to MIDI
├── train_and_evaluate.py   # Main training pipeline
├── example_usage.py         # Example scripts
├── first-model.py           # Your original model
└── data/
    ├── nottingham_github/   # Nottingham dataset
    └── pop909/              # POP909 dataset
```

## Questions?

Each function and class has detailed docstrings explaining:
- What it does
- What each parameter means
- What it returns
- How it works internally

Read the code comments for line-by-line explanations!

