# Markov Chain Model: Complete Line-by-Line Explanation

This document provides an extremely detailed explanation of every line in `markov_chain.py`, the core module that implements the Markov chain model for music generation. This will help you understand the code deeply and write about it in your paper.

---

## Table of Contents
1. [Imports and Setup](#imports-and-setup)
2. [Class Definition and Initialization](#class-definition-and-initialization)
3. [Helper Method: `_get_state_sequence()`](#helper-method-_get_state_sequence)
4. [Training Method: `train()`](#training-method-train)
5. [Probability Calculation: `_get_transition_probabilities()`](#probability-calculation-_get_transition_probabilities)
6. [Generation Method: `generate()`](#generation-method-generate)
7. [Evaluation Methods](#evaluation-methods)
8. [Save/Load Methods](#saveload-methods)
9. [Writing About This in Your Paper](#writing-about-this-in-your-paper)

---

## Imports and Setup

### Lines 1-10: Module Header

```python
"""
Markov Chain Model for Music Generation

This module implements a Markov chain that can model musical sequences.
It supports both first-order (next state depends only on current state) and
higher-order (next state depends on multiple previous states) Markov chains.
```

**What it is**: Module documentation explaining what this file does.

**For your paper**: "The Markov chain model is implemented as a Python class that supports first-order and higher-order chains for modeling sequential musical patterns."

---

### Line 12: NumPy Import

```python
import numpy as np
```

**What it does**: Imports NumPy for numerical operations.

**Why we need it**: 
- `np.random.choice()`: Sampling from probability distributions
- `np.random.randint()`: Random integer generation
- `np.log()`, `np.exp()`: Logarithmic operations for temperature scaling
- `np.array()`: Array operations

**For your paper**: "Numerical operations including probability sampling and logarithmic transformations are performed using NumPy."

---

### Line 13: Collections Import

```python
from collections import defaultdict, Counter
```

**What it does**: Imports special dictionary types from Python's collections module.

**Why we need them**:
- `defaultdict`: Dictionary that automatically creates default values for missing keys
- `Counter`: Dictionary subclass for counting hashable objects (we don't actually use Counter, but it's imported)

**defaultdict explained**:
```python
# Normal dict: raises KeyError if key doesn't exist
normal_dict = {}
normal_dict['new_key'] += 1  # Error!

# defaultdict: automatically creates default value
from collections import defaultdict
dd = defaultdict(int)  # Default value is 0
dd['new_key'] += 1  # Works! Creates key with value 0, then adds 1
```

**For your paper**: "We use Python's defaultdict data structure to efficiently build transition matrices without explicit key existence checks."

---

### Line 14: Typing Import

```python
from typing import List, Tuple, Optional, Dict, Any
```

**What it does**: Imports type hints for function signatures.

**Why**: Makes code more readable and helps catch errors. Doesn't affect runtime.

**For your paper**: "Type annotations ensure code clarity and enable static type checking."

---

### Line 15: Pickle Import

```python
import pickle
```

**What it does**: Imports Python's pickle module for serialization.

**Why we need it**: To save trained models to disk and load them later. Pickle converts Python objects to bytes and back.

**For your paper**: "Trained models are serialized using Python's pickle module for persistence and later reuse."

---

## Class Definition and Initialization

### Lines 18-33: Class Definition

```python
class MarkovChain:
    """
    A Markov chain model for generating musical sequences.
    
    This class can learn transition probabilities from training data and
    generate new sequences based on those probabilities.
    """
```

**What it is**: Defines a Python class (object-oriented programming).

**Class vs Function**: A class is a blueprint for creating objects. It can have:
- Attributes (data): `self.order`, `self.transition_matrix`
- Methods (functions): `train()`, `generate()`, etc.

**For your paper**: "The Markov chain is implemented as a class-based model, encapsulating the transition matrix, state tracking, and generation logic."

---

### Lines 35-56: `__init__()` Method

```python
def __init__(self, order: int = 1):
```

**What it does**: Constructor method - called when you create a new `MarkovChain` object.

**Breaking it down**:
- `__init__`: Special method name in Python (double underscores = "dunder" methods)
- `self`: Reference to the object being created (always first parameter)
- `order: int = 1`: Parameter with type hint and default value

**Example usage**:
```python
model = MarkovChain(order=1)  # Creates a first-order chain
model2 = MarkovChain(order=2)  # Creates a second-order chain
```

**For your paper**: "The model is initialized with an order parameter that determines the Markov chain's memory length."

---

### Line 53: Store Order

```python
self.order = order
```

**What it does**: Stores the order as an instance attribute.

**Breaking it down**:
- `self.order`: Attribute of this object
- `order`: The parameter passed in

**What order means**:
- `order=1`: First-order (next state depends only on current state)
- `order=2`: Second-order (next state depends on previous 2 states)
- `order=N`: N-th order (next state depends on previous N states)

**For your paper**: "The order parameter controls the Markov chain's memory: a first-order chain models P(next|current), while a second-order chain models P(next|current, previous), enabling capture of longer-term dependencies."

---

### Line 54: Initialize Transition Matrix

```python
self.transition_matrix = defaultdict(lambda: defaultdict(int))
```

**What it does**: Creates a nested defaultdict structure for storing transition counts.

**Breaking it down**:
- `defaultdict(lambda: defaultdict(int))`: Nested structure
  - Outer dict: keys are contexts (tuples of previous states)
  - Inner dict: keys are next states, values are counts
- `lambda: defaultdict(int)`: Anonymous function that creates a new defaultdict(int) for each new key

**Structure**:
```python
transition_matrix = {
    (C,): {D: 5, E: 3},  # After C, D appears 5 times, E appears 3 times
    (D,): {E: 4, F: 2},  # After D, E appears 4 times, F appears 2 times
    (C, D): {E: 3, F: 1}  # After C then D, E appears 3 times, F appears 1 time
}
```

**Why nested defaultdict**: 
- `transition_matrix[context][next_state] += 1` works even if `context` or `next_state` doesn't exist yet
- Automatically creates new entries with value 0

**For your paper**: "Transition probabilities are stored in a nested dictionary structure, where outer keys represent contexts (previous N states) and inner keys represent possible next states, with values storing transition counts."

---

### Line 55: Initialize State Counts

```python
self.state_counts = defaultdict(int)
```

**What it does**: Stores total counts for each context (how many times we've seen each context).

**Why we need it**: To normalize counts to probabilities later.

**Example**:
```python
state_counts = {
    (C,): 8,      # We've seen context (C,) 8 times total
    (D,): 6,      # We've seen context (D,) 6 times total
    (C, D): 4     # We've seen context (C, D) 4 times total
}
```

**For your paper**: "We maintain separate counts for each context to enable probability normalization during generation."

---

### Line 56: Initialize State Set

```python
self.all_states = set()  # Track all unique states we've seen
```

**What it does**: Creates an empty set to track all unique states.

**Why a set**: Sets automatically handle uniqueness (no duplicates) and fast membership testing.

**What it stores**: All unique states (pitches or (pitch, duration) tuples) seen during training.

**For your paper**: "We maintain a set of all unique states encountered during training for fallback handling of unseen contexts."

---

## Helper Method: `_get_state_sequence()`

### Function Signature (Lines 58-79)

```python
def _get_state_sequence(self, sequence: List, index: int) -> Tuple:
```

**What it does**: Extracts the context (previous N states) for a given position in a sequence.

**Breaking it down**:
- `_get_state_sequence`: The underscore prefix indicates it's a "private" method (internal use)
- `self`: Instance method (belongs to the object)
- `sequence: List`: The full sequence of states
- `index: int`: Current position
- `-> Tuple`: Returns a tuple

**For your paper**: "Context extraction is performed by a helper method that retrieves the previous N states based on the model's order."

---

### Lines 80-82: Extract Context

```python
start_idx = max(0, index - self.order)
context = tuple(sequence[start_idx:index])
```

**What it does**: Gets the previous `order` states before the current position.

**Breaking it down**:
- `index - self.order`: Where to start (e.g., if index=5 and order=2, start at 3)
- `max(0, ...)`: Ensures we don't go negative (can't have negative index)
- `sequence[start_idx:index]`: List slicing - gets elements from start_idx to index (exclusive)
- `tuple(...)`: Converts list to tuple (tuples are hashable, can be dict keys)

**Example (order=2, index=5)**:
```python
sequence = [A, B, C, D, E, F, G]
# start_idx = max(0, 5-2) = 3
# context = tuple(sequence[3:5]) = (D, E)
```

**For your paper**: "Context extraction uses list slicing to retrieve the previous N states, where N is the model's order parameter."

---

### Lines 84-86: Padding for Short Sequences

```python
while len(context) < self.order:
    context = (None,) + context
```

**What it does**: Pads the context with `None` if we don't have enough history (at the beginning of sequences).

**Breaking it down**:
- `len(context) < self.order`: Check if context is too short
- `(None,) + context`: Tuple concatenation
  - `(None,)`: Tuple with one element (comma is required!)
  - `+`: Concatenates tuples
  - Result: `(None, D, E)` if we needed 3 but only had 2

**Why needed**: At the start of a sequence, we might not have `order` previous states yet.

**Example (order=2, index=1)**:
```python
sequence = [A, B, C]
# start_idx = max(0, 1-2) = 0
# context = tuple(sequence[0:1]) = (A,)
# len(context) = 1 < 2, so pad: (None, A)
```

**For your paper**: "Sequences shorter than the model order are padded with None values to maintain consistent context length."

---

### Line 88: Return Context

```python
return context
```

**What it does**: Returns the context tuple.

**For your paper**: "The method returns a tuple representing the context, which serves as a key in the transition matrix."

---

## Training Method: `train()`

### Function Signature (Lines 90-109)

```python
def train(self, sequences: List[List]):
```

**What it does**: Trains the Markov chain by learning transition probabilities from training sequences.

**For your paper**: "Training iterates through all sequences, extracts contexts, and counts state transitions to build the transition matrix."

---

### Line 110: Print Status

```python
print(f"Training {self.order}-order Markov chain on {len(sequences)} sequences...")
```

**What it does**: Prints a status message showing what's being trained.

**For your paper**: "Training progress is reported to provide user feedback during model learning."

---

### Lines 112-115: Reset Model

```python
self.transition_matrix = defaultdict(lambda: defaultdict(int))
self.state_counts = defaultdict(int)
self.all_states = set()
```

**What it does**: Resets all model data structures to empty.

**Why**: Allows retraining the same model object on new data (or prevents issues if train() is called multiple times).

**For your paper**: "Model state is reset before training to ensure clean initialization."

---

### Line 117: Initialize Counter

```python
total_transitions = 0
```

**What it does**: Counter for total number of transitions learned (for reporting).

**For your paper**: "We track the total number of transitions for statistical reporting."

---

### Line 120: Iterate Through Sequences

```python
for seq_idx, sequence in enumerate(sequences):
```

**What it does**: Loops through each training sequence.

**Breaking it down**:
- `enumerate(sequences)`: Returns (index, item) pairs
- `seq_idx`: The index (not used, but available)
- `sequence`: One sequence (list of states)

**For your paper**: "Training processes each sequence independently, extracting transition patterns from each."

---

### Lines 121-123: Skip Short Sequences

```python
if len(sequence) < self.order + 1:
    continue
```

**What it does**: Skips sequences that are too short to have any transitions.

**Why**: For order=2, we need at least 3 states: 2 for context + 1 for next state.

**For your paper**: "Sequences shorter than order+1 states are skipped, as they cannot provide valid transitions for the model."

---

### Lines 125-127: Track All States

```python
for state in sequence:
    self.all_states.add(state)
```

**What it does**: Adds each unique state to the set of all states.

**Why**: Needed for fallback when we encounter unseen contexts.

**For your paper**: "We maintain a set of all unique states encountered during training for handling unseen contexts during generation."

---

### Line 130: Iterate Through Positions

```python
for i in range(self.order, len(sequence)):
```

**What it does**: Loops through positions where we can extract a valid context.

**Breaking it down**:
- `range(self.order, len(sequence))`: Starts at `order`, goes to end
- Why start at `order`? We need `order` previous states, so we can't start earlier

**Example (order=2, sequence length=5)**:
```python
sequence = [A, B, C, D, E]
# i=2: context from [0:2] = (A, B), next = C
# i=3: context from [1:3] = (B, C), next = D
# i=4: context from [2:4] = (C, D), next = E
```

**For your paper**: "We iterate through each position in the sequence, starting from index `order` to ensure sufficient context history."

---

### Line 132: Get Context

```python
context = self._get_state_sequence(sequence, i)
```

**What it does**: Calls the helper method to extract the context.

**For your paper**: "Context is extracted using the helper method, which retrieves the previous N states."

---

### Line 135: Get Next State

```python
next_state = sequence[i]
```

**What it does**: Gets the state that follows the context.

**For your paper**: "The state at position i represents the next state given the context of previous states."

---

### Lines 137-140: Count Transition

```python
self.transition_matrix[context][next_state] += 1
self.state_counts[context] += 1
total_transitions += 1
```

**What it does**: Increments counters for this transition.

**Breaking it down**:
- `self.transition_matrix[context][next_state] += 1`: Increment count for this specific transition
- `self.state_counts[context] += 1`: Increment total count for this context
- `total_transitions += 1`: Increment global counter

**Example**:
```python
# If context=(C,) and next_state=D:
transition_matrix[(C,)][D] += 1  # Count: (C,) -> D happened once more
state_counts[(C,)] += 1  # Total: we've seen context (C,) one more time
```

**For your paper**: "Each transition increments both the specific transition count and the context's total count, building the empirical transition distribution."

---

### Lines 142-144: Print Statistics

```python
print(f"Learned {len(self.transition_matrix)} unique contexts")
print(f"Total transitions: {total_transitions}")
print(f"Unique states: {len(self.all_states)}")
```

**What it does**: Reports training statistics.

**For your paper**: "Training completion reports the number of unique contexts learned, total transitions observed, and unique states encountered."

---

## Probability Calculation: `_get_transition_probabilities()`

### Function Signature (Lines 146-165)

```python
def _get_transition_probabilities(self, context: Tuple) -> Tuple[List, List]:
```

**What it does**: Converts transition counts to probability distributions.

**Returns**: `(possible_next_states, probabilities)` where probabilities sum to 1.0

**For your paper**: "Probability distributions are computed on-the-fly by normalizing transition counts for a given context."

---

### Lines 166-173: Handle Unseen Context

```python
if context not in self.transition_matrix:
    states = list(self.all_states)
    if not states:
        return [], []
    probs = [1.0 / len(states)] * len(states)
    return states, probs
```

**What it does**: If we've never seen this context, return uniform distribution over all states.

**Breaking it down**:
- `context not in self.transition_matrix`: Check if context was never seen
- `list(self.all_states)`: Convert set to list
- `[1.0 / len(states)] * len(states)`: Create list of equal probabilities
  - Example: 3 states → `[0.333, 0.333, 0.333]`

**Why uniform**: We have no information, so assume all states equally likely.

**For your paper**: "Unseen contexts are handled by returning a uniform probability distribution over all states encountered during training, ensuring the model can always generate a next state."

---

### Lines 175-177: Get Counts

```python
next_state_counts = self.transition_matrix[context]
total_count = self.state_counts[context]
```

**What it does**: Retrieves the counts for this context.

**Breaking it down**:
- `next_state_counts`: Dictionary mapping next states to their counts
- `total_count`: Total number of times we've seen this context

**Example**:
```python
# If context=(C,):
next_state_counts = {D: 5, E: 3}  # D appeared 5 times, E appeared 3 times
total_count = 8  # We saw context (C,) 8 times total
```

**For your paper**: "Counts are retrieved from the transition matrix and state counts dictionary."

---

### Lines 179-185: Handle Zero Count

```python
if total_count == 0:
    states = list(self.all_states)
    if not states:
        return [], []
    probs = [1.0 / len(states)] * len(states)
    return states, probs
```

**What it does**: Fallback if total_count is somehow 0 (shouldn't happen, but safety check).

**For your paper**: "Edge cases with zero counts are handled with uniform fallback distributions."

---

### Lines 187-193: Convert Counts to Probabilities

```python
states = []
probs = []

for state, count in next_state_counts.items():
    states.append(state)
    probs.append(count / total_count)
```

**What it does**: Converts counts to probabilities by dividing by total.

**Breaking it down**:
- `next_state_counts.items()`: Iterates through (state, count) pairs
- `count / total_count`: Probability = count / total
- Example: `{D: 5, E: 3}`, total=8 → `P(D)=5/8=0.625`, `P(E)=3/8=0.375`

**For your paper**: "Transition counts are normalized by dividing each count by the total count for the context, yielding probability distributions that sum to 1.0."

---

### Lines 195-198: Normalize Probabilities

```python
total_prob = sum(probs)
if total_prob > 0:
    probs = [p / total_prob for p in probs]
```

**What it does**: Ensures probabilities sum to exactly 1.0 (handles floating-point errors).

**Why needed**: Floating-point arithmetic can introduce tiny errors. This renormalization ensures exact sum.

**For your paper**: "Probabilities are renormalized to account for floating-point precision errors, ensuring they sum to exactly 1.0."

---

### Line 200: Return Distribution

```python
return states, probs
```

**What it does**: Returns the probability distribution.

**For your paper**: "The method returns a tuple of possible next states and their corresponding probabilities."

---

## Generation Method: `generate()`

### Function Signature (Lines 202-237)

```python
def generate(self, length: int, start_context: Optional[Tuple] = None, 
             temperature: float = 1.0) -> List:
```

**What it does**: Generates a new sequence using the trained model.

**For your paper**: "Generation creates new sequences by iteratively sampling from probability distributions conditioned on the current context."

---

### Lines 238-239: Check if Trained

```python
if not self.transition_matrix:
    raise ValueError("Model not trained yet. Call train() first.")
```

**What it does**: Ensures the model has been trained before generation.

**For your paper**: "Generation requires a trained model; an error is raised if training has not been performed."

---

### Line 241: Initialize Output

```python
generated = []
```

**What it does**: Creates empty list to store generated states.

**For your paper**: "The generated sequence is initialized as an empty list."

---

### Lines 244-251: Initialize Context

```python
if start_context is None:
    available_contexts = list(self.transition_matrix.keys())
    if not available_contexts:
        raise ValueError("No training data available")
    context = available_contexts[np.random.randint(len(available_contexts))]
else:
    context = start_context
```

**What it does**: Sets the initial context for generation.

**Breaking it down**:
- If no start context provided: randomly pick a context seen during training
- `self.transition_matrix.keys()`: All contexts we've seen
- `np.random.randint(len(available_contexts))`: Random index
- If start context provided: use it

**For your paper**: "Generation begins with either a randomly selected context from training data or a user-specified starting context."

---

### Line 254: Generation Loop

```python
for _ in range(length):
```

**What it does**: Loops `length` times to generate that many states.

**Breaking it down**:
- `_`: Variable name indicating we don't use the loop variable
- `range(length)`: Creates sequence 0, 1, 2, ..., length-1

**For your paper**: "Generation iterates for the specified number of states."

---

### Line 256: Get Probability Distribution

```python
next_states, probabilities = self._get_transition_probabilities(context)
```

**What it does**: Gets the probability distribution for next state given current context.

**For your paper**: "At each step, the probability distribution for the next state is computed based on the current context."

---

### Lines 258-263: Handle Empty Distribution

```python
if not next_states:
    all_states = list(self.all_states)
    if not all_states:
        break
    next_state = all_states[np.random.randint(len(all_states))]
```

**What it does**: Fallback if no transitions available - pick random state.

**For your paper**: "If no transitions are available for the current context, a random state is selected from all training states as a fallback."

---

### Lines 265-271: Apply Temperature

```python
if temperature != 1.0:
    log_probs = np.log(np.array(probabilities) + 1e-10)
    scaled_log_probs = log_probs / temperature
    probabilities = np.exp(scaled_log_probs)
    probabilities = probabilities / probabilities.sum()
```

**What it does**: Adjusts probabilities using temperature scaling.

**Breaking it down**:
- `temperature = 1.0`: No change (use probabilities as-is)
- `temperature > 1.0`: Flatten probabilities (more random)
- `temperature < 1.0`: Sharpen probabilities (more deterministic)

**How it works**:
1. Convert to log space: `log(P)`
2. Divide by temperature: `log(P) / T`
3. Convert back: `exp(log(P) / T)`
4. Renormalize: divide by sum

**Example**:
```python
# Original: [0.1, 0.3, 0.6]
# Temperature = 2.0 (more random):
#   After scaling: [0.22, 0.39, 0.39]  # More uniform
# Temperature = 0.5 (more deterministic):
#   After scaling: [0.02, 0.18, 0.80]  # More peaked
```

**For your paper**: "Temperature scaling adjusts the randomness of generation: temperature > 1.0 increases exploration (flatter distribution), while temperature < 1.0 increases exploitation (peaked distribution)."

---

### Line 274: Sample Next State

```python
next_state = np.random.choice(next_states, p=probabilities)
```

**What it does**: Samples a state from the probability distribution.

**Breaking it down**:
- `np.random.choice()`: NumPy function for weighted random sampling
- `next_states`: List of possible states
- `p=probabilities`: Probability weights (must sum to 1.0)

**Example**:
```python
next_states = [D, E, F]
probabilities = [0.5, 0.3, 0.2]
# 50% chance of D, 30% chance of E, 20% chance of F
```

**For your paper**: "The next state is sampled from the probability distribution using weighted random selection."

---

### Line 276: Add to Sequence

```python
generated.append(next_state)
```

**What it does**: Adds the sampled state to the generated sequence.

**For your paper**: "The sampled state is appended to the generated sequence."

---

### Lines 278-282: Update Context

```python
if self.order > 0:
    context = context[1:] + (next_state,)
else:
    context = (next_state,)
```

**What it does**: Updates the context by shifting and adding the new state.

**Breaking it down**:
- `context[1:]`: All but first element (shift left)
- `(next_state,)`: Tuple with new state
- `+`: Concatenate tuples

**Example (order=2)**:
```python
# Current context: (C, D)
# Generated: E
# New context: (D, E)  # Shifted C out, added E
```

**For your paper**: "The context is updated by removing the oldest state and adding the newly generated state, maintaining a sliding window of the previous N states."

---

### Line 284: Return Generated Sequence

```python
return generated
```

**What it does**: Returns the complete generated sequence.

**For your paper**: "The method returns the complete generated sequence of states."

---

## Evaluation Methods

### `calculate_log_likelihood()` (Lines 286-332)

**What it does**: Calculates how well the model predicts a sequence.

**Key lines**:
- Line 311: Check if sequence is long enough
- Line 314: Initialize log-likelihood accumulator
- Line 316: Loop through positions
- Line 317: Get context
- Line 321: Get probability distribution
- Line 323-327: If state is in distribution, add log probability
- Line 329-330: If state never seen, add very small probability (penalty)

**Formula**: `log P(sequence) = sum(log P(state_i | context_i))`

**For your paper**: "Log-likelihood measures model fit by summing the log-probability of each state given its context. Higher log-likelihood indicates better prediction."

---

### `calculate_negative_log_likelihood()` (Lines 334-371)

**What it does**: Calculates average NLL over multiple sequences.

**Key lines**:
- Line 357: Initialize accumulators
- Line 360: Loop through sequences
- Line 361: Check if sequence is valid
- Line 362: Calculate log-likelihood
- Line 363-365: If valid, add to total
- Line 370: Calculate average
- Line 371: Return negative (NLL)

**Formula**: `NLL = -1/N * sum(log P(sequence_i))`

**For your paper**: "Negative Log-Likelihood (NLL) averages the negative log-probability across all sequences. Lower NLL indicates better model performance."

---

## Save/Load Methods

### `save()` (Lines 373-382)

**What it does**: Saves the trained model to a file using pickle.

**Key lines**:
- Line 375: Open file in binary write mode
- Line 376-381: Create dictionary with all model data
- Line 377: Convert defaultdict to regular dict (pickle requirement)
- Line 382: Use pickle.dump() to serialize

**For your paper**: "Trained models are serialized to disk using Python's pickle module, enabling model persistence and reuse without retraining."

---

### `load()` (Lines 384-393)

**What it does**: Loads a saved model from a file.

**Key lines**:
- Line 386: Open file in binary read mode
- Line 387: Use pickle.load() to deserialize
- Line 388-392: Restore all model attributes
- Line 389-390: Reconstruct nested defaultdict structure

**For your paper**: "Saved models can be loaded from disk, reconstructing the transition matrix and state information for immediate use in generation."

---

## Writing About This in Your Paper

### Methodology Section: Markov Chain Model

**Suggested structure:**

1. **Model Definition**
   > "We implement a discrete-time Markov chain where each state represents a musical event (pitch or pitch-duration pair). The model learns transition probabilities P(next_state | context) from training sequences, where context consists of the previous N states determined by the model's order parameter."

2. **Training Procedure**
   > "Training iterates through all sequences, extracting contexts of length N and counting transitions to build an empirical transition matrix. For each context, we maintain counts of all observed next states, which are normalized to probabilities during generation."

3. **Generation Algorithm**
   > "Generation begins with a randomly selected context from training data. At each step, the model (1) retrieves the probability distribution for the next state given the current context, (2) applies temperature scaling to adjust randomness, (3) samples a state from the distribution, and (4) updates the context by shifting the window and adding the new state."

4. **Evaluation Metrics**
   > "We evaluate models using Negative Log-Likelihood (NLL), which measures how well the model predicts validation sequences. NLL is computed as the negative average log-probability of sequences under the model, with lower values indicating better fit."

### Key Technical Details

**Transition Matrix Structure:**
> "The transition matrix is implemented as a nested dictionary: outer keys are context tuples (previous N states), inner keys are possible next states, and values are transition counts. This structure enables efficient lookup and automatic handling of new contexts via Python's defaultdict."

**Temperature Scaling:**
> "Temperature scaling adjusts generation randomness by transforming probabilities: P_new = exp(log(P_old) / T). Temperature > 1.0 flattens distributions (more exploration), while temperature < 1.0 sharpens them (more exploitation of learned patterns)."

**Handling Unseen Contexts:**
> "When encountering contexts not seen during training, the model falls back to a uniform distribution over all training states, ensuring generation can always proceed while acknowledging uncertainty."

### Comparison with Other Approaches

**First-Order vs Higher-Order:**
> "First-order chains (order=1) model P(next|current), capturing immediate dependencies with minimal data requirements. Higher-order chains (order≥2) model P(next|current, previous, ...), capturing longer-term patterns but requiring exponentially more training data as order increases."

### Limitations Section

> "The Markov chain model has several limitations: (1) it cannot model long-term structure beyond the order parameter, (2) it assumes stationarity (transition probabilities don't change over time), (3) it requires sufficient training data for each context to learn reliable probabilities, and (4) it cannot explicitly model musical concepts like key, meter, or phrase structure."

---

This completes the exhaustive line-by-line explanation of the Markov chain model code!

