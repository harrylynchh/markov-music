# MIDI Preprocessing: Complete Line-by-Line Explanation

This document provides an extremely detailed explanation of every line in `midi_loader.py`, the module that preprocesses MIDI files for Markov chain training. This will help you understand the code deeply and write about it in your paper.

---

## Table of Contents
1. [Imports and Setup](#imports-and-setup)
2. [Function 1: `load_midi_files()`](#function-1-load_midi_files)
3. [Function 2: `extract_notes_from_midi()`](#function-2-extract_notes_from_midi)
4. [Function 3: `quantize_duration()`](#function-3-quantize_duration)
5. [Function 4: `preprocess_sequences()`](#function-4-preprocess_sequences)
6. [Function 5: `load_dataset()`](#function-5-load_dataset)
7. [Writing About This in Your Paper](#writing-about-this-in-your-paper)

---

## Imports and Setup

### Lines 1-18: Module Header and Imports

```python
"""
MIDI Data Loader Module

This module handles loading and preprocessing MIDI files from various datasets.
It extracts musical information (pitches, rhythms, durations) that can be used
to train Markov chain models.
"""
```

**What it is**: A docstring (triple-quoted string) that documents the module.

**Why it matters**: This is Python's way of documenting code. When someone imports this module and types `help(midi_loader)`, they'll see this description.

**For your paper**: Mention that the preprocessing pipeline is modular and well-documented, following software engineering best practices.

---

```python
import os
```

**What it does**: Imports Python's `os` module for operating system interfaces.

**Why we need it**: We use `os.path.join()` to build file paths that work on any operating system (Windows, Mac, Linux). This is important because Windows uses backslashes (`\`) while Mac/Linux use forward slashes (`/`).

**Example**: 
- `os.path.join("data", "nottingham", "MIDI")` → `"data/nottingham/MIDI"` on Mac/Linux
- Same code → `"data\nottingham\MIDI"` on Windows

**For your paper**: "The preprocessing pipeline uses cross-platform file path handling to ensure reproducibility across different operating systems."

---

```python
import glob
```

**What it does**: Imports the `glob` module for file pattern matching.

**Why we need it**: `glob` lets us find files using patterns like `*.mid` (all files ending in `.mid`). The `**` pattern means "search recursively in all subdirectories."

**Example**: `glob.glob("data/**/*.mid", recursive=True)` finds all `.mid` files in `data/` and all its subdirectories.

**For your paper**: "We use recursive file pattern matching to automatically discover all MIDI files in nested directory structures, enabling seamless processing of large datasets like Nottingham (3,089 files) and POP909 (2,898 files)."

---

```python
from typing import List, Tuple, Optional
```

**What it does**: Imports type hints from Python's `typing` module.

**Why we need it**: Type hints make code more readable and help catch errors. They don't affect runtime but are used by IDEs and type checkers.

**Syntax breakdown**:
- `List[str]` means "a list containing strings"
- `Tuple[int, float, float]` means "a tuple with 3 elements: int, float, float"
- `Optional[int]` means "either an int or None"

**Example**: 
```python
def load_midi_files(data_dir: str, max_files: Optional[int] = None) -> List[str]:
```
This says: "This function takes a string and an optional int, and returns a list of strings."

**For your paper**: "The preprocessing code uses type annotations to ensure type safety and improve code maintainability."

---

```python
from music21 import converter, stream, note, chord, instrument
```

**What it does**: Imports specific classes from the `music21` library.

**Why we need it**: `music21` is a powerful Python library for music analysis. We use:
- `converter`: Parses MIDI files into music21 objects
- `stream`: Represents musical scores
- `note`: Represents individual notes
- `chord`: Represents chords
- `instrument`: Represents instruments (we don't use this directly, but it's imported for completeness)

**For your paper**: "We leverage the music21 library (Cuthbert & Ariza, 2010) for MIDI file parsing and musical structure analysis, which provides robust handling of various MIDI formats and musical encodings."

---

```python
import numpy as np
```

**What it does**: Imports NumPy (numerical computing library) with the alias `np`.

**Why we need it**: We use NumPy for numerical operations. The `np` alias is a Python convention.

**For your paper**: "Numerical operations are performed using NumPy for efficient array processing."

---

## Function 1: `load_midi_files()`

### Function Signature (Line 21)

```python
def load_midi_files(data_dir: str, max_files: Optional[int] = None) -> List[str]:
```

**Breaking it down**:
- `def`: Python keyword to define a function
- `load_midi_files`: Function name (descriptive, follows Python naming convention)
- `(data_dir: str, max_files: Optional[int] = None)`: Parameters
  - `data_dir: str`: Required parameter, must be a string
  - `max_files: Optional[int] = None`: Optional parameter (defaults to `None`), can be an int or `None`
- `-> List[str]`: Return type annotation (returns a list of strings)

**For your paper**: "The preprocessing pipeline begins by recursively scanning dataset directories to locate all MIDI files, with an optional parameter to limit the number of files processed for rapid prototyping and testing."

---

### Line 43: Initialize Empty List

```python
midi_files = []
```

**What it does**: Creates an empty list to store file paths.

**Why**: We'll append file paths to this list as we find them.

**For your paper**: "We maintain a list data structure to accumulate discovered MIDI file paths during the recursive directory traversal."

---

### Line 46: Build Search Pattern

```python
pattern = os.path.join(data_dir, "**", "*.mid")
```

**What it does**: Builds a file search pattern.

**Breaking it down**:
- `os.path.join()`: Joins path components using the correct separator for the OS
- `data_dir`: The root directory (e.g., `"data/nottingham_github/MIDI"`)
- `"**"`: Special glob pattern meaning "match zero or more directories"
- `"*.mid"`: Pattern meaning "any file ending in `.mid`"

**Result**: `"data/nottingham_github/MIDI/**/*.mid"` (on Mac/Linux)

**Why `**`**: This allows searching in subdirectories at any depth. Without it, we'd only find files in the immediate directory.

**For your paper**: "We employ recursive glob patterns (`**`) to traverse nested directory structures, enabling discovery of MIDI files organized in hierarchical folder structures typical of large music datasets."

---

### Line 47: Find All Matching Files

```python
all_files = glob.glob(pattern, recursive=True)
```

**What it does**: Finds all files matching the pattern.

**Breaking it down**:
- `glob.glob()`: Function that returns a list of matching file paths
- `pattern`: The pattern we built (e.g., `"data/nottingham_github/MIDI/**/*.mid"`)
- `recursive=True`: Enables recursive searching (required for `**` to work)

**Result**: A list like `["data/nottingham_github/MIDI/waltzes1.mid", "data/nottingham_github/MIDI/waltzes2.mid", ...]`

**For your paper**: "The glob pattern matching algorithm recursively searches all subdirectories, returning a comprehensive list of MIDI file paths for batch processing."

---

### Lines 50-51: Limit Files (Optional)

```python
if max_files:
    all_files = all_files[:max_files]
```

**What it does**: Limits the number of files if `max_files` is specified.

**Breaking it down**:
- `if max_files:`: Checks if `max_files` is truthy (not `None`, not `0`)
- `all_files[:max_files]`: List slicing syntax
  - `[:max_files]` means "take elements from index 0 up to (but not including) `max_files`"
  - Example: `[1, 2, 3, 4, 5][:3]` → `[1, 2, 3]`

**Why**: Useful for testing with a small subset of data before processing everything.

**For your paper**: "An optional file limit parameter enables rapid prototyping and testing on data subsets, reducing computational overhead during development iterations."

---

### Line 53: Add Files to List

```python
midi_files.extend(all_files)
```

**What it does**: Adds all found files to our list.

**Breaking it down**:
- `extend()`: Adds all elements from another list to the current list
- Difference from `append()`: `append([1, 2, 3])` adds the list as one element, `extend([1, 2, 3])` adds each element separately

**Example**:
```python
a = [1, 2]
a.extend([3, 4])  # a is now [1, 2, 3, 4]
a.append([5, 6])  # a is now [1, 2, 3, 4, [5, 6]]
```

**For your paper**: "The discovered file paths are aggregated into a single list structure for subsequent batch processing."

---

### Line 55: Return Sorted List

```python
return sorted(midi_files)
```

**What it does**: Returns the list of files, sorted alphabetically.

**Why sort**: Ensures consistent ordering across runs, which is important for reproducibility.

**For your paper**: "File paths are sorted alphabetically to ensure deterministic processing order, a critical requirement for reproducible experimental results."

---

## Function 2: `extract_notes_from_midi()`

### Function Signature (Lines 58-60)

```python
def extract_notes_from_midi(midi_path: str, 
                            track_name: Optional[str] = None,
                            quantize: bool = True) -> List[Tuple[int, float, float]]:
```

**Breaking it down**:
- `midi_path: str`: Path to a MIDI file
- `track_name: Optional[str] = None`: Optional track name filter (e.g., "MELODY" for POP909)
- `quantize: bool = True`: Whether to quantize durations (default: yes)
- `-> List[Tuple[int, float, float]]`: Returns a list of tuples, each containing (pitch, start_time, duration)

**For your paper**: "The extraction function accepts a MIDI file path, optional track selection, and quantization parameters, returning a structured representation of musical events as (pitch, temporal offset, duration) tuples."

---

### Line 93: Error Handling

```python
try:
```

**What it does**: Starts a try-except block for error handling.

**Why**: MIDI files can be corrupted, malformed, or in unexpected formats. We want to catch errors gracefully rather than crashing.

**For your paper**: "Robust error handling ensures the preprocessing pipeline continues processing even when encountering malformed or incompatible MIDI files, improving the system's resilience to dataset heterogeneity."

---

### Line 95: Parse MIDI File

```python
score = converter.parse(midi_path)
```

**What it does**: Uses music21's converter to parse the MIDI file into a music21 Score object.

**Breaking it down**:
- `converter`: The music21 module that converts between different music formats
- `.parse()`: Method that reads a file and converts it to a music21 object
- `score`: A music21 Score object representing the entire musical piece

**What's in `score`**: The score contains Parts (tracks), which contain Measures, which contain Notes/Chords/Rests.

**For your paper**: "We utilize music21's converter module to parse MIDI files into structured musical objects, which provides abstraction over low-level MIDI byte encoding and handles format variations across datasets."

---

### Line 98: Extract Parts

```python
parts = score.parts if hasattr(score, 'parts') else [score]
```

**What it does**: Gets the parts (tracks) from the score, with a fallback.

**Breaking it down**:
- `hasattr(score, 'parts')`: Checks if the score object has a `parts` attribute
- `score.parts`: If it exists, gets the list of parts (tracks)
- `else [score]`: If not, wraps the score itself in a list (some MIDI files have a single track)

**Why this check**: Different MIDI files have different structures. Some have multiple tracks, some have one.

**For your paper**: "We handle both multi-track and single-track MIDI files by checking for the presence of a parts attribute, ensuring compatibility with diverse MIDI file structures."

---

### Lines 101-111: Track Selection Logic

```python
if track_name:
    for part in parts:
        if track_name in str(part.partName):
            selected_part = part
            break
    else:
        # If track not found, use first part
        selected_part = parts[0]
else:
    # Use first part (usually the melody)
    selected_part = parts[0]
```

**What it does**: Selects which track to extract notes from.

**Breaking it down**:
- `if track_name:`: If a track name was specified...
- `for part in parts:`: Loop through all parts
- `if track_name in str(part.partName):`: Check if the track name contains our search string
  - `str(part.partName)`: Converts the part name to string (handles None cases)
  - `in`: String containment check (e.g., `"MELODY" in "MELODY track"` is True)
- `selected_part = part`: Found it! Save this part
- `break`: Exit the loop early
- `else:`: **This is a for-else!** The `else` block runs if the loop completes without breaking
  - This means we didn't find the track, so use the first part as fallback
- `else:` (outer): If no track name specified, just use the first part

**For-else syntax**: This is a Python feature. The `else` block after a `for` loop runs only if the loop completes normally (no `break`).

**For your paper**: "Track selection employs string matching to identify specific instrument tracks (e.g., 'MELODY' in POP909), with fallback to the first track if the specified track is not found, ensuring robust handling of datasets with varying track naming conventions."

---

### Line 114: Initialize Notes List

```python
notes_data = []
```

**What it does**: Creates an empty list to store extracted note data.

**For your paper**: "We initialize a list structure to accumulate extracted musical events during iteration."

---

### Line 117: Iterate Through Musical Elements

```python
for element in selected_part.flat.notes:
```

**What it does**: Iterates through all notes and chords in the selected track.

**Breaking it down**:
- `selected_part`: The track we selected
- `.flat`: Flattens the hierarchical structure (measures, voices, etc.) into a single stream
- `.notes`: Gets all note-like objects (Notes and Chords)
- `for element in ...`: Iterates through each element

**Why `.flat`**: MIDI files have hierarchical structure (Score → Part → Measure → Note). `.flat` gives us a flat list of all notes in order, which is what we need.

**For your paper**: "We flatten the hierarchical musical structure (score → part → measure → note) into a linear sequence of musical events, preserving temporal ordering while simplifying subsequent processing."

---

### Lines 118-128: Handle Single Notes

```python
if isinstance(element, note.Note):
    # Single note
    pitch = element.pitch.midi  # Convert to MIDI number (0-127)
    start = float(element.offset)  # Start time in beats
    dur = float(element.duration.quarterLength)  # Duration in beats
    
    # Quantize duration to common musical values
    if quantize:
        dur = quantize_duration(dur)
    
    notes_data.append((pitch, start, dur))
```

**What it does**: Extracts information from a single note.

**Breaking it down**:
- `isinstance(element, note.Note)`: Checks if the element is a Note (not a Chord)
- `element.pitch.midi`: Gets the MIDI pitch number (0-127)
  - `element.pitch`: The pitch object (e.g., C4)
  - `.midi`: Converts to MIDI number (e.g., C4 → 60)
- `float(element.offset)`: Gets the start time in beats
  - `element.offset`: The temporal position of the note
  - `float()`: Converts to a float (might be a Fraction in music21)
- `float(element.duration.quarterLength)`: Gets duration in quarter-note beats
  - `element.duration`: Duration object
  - `.quarterLength`: Duration in quarter notes (1.0 = quarter note, 0.5 = eighth note)
- `if quantize:`: If quantization is enabled...
- `dur = quantize_duration(dur)`: Round to nearest standard duration
- `notes_data.append((pitch, start, dur))`: Add the tuple to our list

**MIDI pitch numbers**: 
- 60 = C4 (middle C)
- 61 = C#4
- 62 = D4
- etc.

**For your paper**: "Single notes are extracted as (pitch, temporal_offset, duration) tuples, where pitch is encoded as MIDI note numbers (0-127), temporal offset represents the note's onset time in beats, and duration is measured in quarter-note units. Optional quantization rounds durations to standard musical values (eighth, quarter, half notes, etc.) to reduce state space dimensionality."

---

### Lines 130-141: Handle Chords

```python
elif isinstance(element, chord.Chord):
    # For chords, we'll use the root note (lowest pitch)
    # You could also extract all pitches, but for simplicity we use root
    pitches = [p.midi for p in element.pitches]
    root_pitch = min(pitches)  # Lowest note
    start = float(element.offset)
    dur = float(element.duration.quarterLength)
    
    if quantize:
        dur = quantize_duration(dur)
    
    notes_data.append((root_pitch, start, dur))
```

**What it does**: Extracts information from a chord by using the root (lowest) note.

**Breaking it down**:
- `elif isinstance(element, chord.Chord)`: If it's a chord (not a single note)
- `pitches = [p.midi for p in element.pitches]`: List comprehension!
  - `element.pitches`: List of pitch objects in the chord
  - `for p in element.pitches`: Loop through each pitch
  - `p.midi`: Convert each pitch to MIDI number
  - `[p.midi for p in ...]`: Creates a list of MIDI numbers
  - Example: C major chord (C, E, G) → `[60, 64, 67]`
- `root_pitch = min(pitches)`: Gets the lowest pitch (root note)
  - `min()`: Returns the minimum value
  - Example: `min([60, 64, 67])` → `60` (C)
- Rest is same as single notes

**List comprehension syntax**: `[expression for item in iterable]` is Python's concise way to create lists. It's equivalent to:
```python
pitches = []
for p in element.pitches:
    pitches.append(p.midi)
```

**Why root note**: We simplify chords to single notes for the Markov chain. You could extract all pitches, but that would create a much larger state space.

**For your paper**: "Chords are simplified to their root (lowest) pitch to maintain a manageable state space. This simplification trades harmonic richness for computational tractability, though future work could explore multi-pitch representations."

---

### Lines 145-147: Error Handling

```python
except Exception as e:
    print(f"Error loading {midi_path}: {e}")
    return []
```

**What it does**: Catches any errors and returns an empty list instead of crashing.

**Breaking it down**:
- `except Exception as e:`: Catches any exception (error)
- `e`: The exception object (contains error message)
- `print(f"...")`: f-string syntax for formatted printing
  - `f"Error loading {midi_path}: {e}"` inserts variables into the string
- `return []`: Return empty list (no notes extracted)

**f-string syntax**: `f"text {variable}"` is Python 3.6+ syntax for string formatting. It's equivalent to `"text {}".format(variable)` or `"text " + str(variable)`.

**For your paper**: "Error handling ensures graceful degradation: malformed MIDI files result in empty sequences rather than pipeline failure, allowing batch processing to continue with valid files."

---

## Function 3: `quantize_duration()`

### Function Signature (Line 150)

```python
def quantize_duration(duration: float) -> float:
```

**What it does**: Rounds a duration to the nearest standard musical value.

**For your paper**: "Duration quantization maps continuous temporal values to discrete musical durations, reducing the state space and improving model generalization."

---

### Line 172: Define Standard Durations

```python
common_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
```

**What it does**: Lists common musical durations in quarter-note beats.

**Breaking it down**:
- `0.25`: Sixteenth note (1/4 of a quarter note)
- `0.5`: Eighth note
- `0.75`: Dotted eighth note
- `1.0`: Quarter note
- `1.5`: Dotted quarter note
- `2.0`: Half note
- `3.0`: Dotted half note
- `4.0`: Whole note

**For your paper**: "We quantize durations to eight standard values representing common rhythmic subdivisions: sixteenth notes (0.25), eighth notes (0.5), dotted eighths (0.75), quarter notes (1.0), dotted quarters (1.5), half notes (2.0), dotted halves (3.0), and whole notes (4.0)."

---

### Line 175: Find Closest Duration

```python
closest = min(common_durations, key=lambda x: abs(x - duration))
```

**What it does**: Finds the duration in our list that's closest to the input duration.

**Breaking it down**:
- `min(common_durations, key=...)`: Finds the minimum element, but using a custom comparison
- `key=lambda x: abs(x - duration)`: The comparison function
  - `lambda x: ...`: Anonymous function (function without a name)
  - `x`: The argument (each duration in the list)
  - `abs(x - duration)`: Absolute difference between x and the input duration
  - This means: "find the x that minimizes |x - duration|"

**Lambda syntax**: `lambda arguments: expression` creates a small anonymous function. It's equivalent to:
```python
def distance(x):
    return abs(x - duration)
closest = min(common_durations, key=distance)
```

**Example**: If `duration = 0.6`:
- `abs(0.25 - 0.6) = 0.35`
- `abs(0.5 - 0.6) = 0.1` ← smallest!
- `abs(0.75 - 0.6) = 0.15`
- So `closest = 0.5` (eighth note)

**For your paper**: "Quantization employs nearest-neighbor matching: each duration is mapped to the closest standard duration using L1 distance (absolute difference), ensuring minimal information loss while achieving discrete representation."

---

### Line 176: Return Result

```python
return closest
```

**What it does**: Returns the quantized duration.

**For your paper**: "The quantization function returns the nearest standard duration, completing the discretization process."

---

## Function 4: `preprocess_sequences()`

### Function Signature (Lines 179-180)

```python
def preprocess_sequences(notes_data: List[Tuple[int, float, float]], 
                        include_rhythm: bool = True) -> List[Tuple]:
```

**What it does**: Converts temporal note data into state sequences for Markov chain training.

**For your paper**: "The preprocessing function transforms temporal musical events into discrete state sequences suitable for Markov chain modeling, with optional rhythm inclusion."

---

### Lines 205-206: Empty Check

```python
if not notes_data:
    return []
```

**What it does**: Returns empty list if input is empty.

**Breaking it down**:
- `if not notes_data:`: Checks if `notes_data` is "falsy" (empty list, None, etc.)
- `return []`: Early return of empty list

**For your paper**: "Empty input validation ensures robust handling of edge cases where MIDI files contain no extractable notes."

---

### Line 209: Sort by Time

```python
sorted_notes = sorted(notes_data, key=lambda x: x[1])
```

**What it does**: Sorts notes by start time.

**Breaking it down**:
- `sorted(notes_data, key=...)`: Sorts the list using a custom key
- `key=lambda x: x[1]`: Sort by the second element of each tuple
  - `x` is each tuple: `(pitch, start_time, duration)`
  - `x[1]` is `start_time` (index 1)
- Result: Notes ordered chronologically

**Why sort**: Notes might not be in temporal order in the MIDI file (due to how music21 parses). We need chronological order for the Markov chain.

**For your paper**: "Notes are sorted by temporal offset to ensure chronological ordering, which is essential for Markov chain state transitions that model sequential dependencies."

---

### Lines 211-216: Create State Sequences

```python
if include_rhythm:
    # Create (pitch, duration) tuples
    states = [(pitch, duration) for pitch, start, duration in sorted_notes]
else:
    # Only use pitch
    states = [pitch for pitch, start, duration in sorted_notes]
```

**What it does**: Creates state sequences, optionally including rhythm.

**Breaking it down**:
- `if include_rhythm:`: If rhythm should be included...
- `[(pitch, duration) for pitch, start, duration in sorted_notes]`: List comprehension with tuple unpacking
  - `for pitch, start, duration in sorted_notes`: Unpacks each tuple into three variables
  - `(pitch, duration)`: Creates a new tuple with just pitch and duration (drops start_time)
  - `[...]`: Creates a list of these tuples
- `else:`: If rhythm not included...
- `[pitch for pitch, start, duration in sorted_notes]`: Just extract pitch, ignore start and duration

**Tuple unpacking**: `for pitch, start, duration in sorted_notes` automatically unpacks each tuple:
- `(60, 0.0, 1.0)` → `pitch=60, start=0.0, duration=1.0`

**Example**:
- Input: `[(60, 0.0, 1.0), (62, 1.0, 0.5), (64, 1.5, 1.0)]`
- With rhythm: `[(60, 1.0), (62, 0.5), (64, 1.0)]`
- Without rhythm: `[60, 62, 64]`

**For your paper**: "State sequences are constructed by extracting pitch and optionally duration from each temporal event. When rhythm is included, states are (pitch, duration) tuples, enabling modeling of both melodic and rhythmic patterns. When excluded, states are pitch integers only, focusing the model on melodic contour."

---

### Line 218: Return States

```python
return states
```

**What it does**: Returns the state sequence.

**For your paper**: "The function returns a sequence of states ready for Markov chain training."

---

## Function 5: `load_dataset()`

### Function Signature (Lines 221-224)

```python
def load_dataset(data_dir: str, 
                track_name: Optional[str] = None,
                max_files: Optional[int] = None,
                include_rhythm: bool = True) -> List[List[Tuple]]:
```

**What it does**: Main function that loads and preprocesses an entire dataset.

**For your paper**: "The dataset loading function orchestrates the complete preprocessing pipeline, from file discovery through note extraction to state sequence generation."

---

### Line 256: Print Status

```python
print(f"Loading MIDI files from {data_dir}...")
```

**What it does**: Prints a status message.

**For your paper**: "User feedback is provided throughout processing to monitor pipeline progress."

---

### Line 257: Find MIDI Files

```python
midi_files = load_midi_files(data_dir, max_files)
```

**What it does**: Calls our first function to find all MIDI files.

**For your paper**: "File discovery is delegated to the `load_midi_files()` function, which recursively scans the dataset directory."

---

### Line 260: Initialize Sequences List

```python
all_sequences = []
```

**What it does**: Creates empty list to store all sequences (one per MIDI file).

**For your paper**: "We maintain a list structure to accumulate processed sequences from all MIDI files."

---

### Line 262: Iterate Through Files

```python
for i, midi_path in enumerate(midi_files):
```

**What it does**: Loops through each MIDI file with an index counter.

**Breaking it down**:
- `enumerate(midi_files)`: Returns pairs of (index, item)
  - Example: `enumerate(['a', 'b', 'c'])` → `(0, 'a'), (1, 'b'), (2, 'c')`
- `for i, midi_path in ...`: Unpacks into index `i` and path `midi_path`

**Why enumerate**: We need the index to print progress every 100 files.

**For your paper**: "We iterate through discovered MIDI files using enumeration to track processing progress."

---

### Lines 263-264: Progress Reporting

```python
if (i + 1) % 100 == 0:
    print(f"Processed {i + 1}/{len(midi_files)} files...")
```

**What it does**: Prints progress every 100 files.

**Breaking it down**:
- `(i + 1) % 100 == 0`: Checks if `(i + 1)` is divisible by 100
  - `%` is the modulo operator (remainder after division)
  - Example: `100 % 100 = 0`, `101 % 100 = 1`
- `i + 1`: We add 1 because `i` starts at 0, but we want to show "1, 2, 3..." to users

**For your paper**: "Progress reporting occurs every 100 files to provide user feedback during large-scale batch processing."

---

### Line 266: Extract Notes

```python
notes_data = extract_notes_from_midi(midi_path, track_name)
```

**What it does**: Extracts notes from the current MIDI file.

**For your paper**: "Note extraction is performed for each MIDI file using the `extract_notes_from_midi()` function."

---

### Line 267: Preprocess to States

```python
sequence = preprocess_sequences(notes_data, include_rhythm)
```

**What it does**: Converts notes to state sequence.

**For your paper**: "Extracted notes are converted to state sequences via the `preprocess_sequences()` function."

---

### Lines 269-270: Add Non-Empty Sequences

```python
if sequence:  # Only add non-empty sequences
    all_sequences.append(sequence)
```

**What it does**: Only adds sequences that have at least one state.

**Breaking it down**:
- `if sequence:`: Checks if sequence is truthy (not empty, not None)
- `all_sequences.append(sequence)`: Adds the sequence to our list

**Why check**: Some MIDI files might have no extractable notes (empty, corrupted, etc.).

**For your paper**: "Non-empty sequences are added to the dataset, filtering out files that yielded no extractable musical content."

---

### Line 272: Print Summary

```python
print(f"Successfully loaded {len(all_sequences)} sequences")
```

**What it does**: Prints how many sequences were successfully loaded.

**For your paper**: "A summary statistic reports the total number of successfully processed sequences."

---

### Line 273: Return All Sequences

```python
return all_sequences
```

**What it does**: Returns the list of all sequences (one per MIDI file).

**For your paper**: "The function returns a list of state sequences, where each sequence represents one musical piece from the dataset."

---

## Writing About This in Your Paper

### Abstract/Introduction Section

"To enable Markov chain modeling of musical sequences, we developed a preprocessing pipeline that converts MIDI files into discrete state sequences. The pipeline handles multiple datasets (Nottingham folk music, POP909 pop songs) with varying structures, extracts both pitch and rhythmic information, and quantizes temporal values to standard musical durations."

### Methodology Section

**Data Preprocessing Subsection:**

"Our preprocessing pipeline consists of five main functions that transform raw MIDI files into state sequences suitable for Markov chain training:

1. **File Discovery** (`load_midi_files`): Recursively scans dataset directories using glob pattern matching to locate all MIDI files. This enables batch processing of large datasets (e.g., 3,089 files in Nottingham) without manual file enumeration.

2. **Note Extraction** (`extract_notes_from_midi`): Parses MIDI files using the music21 library, extracting musical events as (pitch, temporal_offset, duration) tuples. The function handles both single notes and chords (simplified to root pitch), supports track selection for multi-track MIDI files, and includes error handling for malformed files.

3. **Duration Quantization** (`quantize_duration`): Maps continuous duration values to eight discrete musical durations (sixteenth notes through whole notes) using nearest-neighbor matching with L1 distance. This reduces state space dimensionality while preserving musically meaningful rhythmic patterns.

4. **Sequence Construction** (`preprocess_sequences`): Converts temporal note data into ordered state sequences. States can be pitch-only integers or (pitch, duration) tuples, enabling modeling of melodic patterns alone or combined melodic-rhythmic patterns.

5. **Dataset Loading** (`load_dataset`): Orchestrates the complete pipeline, processing all MIDI files in a dataset and returning a list of state sequences ready for Markov chain training."

### Technical Details Section

**State Representation:**
- "States are represented as either pitch integers (MIDI note numbers 0-127) or (pitch, duration) tuples, depending on whether rhythmic information is included."
- "Duration quantization reduces the continuous temporal space to eight discrete values, balancing model expressiveness with computational tractability."

**Error Handling:**
- "The pipeline includes robust error handling: malformed MIDI files result in empty sequences rather than pipeline failure, ensuring batch processing continues with valid files."

**Track Selection:**
- "For multi-track MIDI files (e.g., POP909 with separate MELODY, BRIDGE, and PIANO tracks), we support track name matching to extract specific instrument parts."

### Results Section

**Dataset Statistics:**
- "The preprocessing pipeline successfully processed X MIDI files from the Nottingham dataset, extracting Y total musical events."
- "Average sequence length: Z states per piece."

### Discussion Section

**Limitations:**
- "Chord simplification to root pitch loses harmonic information, though this trade-off maintains a manageable state space."
- "Duration quantization may lose subtle rhythmic nuances present in the original music."

**Future Work:**
- "Future improvements could include multi-pitch chord representations, finer-grained duration quantization, or polyphonic (multi-voice) modeling."

---

## Key Python Concepts Explained

### List Comprehensions
```python
[x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```
Creates a list by applying an expression to each element.

### Lambda Functions
```python
lambda x: x * 2  # Anonymous function that doubles its input
```
Small anonymous functions, often used with `map()`, `filter()`, `sorted()`, etc.

### Tuple Unpacking
```python
a, b, c = (1, 2, 3)  # a=1, b=2, c=3
```
Automatically assigns tuple elements to variables.

### F-strings
```python
f"Value: {x}"  # Formats string with variable x
```
Modern Python string formatting (Python 3.6+).

### For-else
```python
for item in items:
    if condition:
        break
else:
    # Runs if loop completes without break
```
The `else` block runs if the loop completes normally (no `break`).

### Type Hints
```python
def func(x: int) -> str:
```
Annotations that document expected types (don't affect runtime).

---

This completes the exhaustive line-by-line explanation of the MIDI preprocessing code!

