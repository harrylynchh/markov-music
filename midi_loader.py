"""
MIDI Data Loader Module

This module handles loading and preprocessing MIDI files from various datasets.
It extracts musical information (pitches, rhythms, durations) that can be used
to train Markov chain models.

Key Functions:
- load_midi_files(): Scans directories for MIDI files
- extract_notes_from_midi(): Extracts note sequences from a single MIDI file
- preprocess_sequences(): Converts raw MIDI data into sequences suitable for training
"""

import os
import glob
from typing import List, Tuple, Optional
from music21 import converter, stream, note, chord, instrument
import numpy as np


def load_midi_files(data_dir: str, max_files: Optional[int] = None) -> List[str]:
    """
    Recursively finds all MIDI files in a directory.
    
    Parameters:
    -----------
    data_dir : str
        Root directory to search for MIDI files
    max_files : int, optional
        Maximum number of files to load (None = load all)
    
    Returns:
    --------
    List[str]
        List of file paths to MIDI files
    
    Explanation:
    ------------
    This function walks through the directory tree and collects all .mid files.
    It's useful for loading datasets like Nottingham or POP909 that have many
    MIDI files organized in subdirectories.
    """
    midi_files = []
    
    # Use glob to recursively find all .mid files
    pattern = os.path.join(data_dir, "**", "*.mid")
    all_files = glob.glob(pattern, recursive=True)
    
    # Limit the number if specified (useful for testing)
    if max_files:
        all_files = all_files[:max_files]
    
    midi_files.extend(all_files)
    
    return sorted(midi_files)


def extract_notes_from_midi(midi_path: str, 
                            track_name: Optional[str] = None,
                            quantize: bool = True) -> List[Tuple[int, float, float]]:
    """
    Extracts note sequences from a MIDI file.
    
    Parameters:
    -----------
    midi_path : str
        Path to the MIDI file
    track_name : str, optional
        If specified, only extract notes from tracks with this name
        (e.g., "MELODY" for POP909, "Piano" for Nottingham)
    quantize : bool
        If True, quantize note durations to common musical values
    
    Returns:
    --------
    List[Tuple[int, float, float]]
        List of (pitch, start_time, duration) tuples
        - pitch: MIDI note number (0-127)
        - start_time: Start time in beats
        - duration: Duration in beats
    
    Explanation:
    ------------
    This function:
    1. Loads the MIDI file using music21
    2. Extracts notes from the specified track (or first track if not specified)
    3. Converts each note to (pitch, start_time, duration)
    4. Optionally quantizes durations to musical values (quarter, eighth, etc.)
    
    The output format is perfect for Markov chain training because we can
    model transitions between (pitch, duration) pairs.
    """
    try:
        # Load the MIDI file
        score = converter.parse(midi_path)
        
        # Find the appropriate part/track
        parts = score.parts if hasattr(score, 'parts') else [score]
        
        # If track_name is specified, find that track
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
        
        # Extract notes from the selected part
        notes_data = []
        
        # Get all notes and chords from the part
        for element in selected_part.flat.notes:
            if isinstance(element, note.Note):
                # Single note
                pitch = element.pitch.midi  # Convert to MIDI number (0-127)
                start = float(element.offset)  # Start time in beats
                dur = float(element.duration.quarterLength)  # Duration in beats
                
                # Quantize duration to common musical values
                if quantize:
                    dur = quantize_duration(dur)
                
                notes_data.append((pitch, start, dur))
                
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
        
        return notes_data
        
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return []


def quantize_duration(duration: float) -> float:
    """
    Quantizes a duration to the nearest common musical value.
    
    Parameters:
    -----------
    duration : float
        Duration in beats
    
    Returns:
    --------
    float
        Quantized duration (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, etc.)
    
    Explanation:
    ------------
    Musical durations are typically powers of 2 (whole, half, quarter, eighth, etc.).
    This function rounds durations to the nearest standard value to reduce
    the number of unique states in our Markov chain, making the model more
    generalizable.
    """
    # Common musical durations in beats
    common_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    # Find the closest duration
    closest = min(common_durations, key=lambda x: abs(x - duration))
    return closest


def preprocess_sequences(notes_data: List[Tuple[int, float, float]], 
                        include_rhythm: bool = True) -> List[Tuple]:
    """
    Converts raw note data into sequences suitable for Markov chain training.
    
    Parameters:
    -----------
    notes_data : List[Tuple[int, float, float]]
        List of (pitch, start_time, duration) tuples
    include_rhythm : bool
        If True, include rhythm information (duration) in the state
        If False, only use pitch
    
    Returns:
    --------
    List[Tuple]
        List of states. If include_rhythm=True: (pitch, duration) tuples
        If include_rhythm=False: just pitch integers
    
    Explanation:
    ------------
    This function converts the temporal note data into a sequence of states.
    Each state represents a musical event. By including rhythm, we can model
    both melodic and rhythmic patterns, which is more musically interesting
    than pitch alone.
    """
    if not notes_data:
        return []
    
    # Sort by start time to ensure correct order
    sorted_notes = sorted(notes_data, key=lambda x: x[1])
    
    if include_rhythm:
        # Create (pitch, duration) tuples
        states = [(pitch, duration) for pitch, start, duration in sorted_notes]
    else:
        # Only use pitch
        states = [pitch for pitch, start, duration in sorted_notes]
    
    return states


def load_dataset(data_dir: str, 
                track_name: Optional[str] = None,
                max_files: Optional[int] = None,
                include_rhythm: bool = True) -> List[List[Tuple]]:
    """
    Loads and preprocesses an entire dataset of MIDI files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIDI files
    track_name : str, optional
        Track name to extract (e.g., "MELODY")
    max_files : int, optional
        Maximum number of files to process
    include_rhythm : bool
        Whether to include rhythm in states
    
    Returns:
    --------
    List[List[Tuple]]
        List of sequences, where each sequence is a list of states
        (one sequence per MIDI file)
    
    Explanation:
    ------------
    This is the main function you'll use to load a dataset. It:
    1. Finds all MIDI files in the directory
    2. Extracts notes from each file
    3. Converts them to state sequences
    4. Returns all sequences ready for training
    
    Each sequence represents one piece of music, and we can train
    the Markov chain on all sequences together.
    """
    print(f"Loading MIDI files from {data_dir}...")
    midi_files = load_midi_files(data_dir, max_files)
    print(f"Found {len(midi_files)} MIDI files")
    
    all_sequences = []
    
    for i, midi_path in enumerate(midi_files):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(midi_files)} files...")
        
        notes_data = extract_notes_from_midi(midi_path, track_name)
        sequence = preprocess_sequences(notes_data, include_rhythm)
        
        if sequence:  # Only add non-empty sequences
            all_sequences.append(sequence)
    
    print(f"Successfully loaded {len(all_sequences)} sequences")
    return all_sequences


if __name__ == "__main__":
    # Example usage
    nottingham_dir = "data/nottingham_github/MIDI"
    sequences = load_dataset(nottingham_dir, max_files=10, include_rhythm=True)
    
    print(f"\nLoaded {len(sequences)} sequences")
    if sequences:
        print(f"First sequence length: {len(sequences[0])}")
        print(f"First 10 states: {sequences[0][:10]}")

