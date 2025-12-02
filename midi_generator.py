"""
MIDI Generator Module

This module converts generated state sequences back into MIDI files that can be
played or analyzed. It handles both pitch-only and pitch+rhythm sequences.
"""

from music21 import stream, note, tempo, meter
from typing import List, Tuple, Union


def sequence_to_midi(sequence: List[Union[int, Tuple]], 
                     output_path: str = "generated_music.mid",
                     tempo_bpm: int = 120,
                     time_signature: str = "4/4"):
    """
    Convert a generated sequence of states into a MIDI file.
    
    Parameters:
    -----------
    sequence : List[Union[int, Tuple]]
        Generated sequence. Can be:
        - List of pitches (int): [60, 62, 64, ...]
        - List of (pitch, duration) tuples: [(60, 1.0), (62, 0.5), ...]
    output_path : str
        Path where the MIDI file will be saved
    tempo_bpm : int
        Tempo in beats per minute
    time_signature : str
        Time signature (e.g., "4/4", "3/4")
    
    Returns:
    --------
    None
        Saves a MIDI file to output_path
    
    Explanation:
    ------------
    This function:
    1. Creates a music21 Stream (the container for musical elements)
    2. Sets tempo and time signature
    3. For each state in the sequence:
       - If it's a tuple (pitch, duration), uses that duration
       - If it's just a pitch, uses a default duration (quarter note)
       - Creates a Note object and adds it to the stream
    4. Writes the stream to a MIDI file
    
    The resulting MIDI file can be played in any music player or DAW.
    """
    # Create a new stream
    score = stream.Stream()
    
    # Set tempo
    score.append(tempo.MetronomeMark(number=tempo_bpm))
    
    # Set time signature
    score.append(meter.TimeSignature(time_signature))
    
    # Track current time position
    current_time = 0.0
    
    # Process each state in the sequence
    for state in sequence:
        if isinstance(state, tuple):
            # State is (pitch, duration)
            pitch, duration = state
        else:
            # State is just a pitch (int)
            pitch = state
            duration = 1.0  # Default to quarter note
        
        # Create a note
        # MIDI pitch 60 = C4 (middle C)
        # We need to ensure pitch is in valid range (0-127)
        if isinstance(pitch, (int, float)):
            pitch = int(pitch)
            if 0 <= pitch <= 127:
                n = note.Note(pitch)
                n.duration.quarterLength = duration
                n.offset = current_time
                score.append(n)
                
                # Move forward in time
                current_time += duration
        else:
            # Skip invalid pitches
            continue
    
    # Write to MIDI file
    score.write('midi', fp=output_path)
    print(f"Generated MIDI saved to {output_path}")


def sequences_to_midi(sequences: List[List[Union[int, Tuple]]],
                     output_dir: str = "generated",
                     prefix: str = "generated"):
    """
    Convert multiple sequences to MIDI files.
    
    Parameters:
    -----------
    sequences : List[List]
        List of generated sequences
    output_dir : str
        Directory to save MIDI files
    prefix : str
        Prefix for output filenames
    
    Returns:
    --------
    None
        Saves multiple MIDI files
    
    Explanation:
    ------------
    Useful for generating multiple pieces and saving them all.
    Each sequence becomes a separate MIDI file.
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sequence in enumerate(sequences):
        output_path = os.path.join(output_dir, f"{prefix}_{i+1}.mid")
        sequence_to_midi(sequence, output_path)
    
    print(f"Generated {len(sequences)} MIDI files in {output_dir}")

