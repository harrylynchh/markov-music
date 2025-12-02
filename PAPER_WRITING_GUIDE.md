# Writing About MIDI Preprocessing in Your Paper

This guide provides ready-to-use text snippets and explanations for writing about the MIDI preprocessing pipeline in your academic paper.

---

## Quick Summary for Abstract/Introduction

**One-paragraph version:**
> "We developed a preprocessing pipeline that converts MIDI files into discrete state sequences suitable for Markov chain modeling. The pipeline handles multiple datasets (Nottingham folk music, POP909 pop songs), extracts both pitch and rhythmic information using the music21 library, quantizes durations to standard musical values, and includes robust error handling for batch processing of large datasets."

**Two-sentence version:**
> "MIDI files are preprocessed into discrete state sequences where each state represents a musical event (pitch, optionally with duration). The preprocessing pipeline includes recursive file discovery, note extraction with track selection, duration quantization to eight standard musical values, and sequence construction with chronological ordering."

---

## Methodology Section: Data Preprocessing

### Full Subsection Text

**Title: "Data Preprocessing and State Representation"**

> "To enable Markov chain modeling, we preprocess MIDI files into discrete state sequences. Our pipeline consists of five main components:
> 
> **File Discovery**: We recursively scan dataset directories using Python's `glob` module with pattern matching (`**/*.mid`) to automatically discover all MIDI files, enabling batch processing of large datasets without manual enumeration.
> 
> **Note Extraction**: MIDI files are parsed using the music21 library (Cuthbert & Ariza, 2010), which abstracts over low-level MIDI byte encoding. For each MIDI file, we extract musical events as tuples of (pitch, temporal_offset, duration), where pitch is encoded as MIDI note numbers (0-127), temporal offset represents onset time in beats, and duration is measured in quarter-note units. The extraction handles both single notes and chords, with chords simplified to their root (lowest) pitch to maintain a manageable state space. For multi-track MIDI files (e.g., POP909 with separate MELODY, BRIDGE, and PIANO tracks), we support track name matching to extract specific instrument parts. Error handling ensures graceful degradation: malformed MIDI files result in empty sequences rather than pipeline failure.
> 
> **Duration Quantization**: Continuous duration values are mapped to eight discrete musical durations using nearest-neighbor matching with L1 distance. The standard durations are: sixteenth notes (0.25 beats), eighth notes (0.5), dotted eighths (0.75), quarter notes (1.0), dotted quarters (1.5), half notes (2.0), dotted halves (3.0), and whole notes (4.0). This quantization reduces state space dimensionality while preserving musically meaningful rhythmic patterns.
> 
> **Sequence Construction**: Temporal note data is sorted by onset time to ensure chronological ordering, which is essential for Markov chain state transitions that model sequential dependencies. States are constructed as either pitch-only integers or (pitch, duration) tuples, depending on whether rhythmic information is included. This design choice enables modeling of melodic patterns alone or combined melodic-rhythmic patterns.
> 
> **Dataset Loading**: The complete pipeline is orchestrated by a dataset loading function that processes all MIDI files in a dataset directory, applies the preprocessing steps, and returns a list of state sequences ready for Markov chain training. Progress reporting occurs every 100 files to provide user feedback during large-scale batch processing."

---

## Technical Details for Methods Section

### State Representation

**Option 1 (Concise):**
> "States are represented as either pitch integers (MIDI note numbers 0-127) or (pitch, duration) tuples. When rhythm is included, states capture both melodic and rhythmic patterns; when excluded, the model focuses solely on melodic contour."

**Option 2 (Detailed):**
> "Our state representation supports two modes: (1) pitch-only, where states are integers representing MIDI note numbers (0-127), enabling modeling of melodic patterns; and (2) pitch-rhythm, where states are (pitch, duration) tuples, enabling simultaneous modeling of melodic and rhythmic patterns. Duration values are quantized to eight standard musical durations to maintain a discrete state space."

### Quantization Rationale

> "Duration quantization serves two purposes: (1) it reduces the state space from a continuous temporal domain to a discrete set of musically meaningful values, improving model generalization; and (2) it aligns with standard musical notation, where durations are typically expressed as powers of 2 (whole, half, quarter, eighth notes, etc.). We employ nearest-neighbor matching with L1 distance to map each continuous duration to its closest standard value, minimizing information loss while achieving discrete representation."

### Chord Simplification

> "Chords are simplified to their root (lowest) pitch to maintain computational tractability. While this simplification loses harmonic information, it prevents exponential growth in state space that would occur if all chord combinations were represented. Future work could explore multi-pitch representations or separate models for harmony and melody."

---

## Results Section: Dataset Statistics

### Template

> "The preprocessing pipeline successfully processed [X] MIDI files from the [Dataset Name] dataset, extracting [Y] total musical events. After preprocessing, we obtained [Z] state sequences with an average length of [A] states per sequence. [B]% of files were successfully processed, with [C] files filtered out due to extraction errors or empty content."

### Example (Nottingham)

> "The preprocessing pipeline successfully processed 3,089 MIDI files from the Nottingham dataset, extracting approximately 450,000 total musical events. After preprocessing, we obtained 3,045 state sequences with an average length of 148 states per sequence. 98.6% of files were successfully processed, with 44 files filtered out due to extraction errors or empty content."

---

## Discussion Section: Limitations and Future Work

### Limitations

> "Our preprocessing pipeline has several limitations. First, chord simplification to root pitch loses harmonic richness, though this trade-off maintains a manageable state space. Second, duration quantization to eight discrete values may lose subtle rhythmic nuances present in the original music. Third, the pipeline processes monophonic sequences (one note at a time), limiting its applicability to polyphonic music where multiple notes occur simultaneously. Finally, track selection for multi-track MIDI files requires manual specification of track names, which may not generalize across all datasets."

### Future Work

> "Several directions could improve the preprocessing pipeline. Multi-pitch chord representations could preserve harmonic information while maintaining computational tractability. Finer-grained duration quantization or adaptive quantization based on musical context could capture more rhythmic nuance. Polyphonic modeling could extend the approach to handle multiple simultaneous voices. Automatic track detection could eliminate the need for manual track name specification."

---

## Code Architecture Description

### For Technical Sections

> "The preprocessing pipeline is implemented as a modular Python package with five main functions:
> 
> 1. `load_midi_files(data_dir, max_files)`: Recursively discovers MIDI files using glob pattern matching
> 2. `extract_notes_from_midi(midi_path, track_name, quantize)`: Parses MIDI files and extracts (pitch, offset, duration) tuples
> 3. `quantize_duration(duration)`: Maps continuous durations to discrete musical values
> 4. `preprocess_sequences(notes_data, include_rhythm)`: Converts temporal events to state sequences
> 5. `load_dataset(data_dir, track_name, max_files, include_rhythm)`: Orchestrates the complete pipeline
> 
> The modular design enables easy extension and testing of individual components."

---

## Key Metrics to Report

### Dataset Statistics
- Total MIDI files processed
- Successfully processed files (percentage)
- Average sequence length
- Total musical events extracted
- Unique states (pitch or pitch-duration combinations)

### Processing Statistics
- Processing time per file
- Memory usage
- Error rate (files that failed to process)

### Example Table Format

| Dataset | Files | Processed | Avg Length | Unique States |
|---------|-------|-----------|------------|---------------|
| Nottingham | 3,089 | 3,045 (98.6%) | 148 | 1,247 |
| POP909 | 909 | 887 (97.6%) | 312 | 2,156 |

---

## Academic Writing Tips

### Use Precise Terminology
- ✅ "temporal offset" or "onset time" (not "time")
- ✅ "duration in quarter-note beats" (not "length")
- ✅ "state sequence" (not "list of notes")
- ✅ "quantization" (not "rounding")
- ✅ "discrete representation" (not "simplified")

### Cite music21
> "We utilize the music21 library (Cuthbert & Ariza, 2010) for MIDI file parsing and musical structure analysis."

**Citation format:**
```
Cuthbert, M. S., & Ariza, C. (2010). music21: A toolkit for computer-aided musicology and symbolic music data. 
In Proceedings of the 11th International Society for Music Information Retrieval Conference (ISMIR 2010).
```

### Emphasize Design Decisions
- "We chose to quantize durations to reduce state space dimensionality..."
- "Chord simplification was necessary to maintain computational tractability..."
- "Chronological sorting ensures proper sequential modeling..."

### Connect to Markov Chain Theory
- "State sequences enable Markov chain modeling by providing discrete states with clear transition relationships."
- "Quantization creates a finite state space required for Markov chain transition matrices."
- "Chronological ordering preserves temporal dependencies that Markov chains model through state transitions."

---

## Common Questions from Reviewers (and Answers)

**Q: Why quantize durations instead of using continuous values?**
> A: "Markov chains require a discrete state space. Quantization to standard musical durations (1) creates a finite, manageable state space, (2) aligns with musical notation conventions, and (3) improves model generalization by grouping similar durations."

**Q: Why simplify chords to root pitch?**
> A: "Full chord representation would create an exponential state space (all combinations of pitches). Root pitch simplification maintains a manageable state space while preserving the most harmonically salient information. This is a common simplification in music information retrieval."

**Q: How do you handle polyphonic music?**
> A: "The current pipeline extracts monophonic sequences (one note at a time). For multi-track MIDI files, we select a single track (typically the melody). This limitation could be addressed in future work with polyphonic modeling approaches."

**Q: What about tempo and key information?**
> A: "Tempo and key information are not currently included in the state representation, as we focus on pitch and rhythm patterns. These could be incorporated as additional state dimensions or as separate models."

---

## Checklist for Your Paper

- [ ] Describe the five main preprocessing functions
- [ ] Explain state representation (pitch-only vs pitch-rhythm)
- [ ] Justify duration quantization (why 8 values, why these specific values)
- [ ] Explain chord simplification rationale
- [ ] Report dataset statistics (files processed, success rate, average sequence length)
- [ ] Discuss limitations (chord simplification, quantization granularity, monophonic limitation)
- [ ] Cite music21 library
- [ ] Connect preprocessing choices to Markov chain requirements
- [ ] Include a figure/diagram showing the preprocessing pipeline (optional but recommended)

---

## Figure/Diagram Suggestions

**Pipeline Diagram:**
```
MIDI Files → File Discovery → Note Extraction → Quantization → Sequence Construction → State Sequences
                ↓                  ↓              ↓                    ↓
            glob pattern      music21 parse   nearest-neighbor    chronological sort
```

**State Representation Diagram:**
```
Raw MIDI: Note(pitch=C4, offset=0.0, duration=1.0)
    ↓
Extract: (60, 0.0, 1.0)
    ↓
Quantize: (60, 1.0)  [duration quantized to 1.0]
    ↓
State: (60, 1.0)  [pitch-rhythm mode]
   OR
State: 60  [pitch-only mode]
```

---

This guide should give you everything you need to write about the preprocessing pipeline in your paper!

