"""
Example Usage Script

This script demonstrates how to use the Markov chain music generation system
step by step, with detailed explanations of what each part does.

Run this script to see a complete example from loading data to generating music.
"""

from midi_loader import load_dataset
from markov_chain import MarkovChain
from midi_generator import sequence_to_midi
import numpy as np


def example_basic_usage():
    """
    Basic example: Load data, train a first-order Markov chain, and generate music.
    
    This demonstrates the simplest workflow.
    """
    print("="*60)
    print("EXAMPLE 1: Basic Usage - First-Order Markov Chain")
    print("="*60)
    
    # Step 1: Load a small subset of the Nottingham dataset
    print("\n[Step 1] Loading MIDI files from Nottingham dataset...")
    print("(Using max_files=50 for quick demonstration)")
    
    sequences = load_dataset(
        data_dir="data/nottingham_github/MIDI",
        max_files=50,  # Use only 50 files for quick demo
        include_rhythm=True  # Include both pitch and rhythm
    )
    
    print(f"\nLoaded {len(sequences)} musical pieces")
    if sequences:
        print(f"First piece has {len(sequences[0])} notes")
        print(f"First 5 states: {sequences[0][:5]}")
    
    # Step 2: Split into training and validation
    print("\n[Step 2] Splitting data into training (80%) and validation (20%)...")
    
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    print(f"Training: {len(train_sequences)} pieces")
    print(f"Validation: {len(val_sequences)} pieces")
    
    # Step 3: Train a first-order Markov chain
    print("\n[Step 3] Training first-order Markov chain...")
    print("(This learns: P(next_note | current_note))")
    
    model = MarkovChain(order=1)
    model.train(train_sequences)
    
    # Step 4: Evaluate on validation set
    print("\n[Step 4] Evaluating model on validation set...")
    nll = model.calculate_negative_log_likelihood(val_sequences)
    print(f"Negative Log-Likelihood (NLL): {nll:.4f}")
    print("(Lower NLL = better model)")
    
    # Step 5: Generate new music
    print("\n[Step 5] Generating a new musical sequence...")
    generated = model.generate(length=50, temperature=1.0)
    print(f"Generated {len(generated)} states")
    print(f"First 10 states: {generated[:10]}")
    
    # Step 6: Convert to MIDI and save
    print("\n[Step 6] Converting to MIDI file...")
    sequence_to_midi(generated, "example_output_basic.mid")
    print("Saved to: example_output_basic.mid")
    
    print("\n" + "="*60)
    print("Basic example complete! Check example_output_basic.mid")
    print("="*60)


def example_higher_order():
    """
    Example: Compare first-order vs second-order Markov chains.
    
    Higher-order chains can capture longer-term patterns but require more data.
    """
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Comparing First-Order vs Second-Order")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    sequences = load_dataset(
        data_dir="data/nottingham_github/MIDI",
        max_files=100,  # Use more files for higher-order model
        include_rhythm=True
    )
    
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    # Train first-order model
    print("\nTraining first-order model (P(next | current))...")
    model_1 = MarkovChain(order=1)
    model_1.train(train_sequences)
    nll_1 = model_1.calculate_negative_log_likelihood(val_sequences)
    print(f"First-order NLL: {nll_1:.4f}")
    
    # Train second-order model
    print("\nTraining second-order model (P(next | current, previous))...")
    model_2 = MarkovChain(order=2)
    model_2.train(train_sequences)
    nll_2 = model_2.calculate_negative_log_likelihood(val_sequences)
    print(f"Second-order NLL: {nll_2:.4f}")
    
    # Compare
    print("\n" + "-"*60)
    if nll_2 < nll_1:
        print("Second-order model is better (lower NLL)!")
    else:
        print("First-order model is better (lower NLL)!")
    print(f"Difference: {abs(nll_2 - nll_1):.4f}")
    print("-"*60)
    
    # Generate with both models
    print("\nGenerating samples with both models...")
    gen_1 = model_1.generate(length=50)
    gen_2 = model_2.generate(length=50)
    
    sequence_to_midi(gen_1, "example_output_order1.mid")
    sequence_to_midi(gen_2, "example_output_order2.mid")
    
    print("First-order: example_output_order1.mid")
    print("Second-order: example_output_order2.mid")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


def example_temperature():
    """
    Example: How temperature affects generation.
    
    Temperature controls randomness:
    - Low temperature (< 1.0): More deterministic, follows training data closely
    - High temperature (> 1.0): More random, more creative but less coherent
    """
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Temperature Effects")
    print("="*60)
    
    # Load and train
    sequences = load_dataset(
        data_dir="data/nottingham_github/MIDI",
        max_files=50,
        include_rhythm=True
    )
    
    train_sequences = sequences[:int(len(sequences) * 0.8)]
    
    model = MarkovChain(order=1)
    model.train(train_sequences)
    
    # Generate with different temperatures
    temperatures = [0.5, 1.0, 2.0]
    
    print("\nGenerating with different temperatures...")
    for temp in temperatures:
        generated = model.generate(length=50, temperature=temp)
        filename = f"example_output_temp{temp}.mid"
        sequence_to_midi(generated, filename)
        print(f"Temperature {temp}: {filename}")
        print(f"  (temp < 1.0 = more conservative, temp > 1.0 = more creative)")
    
    print("\n" + "="*60)
    print("Temperature examples complete!")
    print("="*60)


def example_pitch_only_vs_rhythm():
    """
    Example: Compare models with and without rhythm information.
    """
    print("\n\n" + "="*60)
    print("EXAMPLE 4: Pitch-Only vs Pitch+Rhythm")
    print("="*60)
    
    # Load with rhythm
    print("\nLoading data with rhythm information...")
    sequences_with_rhythm = load_dataset(
        data_dir="data/nottingham_github/MIDI",
        max_files=50,
        include_rhythm=True
    )
    
    # Load without rhythm (pitch only)
    print("Loading data without rhythm (pitch only)...")
    sequences_pitch_only = load_dataset(
        data_dir="data/nottingham_github/MIDI",
        max_files=50,
        include_rhythm=False
    )
    
    # Train both models
    print("\nTraining pitch+rhythm model...")
    model_rhythm = MarkovChain(order=1)
    model_rhythm.train(sequences_with_rhythm[:int(len(sequences_with_rhythm) * 0.8)])
    
    print("Training pitch-only model...")
    model_pitch = MarkovChain(order=1)
    model_pitch.train(sequences_pitch_only[:int(len(sequences_pitch_only) * 0.8)])
    
    # Generate
    print("\nGenerating samples...")
    gen_rhythm = model_rhythm.generate(length=50)
    gen_pitch = model_pitch.generate(length=50)
    
    sequence_to_midi(gen_rhythm, "example_output_with_rhythm.mid")
    sequence_to_midi(gen_pitch, "example_output_pitch_only.mid")
    
    print("With rhythm: example_output_with_rhythm.mid")
    print("Pitch only: example_output_pitch_only.mid")
    print("\nNote: Rhythm model captures both melodic and rhythmic patterns!")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MARKOV CHAIN MUSIC GENERATION - EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates various features of the Markov chain model.")
    print("Each example will generate MIDI files you can listen to.\n")
    
    # Run examples
    try:
        example_basic_usage()
        example_higher_order()
        example_temperature()
        example_pitch_only_vs_rhythm()
        
        print("\n\n" + "="*60)
        print("ALL EXAMPLES COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - example_output_basic.mid")
        print("  - example_output_order1.mid")
        print("  - example_output_order2.mid")
        print("  - example_output_temp0.5.mid")
        print("  - example_output_temp1.0.mid")
        print("  - example_output_temp2.0.mid")
        print("  - example_output_with_rhythm.mid")
        print("  - example_output_pitch_only.mid")
        print("\nListen to these files to hear the differences!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Installed required packages: music21, numpy")
        print("  2. Downloaded the datasets (Nottingham, POP909)")
        print("  3. Set the correct data directory paths")

