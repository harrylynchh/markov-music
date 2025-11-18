# Notes from: https://medium.com/data-science/markov-chain-for-music-generation-932ea8a88305

# Same as natural languages we may think about music as a sequence of notes. 
# But because I play guitar I’ll be operating with chords. If we take chords 
# sequence and learn its patternt we may find that certain chords may follow 
# particular chords more often, and other chords rarely follow that chords. 
# We will build our model to find and understand this pattern.

# General plan:
# 1. Take corpus of chords
# 2. Calculate probability distribution for chords to follow a particular chord
# 3. Define the first chord or make a random choice
# 4. Make a random choice of the next chord taking into an account probability 
# distribution
# 5. Repeat steps 4 for the generated chord
# 6. …
# 7. Stochastic music is awesome!

import numpy as np 

# 1. First make the chords into bigrams... why? Making chords into bigrams for 
# Markov music generation is a way to model the statistical likelihood of one 
# chord following another, creating more musically plausible progressions than 
# random selection

# 2. Choose a chord randomly, so need to calc prob of other chords to follow it

# 3. So, calc the freq of each unique bigram to appear in a sequence with that
# chosen first chord 

# 3. Normalize to get prob distribution of these unique bigrams 

# 4. Move to next state choose randomly from all the options of chords, but all
# the chords have a more or less likely chance to get chosen (more likely to 
# choose C cord over G7) 
# Use np.random.choice(options, p=probabilities) for next choice of chord


def predict_next_state(chord:str, data:list=bigrams):
    """Predict next chord based on current state."""
    # create list of bigrams which stats with current chord
    bigrams_with_current_chord = [bigram for bigram in bigrams if bigram.split(' ')[0]==chord]
    # count appearance of each bigram
    count_appearance = dict(Counter(bigrams_with_current_chord))
    # convert apperance into probabilities
    for ngram in count_appearance.keys():
        count_appearance[ngram] = count_appearance[ngram]/len(bigrams_with_current_chord)
    # create list of possible options for the next chord
    options = [key.split(' ')[1] for key in count_appearance.keys()]
    # create  list of probability distribution
    probabilities = list(count_appearance.values())
    # return random prediction
    return np.random.choice(options, p=probabilities)

def generate_sequence(chord:str=None, data:list=bigrams, length:int=30):
    """Generate sequence of defined length."""
    # create list to store future chords
    chords = []
    for n in range(length):
        # append next chord for the list
        chords.append(predict_next_state(chord, bigrams))
        # use last chord in sequence to predict next chord
        chord = chords[-1]
    return chords
