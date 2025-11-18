# Notes from: https://medium.com/data-science/markov-chain-for-music-generation-932ea8a88305

# Same as natural languages we may think about music as a sequence of notes. 
# But because I play guitar I’ll be operating with chords. If we take chords 
# sequence and learn its patternt we may find that certain chords may follow 
# particular chords more often, and other chords rarely follow that chords. 
# We will build our model to find and understand this pattern.

# Okay, here is the plan:
# 1. Take corpus of chords
# 2. Calculate probability distribution for chords to follow a particular chord
# 3. Define the first chord or make a random choice
# 4. Make a random choice of the next chord taking into an account probability 
# distribution
# 5. Repeat steps 4 for the generated chord
# 6. …
# 7. Stochastic music is awesome!

