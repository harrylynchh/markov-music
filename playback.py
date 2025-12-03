import sys
from mido import MidiFile

def playback_midi(path):
    midi = MidiFile(path)
    midi.play()

if __name__ == '__main__':
    playback_midi(sys.argv[1])
    
    # python playback.py 