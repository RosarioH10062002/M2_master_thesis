import sounddevice as sd
import numpy as np
from binaural import generate_isochronic_tone
import colorednoise as cn
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_streams
import time
#print(sd.query_devices())
#def generate_oneit(fb):

#def generate_oneit(fb):
fs = 44100
fb = 10
duration = 88 # gap of moreless 1 second 
tone = generate_isochronic_tone(
    frequency=440, #carrier frequency
    pulse_rate=fb, # alpha
    amplitude= 1,
    duration=duration
)

tone = tone / np.max(np.abs(tone))
#return tone.shape
#sd.play(tone, fs)
#sd.wait()

flag = True
index_psychopy = None
streams = resolve_streams() 
for i, s in enumerate(streams):
    print(f"Stream {i}:")
    print("  Name:", s.name())
    #print(type(s.name()))
    print("  Type:", s.type())
    print("  Channels:", s.channel_count())
    print("  Source ID:", s.source_id())
    print("------------------")
    
    if s.name() == "Psychopy_markers": 
        index_psychopy = i
        print(index_psychopy)

inlet = StreamInlet(streams[index_psychopy]) 
while flag == True:
    sample, timestamp = inlet.pull_sample() # wait until find
    if sample == [1000]:
        sd.play(tone, fs)
        sd.wait()
        print(sample, timestamp)
    elif sample == [1002]: 
        flag = False 

