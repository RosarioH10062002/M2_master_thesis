import numpy as np 
import pandas as pd 
import pyplnoise
import sounddevice as sd
from pylsl import StreamInlet, resolve_streams

#Pink Noise 
fs = 44100
duration = 90 #seconds
samples = fs*duration
print(samples)

t = np.linspace(0, duration, int(samples))
pknoise = pyplnoise.PinkNoise(fs, 1e-2, 50.)
x_pk = pknoise.get_series(int(samples))

# I want to look for the streams 
index_psychopy = None 
flag = True 
streams = resolve_streams()
for i,s in enumerate(streams): 
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
    sample, timestamp = inlet.pull_sample()
    if sample == [1000]:
        sd.play(x_pk, fs)
        sd.wait()
        print(sample, timestamp)
    elif sample == [1002]: 
        flag = False 


#print(x_pk.shape)


