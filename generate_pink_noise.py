import numpy as np 
import pandas as pd 
import pyplnoise
import sounddevice as sd
from pylsl import StreamInlet, resolve_streams
from scipy.io import wavfile


#Pink Noise 
fs = 44100
duration = 91 #seconds
samples = fs*duration
print(samples)

t = np.linspace(0, duration, int(samples))
pknoise = pyplnoise.PinkNoise(fs, 1e-2, 50.)
x_pk = pknoise.get_series(int(samples))


#Ding Sound
wav_file = "/home/telecom/Bureau/Rosario/M2_master_thesis-main/ding_sound.wav" # wav file needs the whole path 
samplerate, data = wavfile.read(wav_file)
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
    if sample == [1003]:
        sd.play(data, samplerate)
        sd.wait()
    elif sample == [1002]: 
        flag = False 


#print(x_pk.shape)


