import numpy as np 
import pandas as pd 
import pyplnoise
import datetime
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
x_pk = x_pk / np.max(np.abs(x_pk))


#Ding Sound
samplerate, data = wavfile.read("ding_sound.wav")

# Feedback sounds for practice (Mode X)
fs_right, right_audio = wavfile.read("right.wav")
fs_wrong, wrong_audio = wavfile.read("wrong.wav")

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

if index_psychopy is None:
    raise RuntimeError("Psychopy_markers stream not found")

inlet = StreamInlet(streams[index_psychopy])

try:
    while flag:
        sample, timestamp = inlet.pull_sample()
        value = int(sample[0])

        if value == 1000:
            sd.play(x_pk, fs, blocking=False)
            print(f"Pink noise started {datetime.datetime.now()}")

        elif value == 104:
            print("Correct response: RIGHT")
            sd.play(right_audio, fs_right, blocking=False)

        elif value == 105:
            print("Incorrect response: WRONG")
            sd.play(wrong_audio, fs_wrong, blocking=False)

        elif value == 1003:
            sd.play(data, samplerate, blocking=False)
            print(f"Ding {datetime.datetime.now()}")

        elif value == 1002:
            sd.stop()
            flag = False
            print(f"Stopping {datetime.datetime.now()}")

except KeyboardInterrupt:
    sd.stop()
    print("Stopped by user")
