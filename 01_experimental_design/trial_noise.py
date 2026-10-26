import numpy as np 
import pandas as pd 
import sounddevice as sd
import pyplnoise

#Pink Noise 
fs = 44100
duration = 3 #seconds
samples = fs*duration
print(samples)
t = np.linspace(0, duration, int(samples))
pknoise = pyplnoise.PinkNoise(fs, 1e-2, 50.)
x_pk = pknoise.get_series(int(samples))
x_pk = x_pk / np.max(np.abs(x_pk))
sd.play(x_pk, fs, blocking=True)
rms = np.sqrt(np.mean(x_pk**2))
print(rms)
