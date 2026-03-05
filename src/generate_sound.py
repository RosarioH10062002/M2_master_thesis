from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv 

target_rms = 0.1 # common psychophysics range, the idea is that all stimulus have the identical energy 
#----------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(x**2))
#-
def set_rms(x, target_rms, eps=1e-12):
    r = rms(x)
    if r < eps:
        return x
    return x * (target_rms / r)
    
def generate_pink_noise(duration, fs):
    N = int(duration * fs)

    x = np.random.randn(N)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # 1/sqrt(f) =  pink noise 
    shaping = np.ones_like(freqs)
    shaping[1:] = 1.0 / np.sqrt(freqs[1:])
    Xpink = X * shaping

    pink = np.fft.irfft(Xpink, n=N) # time domain
    pink = set_rms(pink, target_rms)
   
    #pink = pink / (np.max(np.abs(pink)) + 1e-12)

    return pink.astype(np.float32)

def generate_ITS_withnoise(fc, fb, dc, duration_seconds,fs, its_ratio): # The loudness is an issue here, since it will depend of the its_ratio creating a cofound in the experiments 

    t = np.arange(0, duration_seconds, 1/fs)

    # ----- ITS -----
    carrier = np.sin(2 * np.pi * fc * t)
    envelope = (np.sign(np.sin(2 * np.pi * fb * t)) + 1) / 2
    envelope = (envelope > (1 - dc)).astype(float)

    its = carrier * envelope
    #its = its / np.max(np.abs(its))
    its = set_rms(its, target_rms)
    # ----- Pink noise -----
    noise = generate_pink_noise(duration_seconds, fs)
    
    #noise = set_rms(noise, target_rms)
    # ensure same length
    N = min(len(its), len(noise))
    its = its[:N]
    noise = noise[:N]

    # ----- Mix -----
    mix = its_ratio * its + (1 - its_ratio) * noise # its_ratio is the ratio between ITS/NOISE
    mix = set_rms(mix, target_rms)
    #mix = mix / np.max(np.abs(mix))

    return t[:N], mix.astype(np.float32)