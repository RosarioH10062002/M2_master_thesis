import numpy as np 
import pandas as pd 
import pyplnoise
import datetime
import time
import sounddevice as sd
from pylsl import StreamInlet, resolve_streams
from scipy.io import wavfile
from binaural import generate_isochronic_tone
from itertools import permutations
import random

fs = 44100
#duration = 91 #seconds


data = [12, 12, 20, 20, 30, 30]
all_sequences = list(set(permutations(data)))

random.shuffle(all_sequences) 



def normalize_rms(signal, target_rms=0.1):
    rms = np.sqrt(np.mean(signal**2))
    return signal * (target_rms / rms)

def generate_pink_noise(duration,fs):
    #Pink Noise 

    samples = fs*duration
    print(samples)

    t = np.linspace(0, duration, int(samples))
    pknoise = pyplnoise.PinkNoise(fs, 1e-2, 50.)
    x_pk = pknoise.get_series(int(samples))
    x_pk = x_pk / np.max(np.abs(x_pk))
    x_pk = normalize_rms(x_pk)

    return x_pk

def get_sequence():
    return all_sequences.pop()



def generate_its(duration):
    sequence_fb = get_sequence()

    list_tone = []

    for fb in sequence_fb:
        tone = generate_isochronic_tone(
            frequency=440,
            pulse_rate=fb,
            amplitude=1,
            duration=duration
        )
        tone = normalize_rms(tone)
        list_tone.append(tone)

    return list_tone, sequence_fb

def generate_it_pink_noise(duration, fs, alpha=0.05):
    tones, sequence_fb = generate_its(duration)

    mixed_signals = []

    for tone in tones:
        pink = generate_pink_noise(duration, fs)

        n = min(len(pink), len(tone))
        tone = tone[:n]
        pink_seg = pink[:n]

        mixed = (1 - alpha) * pink_seg + alpha * tone
        mixed = normalize_rms(mixed, 0.1)

        mixed_signals.append(mixed)

    return mixed_signals, sequence_fb

def generate_only_one_it(duration, fs, alpha = 0.1): 
    fb = 10
    tone = generate_isochronic_tone(
    frequency=440, #carrier frequency
    pulse_rate=fb, # alpha
    amplitude= 1,
    duration=duration)
    pink = generate_pink_noise(duration, fs)
    n = min(len(tone), len(pink))
    tone = tone[:n]
    pink_seg = pink[:n]

    mixed = (1 - alpha) * pink_seg + alpha * tone
    mixed_one_signal = normalize_rms(mixed, 0.1)
    return mixed_one_signal


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

if index_psychopy is None:
    raise RuntimeError("Psychopy_markers stream not found")

alpha_cache = {}


inlet = StreamInlet(streams[index_psychopy])

try:
    
    xpk = generate_pink_noise(duration = 8,fs = fs)
    alpha_values = [0.15, 0.20, 0.25, 0.30]

    stimuli = []
    for alpha in alpha_values:
        stimuli.append(generate_only_one_it(duration=8, fs=fs, alpha=alpha))

    current_idx = 0
    attemps_2afc = 0


    idx = 0
    last_timestamp = None

    while flag:
        sample, timestamp = inlet.pull_sample()
        value = int(sample[0])
        if timestamp == last_timestamp:
            continue

        last_timestamp = timestamp

        if value == 1000:
            audio = mixed_signals[idx]
            print(f"Playing condition {idx}, freq {sequence_fb[idx]} Hz at {datetime.datetime.now()}")
            sd.play(audio, fs, blocking=False)
            idx += 1

        elif value == 1003:
            sd.play(data, samplerate, blocking=False)
            print(f"Ding {datetime.datetime.now()}")

        elif value == 1002:
            sd.stop()
            flag = False
            print(f"Stopping {datetime.datetime.now()}")
            
        elif value == 111: #PINK NOISE 
            #current_idx = 0
            sd.play(xpk, fs, blocking=False)
            print(f"Pink noise at {datetime.datetime.now()}")
            

        elif value == 222:  # IT
            if current_idx < len(stimuli):
                sd.play(stimuli[current_idx], fs)
                print(f"Playing alpha {alpha_values[current_idx]}")
                print(f"Current idx {current_idx}")

                attemps_2afc += 1

                if attemps_2afc == 5:
                    attemps_2afc = 0
                    current_idx += 1
            else:
                print("All alpha levels completed.")

        elif value ==333: #once we get the final its ratio 
            final_its_ratio = alpha_values[current_idx-1]-0.05
            mixed_signals, sequence_fb = generate_it_pink_noise(duration = 92, fs = 44100, alpha=final_its_ratio)
            print(f"Final ITS ratio {final_its_ratio}")




except KeyboardInterrupt:
    sd.stop()
    print("Stopped by user")


