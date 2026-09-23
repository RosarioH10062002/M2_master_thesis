import numpy as np
import pyplnoise
from scipy.io import wavfile
from binaural import generate_isochronic_tone


# ============================================================
# PARAMETERS
# ============================================================

fs = 44100          # Sampling frequency
duration = 15       # seconds

carrier_freq = 440  # Hz
pulse_rate = 10     # Isochronic tone frequency (Hz)

alpha = 0.10        # 10% isochronic tone + 90% pink noise
target_rms = 0.1


# ============================================================
# RMS NORMALIZATION
# ============================================================

def normalize_rms(signal, target_rms=0.1):
    rms = np.sqrt(np.mean(signal ** 2))

    if rms == 0:
        return signal

    return signal * (target_rms / rms)


# ============================================================
# GENERATE PINK NOISE
# ============================================================

def generate_pink_noise(duration, fs):

    samples = int(fs * duration)

    pknoise = pyplnoise.PinkNoise(
        fs,
        1e-2,
        50.
    )

    pink = pknoise.get_series(samples)

    # Peak normalization
    pink = pink / np.max(np.abs(pink))

    # RMS normalization
    pink = normalize_rms(pink, target_rms)

    return pink


# ============================================================
# GENERATE ISOCHRONIC TONE
# ============================================================

tone = generate_isochronic_tone(
    frequency=carrier_freq,
    pulse_rate=pulse_rate,
    amplitude=1,
    duration=duration
)

tone = np.asarray(tone)
tone = normalize_rms(tone, target_rms)


# ============================================================
# GENERATE PINK NOISE
# ============================================================

pink = generate_pink_noise(
    duration=duration,
    fs=fs
)


# ============================================================
# MAKE BOTH SIGNALS SAME LENGTH
# ============================================================

n = min(len(tone), len(pink))

tone = tone[:n]
pink = pink[:n]


# ============================================================
# MIX
# ============================================================

mixed = (
    alpha * tone
    + (1 - alpha) * pink
)

# Normalize final signal
mixed = normalize_rms(
    mixed,
    target_rms
)


# ============================================================
# CONVERT TO 16-BIT WAV
# ============================================================

mixed_wav = np.int16(
    mixed / np.max(np.abs(mixed)) * 32767
)


# ============================================================
# SAVE
# ============================================================

filename = "isochronic_10Hz_pink_noise_15s.wav"

wavfile.write(
    filename,
    fs,
    mixed_wav
)

print(f"Saved: {filename}")
print(f"Duration: {duration} s")
print(f"Carrier frequency: {carrier_freq} Hz")
print(f"Isochronic pulse rate: {pulse_rate} Hz")
print(f"Isochronic tone proportion: {alpha * 100:.0f}%")
print(f"Pink noise proportion: {(1-alpha) * 100:.0f}%")

import os

print("Current working directory:")
print(os.getcwd())

print("WAV will be saved at:")
print(os.path.abspath(filename))