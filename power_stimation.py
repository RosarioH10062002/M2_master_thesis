import numpy as np
import sounddevice as sd
import pyplnoise


fs = 44100
duration = 3  # seconds

# amplitude reduction factor
step_factor = 0.8

# initial amplitude
amp = 1.0

# -----------------------------
# GENERATE PINK NOISE
# -----------------------------
samples = fs * duration

pknoise = pyplnoise.PinkNoise(fs, 1e-2, 50.)
x_pk = pknoise.get_series(int(samples))

# normalize
x_pk = x_pk / np.max(np.abs(x_pk))

print("Base RMS:", np.sqrt(np.mean(x_pk**2)))

# -----------------------------
# THRESHOLD SEARCH
# -----------------------------
last_audible_amp = amp

while True:

    signal = amp * x_pk

    print(f"\nPlaying amplitude: {amp:.8f}")

    sd.play(signal, fs, blocking=True)

    response = input(
        "Did participant hear the sound? (y/n/q): "
    ).lower()

    if response == "y":

        last_audible_amp = amp

        # decrease amplitude
        amp *= step_factor

    elif response == "n":

        print("Threshold reached")
        break

    elif response == "q":

        print("Calibration stopped")
        break

# -----------------------------
# THRESHOLD RMS
# -----------------------------
signal_ref = last_audible_amp * x_pk

rms_ref = np.sqrt(
    np.mean(signal_ref**2)
)

print("\nThreshold amplitude:", last_audible_amp)
print("Threshold RMS:", rms_ref)

# -----------------------------
# EXPERIMENT SIGNAL
# Example: actual stimulation level
# -----------------------------
exp_amp = 0.1

signal_exp = exp_amp * x_pk

rms_exp = np.sqrt(
    np.mean(signal_exp**2)
)

print("\nExperiment RMS:", rms_exp)

# -----------------------------
# SPL ESTIMATION
# -----------------------------
SPL = 20 * np.log10(
    rms_exp / rms_ref
)

print(
    f"\nEstimated SPL above threshold: "
    f"{SPL:.2f} dB"
)