import csv
from pathlib import Path

import numpy as np
import sounddevice as sd
import pyplnoise

from pylsl import StreamInlet, resolve_byprop, local_clock


# ============================================================
# SYNCHRONIZATION VALIDATION - AUDITORY LISTENER
# ============================================================

N_TRIGGERS = 300

FS = 44100
TEST_DURATION = 1.0  # seconds

MARKER_VALUE = "1000"

# Must match the stream name from the PsychoPy validation script
LSL_STREAM_NAME = "PsychoPyMarkersValidation"

# None = use the normal/default audio output
OUTPUT_DEVICE = None

output_file = Path("auditory_trigger_timestamps_4.csv")


# ============================================================
# GENERATE TEST AUDIO
# ============================================================
# Generate the signal BEFORE receiving markers so that signal
# generation itself does not influence trigger-to-playback latency.

samples = int(FS * TEST_DURATION)

pknoise = pyplnoise.PinkNoise(
    FS,
    1e-2,
    50.0
)

signal = pknoise.get_series(samples)

# RMS normalize exactly as in the experimental implementation
rms = np.sqrt(np.mean(signal**2))
signal = signal * (0.1 / rms)

# sounddevice works well with float32
signal = signal.astype(np.float32)

print(f"Test signal RMS: {np.sqrt(np.mean(signal**2)):.6f}")


# ============================================================
# FIND PSYCHOPY LSL STREAM
# ============================================================

print("\nSearching for PsychoPy validation marker stream...")

streams = resolve_byprop(
    "name",
    LSL_STREAM_NAME,
    timeout=30
)

if len(streams) == 0:
    raise RuntimeError(
        f"Could not find LSL stream: {LSL_STREAM_NAME}"
    )

inlet = StreamInlet(streams[0])

print("LSL marker stream found.")
print("Waiting for marker 1000...\n")


# ============================================================
# OPEN OUTPUT CSV
# ============================================================

with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "trigger_number",
        "marker",
        "psychopy_marker_timestamp",
        "auditory_receive_time",
        "audio_command_time",
        "audio_call_return_time",
        "send_to_receive_ms",
        "receive_to_command_ms",
        "send_to_command_ms",
        "sd_play_call_duration_ms"
    ])

    trigger_number = 0

    # ========================================================
    # LISTEN FOR 100 TRIGGERS
    # ========================================================

    while trigger_number < N_TRIGGERS:

        # Wait for next LSL marker
        sample, marker_timestamp = inlet.pull_sample(
            timeout=10.0
        )

        if sample is None:
            print("No marker received within 10 s...")
            continue

        marker = str(sample[0])

        # Ignore markers other than 1000
        if marker != MARKER_VALUE:
            print(f"Ignoring marker: {marker}")
            continue

        trigger_number += 1

        # ----------------------------------------------------
        # Timestamp when auditory process has received marker
        # ----------------------------------------------------

        t_receive = local_clock()

        # ----------------------------------------------------
        # Timestamp immediately before requesting playback
        # ----------------------------------------------------

        t_audio_command = local_clock()

        sd.play(
            signal,
            samplerate=FS,
            device=OUTPUT_DEVICE,
            blocking=False
        )

        # Time immediately after sd.play() returned
        t_after_play_call = local_clock()

        # ----------------------------------------------------
        # CALCULATE SOFTWARE LATENCIES
        # ----------------------------------------------------

        send_to_receive_ms = (
            t_receive - marker_timestamp
        ) * 1000

        receive_to_command_ms = (
            t_audio_command - t_receive
        ) * 1000

        send_to_command_ms = (
            t_audio_command - marker_timestamp
        ) * 1000

        play_call_duration_ms = (
            t_after_play_call - t_audio_command
        ) * 1000

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        writer.writerow([
            trigger_number,
            marker,
            f"{marker_timestamp:.9f}",
            f"{t_receive:.9f}",
            f"{t_audio_command:.9f}",
            f"{t_after_play_call:.9f}",
            f"{send_to_receive_ms:.6f}",
            f"{receive_to_command_ms:.6f}",
            f"{send_to_command_ms:.6f}",
            f"{play_call_duration_ms:.6f}"
        ])

        f.flush()

        print(
            f"Trigger {trigger_number:03d}/{N_TRIGGERS} | "
            f"LSL: {send_to_receive_ms:.3f} ms | "
            f"Processing: {receive_to_command_ms:.3f} ms | "
            f"Total: {send_to_command_ms:.3f} ms"
        )


print("\nSynchronization validation completed.")
print(f"Results saved to: {output_file.resolve()}")

# Make sure any final playback finishes
sd.wait()