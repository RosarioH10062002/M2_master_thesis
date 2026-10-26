# Experimental Design

The experimental implementation is organized into two main folders: `linux_version` and `stimuli`. Additional scripts are provided for volume adjustment, auditory threshold estimation, and EEG data retrieval (OpenBCI).

Before starting, you can ask participants which computer volume level is comfortable for them. To do this, run `trial_noise`, which plays pink noise for 3 seconds. Make sure not to start at maximum volume. I started at 70% and then reduced it according to the participant's preference. The idea is to find a comfortable volume level, which should ideally remain constant throughout the experiment. You can repeat this procedure until a consistent volume setting is reached.

After that, you can estimate their auditory threshold and record it for reference. To do this, run `power_stimation` and ask the participant whether they can hear the sound. Enter y if they can hear it and n if they cannot. The script will then calculate the final RMS value. Since you only need to record one value, you can perform this measurement after establishing a stable volume setting.

## 1. `linux_version`

This folder contains the Python scripts required to run the Sustained Attention to Response Task (SART) paradigm. The main entry point is `main.py`, which supports several experimental modes.

### Experimental modes

**Mode X — Practice and baseline**

This mode assesses whether a prospective participant understands the task. Participants first complete a SART practice session with visual and auditory feedback. Once they achieve an accuracy above 80%, they proceed to six SART blocks while listening to pink noise.

**Mode Z — Isochronic-tone stimulation**

This mode includes the following sequence:

1. **Pre-task EEG:** One minute with eyes open, followed by one minute with eyes closed. A ding sound signals when the participant should close their eyes.
2. **2AFC calibration:** A two-alternative forced-choice task is used to determine the individual isochronic-tone (IT) mixing ratio.
3. **SART with auditory stimulation:** Participants complete six SART blocks while listening to a combination of pink noise and isochronic tones at different modulation frequencies.
4. **Post-task EEG:** The same eyes-open and eyes-closed recording procedure is repeated.

The experimental mode can be selected in `main.py` by changing the `EXPERIMENT_MODE` variable (currently located around line 83).

## 2. Running the experiment

The experiment requires two scripts to run simultaneously: the PsychoPy experiment and a separate Python script responsible for audio playback.

The audio script can be executed from a Python environment or IDE, such as Visual Studio Code.

| Experimental mode | Audio playback script |
|---|---|
| X — Practice + baseline | `generate_pink_noise.py` |
| Z — 2AFC + IT stimulation | `generate_it.py` |

**Execution procedure:**

1. Open `main.py` in PsychoPy and select the desired experimental mode.
2. Run `main.py` until the PsychoPy window appears.
3. Start the corresponding audio playback script in a separate Python terminal, such as the integrated terminal in Visual Studio Code.
4. Check that the audio script has successfully connected to the LSL stream and that no errors have occurred.
5. Return to PsychoPy and continue the experiment.

The PsychoPy experiment and the audio playback script communicate through **Lab Streaming Layer (LSL)** markers.

For sessions involving EEG, ensure that the EEG acquisition system and LSL marker recording are also running and connected before proceeding with the experimental tasks.

## 3. `stimuli`

This folder contains the visual and auditory stimuli required for the practice session (including feedback) and the experimental SART blocks.

**Important:** The contents of `linux_version` and `stimuli` must be placed in the same working directory when running the experiment. They are provided separately only for organizational clarity.


## 4. Output files

At the end of each experimental mode, PsychoPy displays the filename and directory where the experimental data have been saved.

## 5. Extra Information

There is a file called retrieve_eeg_data, which was developed for the previous EEG equipment (OpenBCI). Therefore, it is no longer needed for the current protocol.