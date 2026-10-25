# Participant Data

## Contents

- `01_screening/`: Behavioral data collected from the initial 14 participants. Participants first completed a practice task with feedback to ensure they understood the SART (`01_practice/`). After successfully completing the practice, they performed the screening task (`02_assessment/`), from which the behavioral data used for participant selection were collected.

- `02_longitudinal/`: Following the screening process, six participants were selected for the longitudinal study. This directory contains:

  - `01_metadata/`: Summary of all experimental sessions and session-level metadata, including volume, brightness, phase, mood, hours of sleep, tiredness, IT (isochronic tone/pink noise mixing ratio), F_ORDER (frequency order across the six blocks), and SESSION (time of day: morning, afternoon, or evening).

  - `ID_x/`: Individual participant folders, where `x` represents the participant ID (e.g., `ID_5/`). Each folder contains:
    - **Behavioral data:** Files named `behavior_date_phase`, where the date follows the `dd-mm-yy` format and the phase indicates the experimental condition (A = pink noise; B = pink noise + isochronic tones).
    - **EEG data:** Files named `EEG_Behavior_date_phase`, following the same date and phase conventions.

- `info_participants.csv`: Participant information (gender, ASRS scores) collected during the initial screening session for all 14 participants.

## Experimental Conditions

The initial 14 participants completed six blocks of the Sustained Attention to Response Task (SART), each consisting of 90 trials (540 trials in total), while listening to pink noise.