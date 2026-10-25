from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv 

def save_trials_to_csv(all_trials, filename):
    if not all_trials:
        print("No trials to save.")
        return

    fieldnames = list(all_trials[0].keys())

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_trials)

    print(f"Saved {len(all_trials)} trials to {filename}")
    