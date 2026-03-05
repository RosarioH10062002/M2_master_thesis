from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv 
from announcement import announcement #!
from generate_sound import generate_pink_noise, generate_ITS_withnoise
from go_no_go_task import go_no_go, block_stimulus
from baseline_block import play_baseline_block
from its_block import play_its_block
from trial import play_trial_block, main_trial
from two_afc import main_twoafc
from save_file import save_trials_to_csv

#----------------------------------------
#FREQUENCY PARAMS FOR THE AUDIO GENERATION
FS = 44100  
fc = 400 # fixed based on the literature 
fb = 10 # We are gonna change this 
DC = 0.5
FB_LIST = [10, 20, 12, 30, 15, 40]  
FB_TRIAL = [10]
#DURATION OF BLOCKS 
duration_block = 92 #for audio I prefer to have a margin of 2 seconds 
stim_on = 0.250      # image display duration
resp_window = 1.0   # response acceptance window
step = 0.05
ITS_RATIO = 0.1 # initial ITS ratio by default 
#IMAGES
IMG_GO = "white_go_task.jpg"      
IMG_NOGO= ["white_nogo_task1.jpg","white_nogo_task3.jpg","white_nogo_task4.jpg", "white_nogo_task2.jpg"]

#----------------------------------------
#DATA TO BE CHANGED BY THE EXPERIMENTER (THE DATA WILL BE SAVE WITH THE PARTICIPANT_ID AND THE DATE)
EXPERIMENT_MODE = "W"  
PARTICIPANT_ID = "ROSARIO_2"
DATE = "03-03"
# opciones: "X", "Y", "Z", "W"
# X = trial + baseline
# Y = baseline
# Z = 2afc + its, noise
# W = its, noise 
#----------------------------------------

win = visual.Window([1366, 768], color='black', units='pix', fullscr = True)
fr = win.getActualFrameRate()  # 60 hz so each frame is 20 ms 
go_stim = visual.ImageStim(win, image=IMG_GO, size=(200,200), units="pix")
no_go_stim_1 = visual.ImageStim(win, image=IMG_NOGO[0], size=(200,200), units="pix")
no_go_stim_2 = visual.ImageStim(win, image=IMG_NOGO[1], size=(200,200), units="pix")
no_go_stim_3 = visual.ImageStim(win, image=IMG_NOGO[2], size=(200,200), units="pix")
no_go_stim_4 = visual.ImageStim(win, image=IMG_NOGO[3], size=(200,200), units="pix")

no_go_stim = [no_go_stim_1,no_go_stim_2,no_go_stim_3,no_go_stim_4]
fixation = visual.TextStim(win, text="+", color="white", height=40)
A_announ = visual.TextStim(win, text="Interval A", color="white", height=40)
B_announ = visual.TextStim(win, text = "Interval B", color = "white", height = 40)


def main_x(): #trial + baseline
    trials_data = main_trial(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block)
    save_trials_to_csv(trials_data, filename = f"TRIALS_{PARTICIPANT_ID}_{DATE}.csv") 
    baseline_data = play_baseline_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO)
    save_trials_to_csv(baseline_data, filename = f"BASELINE_{PARTICIPANT_ID}_{DATE}.csv")
    #print(f"Saved!, SUCESSFUL_TRIAL_{PARTICIPANT_ID}_{DATE}")
    
def main_y(): #baseline
    baseline_data = play_baseline_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO)
    save_trials_to_csv(baseline_data, filename = f"BASELINE_{PARTICIPANT_ID}_{DATE}.csv")
    
def main_z(): #2afc + its, noise
    final_its_ratio = main_twoafc(win, fr,  FS, fc, A_announ, B_announ, fb, dc=DC, duration_seconds=10, its_ratio=ITS_RATIO, n_trials = 5)
    its_data = play_its_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, FB_LIST, dc=DC, duration_seconds=duration_block, its_ratio=final_its_ratio)
    save_trials_to_csv(its_data, filename = f"ITS_NOISE_{PARTICIPANT_ID}_{DATE}.csv")

def main_w(): # its, noise 
    its_data = play_its_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, FB_LIST, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO)
    save_trials_to_csv(its_data, filename = f"ONLY_ITS_NOISE_{PARTICIPANT_ID}_{DATE}.csv")
    
if __name__ == "__main__": 
    if EXPERIMENT_MODE == "X":
        
        main_x()

    elif EXPERIMENT_MODE == "Y":
        main_y()

    elif EXPERIMENT_MODE == "Z":
        main_z()
        
    elif EXPERIMENT_MODE == "W": 
        main_w()
    else:
        raise ValueError("Unknown experimental scenario")

win.close()
core.quit()