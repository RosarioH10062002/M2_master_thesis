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
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet
from mini_version import play_baseline_block_mini_trial
from pre_phase_eeg import run_pre_phase

#----------------------------------------
#FREQUENCY PARAMS FOR THE AUDIO GENERATION
FS = 44100  
fc = 400 # fixed based on the literature 
fb = 10 # We are gonna change this 
DC = 0.5
FB_LIST = [10, 20, 12, 30, 15, 40]  
FB_TRIAL = [10]
#DURATION OF BLOCKS 
duration_block = 91 #for audio I prefer to have a margin of 2 seconds 
stim_on = 0.250      # image display duration
resp_window = 1.0   # response acceptance window
step = 0.05
ITS_RATIO = 0.15 # initial ITS ratio by default 
#IMAGES
IMG_GO = "white_go_task.jpg"      
IMG_NOGO= ["white_nogo_task1.jpg","white_nogo_task3.jpg","white_nogo_task4.jpg", "white_nogo_task2.jpg"]

IMG_Check = "check.png"
IMG_no_check = "wrong.png"

WRONG_SOUND = "wrong.wav"
RIGHT_SOUND = "right.wav"

#----------------------------------------
#TO SEND THE MARKERS 
marker_info = StreamInfo(
    name='Psychopy_markers',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format='int32'
)

marker_outlet = StreamOutlet(marker_info)
#print("Marker is mounted")
#marker_outlet = False 
#----------------------------------------
#DATA TO BE CHANGED BY THE EXPERIMENTER (THE DATA WILL BE SAVE WITH THE PARTICIPANT_ID AND THE DATE)
EXPERIMENT_MODE = "COMPLETE" 
PARTICIPANT_ID = "trial"
#DATE = "03-03"
# opciones: "X", "Y", "Z", "W"
# X = trial + baseline **
# Y = baseline
# Z = 2afc + its, noise **
# W = its, noise 
# M = Mini trial 
# PP = Pre/Post phase + Mini trial
#----------------------------------------
win = visual.Window([1536, 864], color='black', units='pix')
#win = visual.Window([1536, 864], color='black', units='pix', fullscr = True)
fr = win.getActualFrameRate()  # 60 hz so each frame is 20 ms 
go_stim = visual.ImageStim(win, image=IMG_GO, size=(200,200), units="pix")
no_go_stim_1 = visual.ImageStim(win, image=IMG_NOGO[0], size=(200,200), units="pix")
no_go_stim_2 = visual.ImageStim(win, image=IMG_NOGO[1], size=(200,200), units="pix")
no_go_stim_3 = visual.ImageStim(win, image=IMG_NOGO[2], size=(200,200), units="pix")
no_go_stim_4 = visual.ImageStim(win, image=IMG_NOGO[3], size=(200,200), units="pix")

img_check = visual.ImageStim(win, image=IMG_Check, pos=(0, 0), size = (200,200))
img_no_check = visual.ImageStim(win, image = IMG_no_check, pos = (0,0), size = (200,200))

no_go_stim = [no_go_stim_1,no_go_stim_2,no_go_stim_3,no_go_stim_4]
fixation = visual.TextStim(win, text="+", color="white", height=40)
A_announ = visual.TextStim(win, text="Interval A", color="white", height=40)
B_announ = visual.TextStim(win, text = "Interval B", color = "white", height = 40)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def main_x(): #trial + baseline
    trials_data = main_trial(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, right_sound = RIGHT_SOUND, wrong_sound = WRONG_SOUND, right_img = img_check, wrong_img = img_no_check)
    save_trials_to_csv(trials_data, filename = f"TRIALS_{PARTICIPANT_ID}_{TIMESTAMP}.csv") 
    baseline_data = play_baseline_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO)
    save_trials_to_csv(baseline_data, filename = f"BASELINE_{PARTICIPANT_ID}_{TIMESTAMP}.csv") 
    print(f"Saved!, SUCESSFUL_TRIAL_{PARTICIPANT_ID}_{TIMESTAMP}")
    
    
def main_y(): #baseline
    baseline_data = play_baseline_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO, marker_outlet = marker_outlet)
    save_trials_to_csv(baseline_data, filename = f"ONLY_BASELINE_{PARTICIPANT_ID}_{TIMESTAMP}.csv")
    
def main_z(): #2afc + its, noise
    final_its_ratio = main_twoafc(win, fr,  FS, fc, A_announ, B_announ, fb, dc=DC, duration_seconds=8, its_ratio=ITS_RATIO, n_trials = 5)
    its_data = play_its_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, FB_LIST, dc=DC, duration_seconds=duration_block, its_ratio=final_its_ratio)
    save_trials_to_csv(its_data, filename = f"ITS_NOISE_{PARTICIPANT_ID}_{TIMESTAMP}.csv")

def main_w(): # its, noise 
    its_data = play_its_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, FB_LIST, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO)
    save_trials_to_csv(its_data, filename = f"ONLY_ITS_NOISE_{PARTICIPANT_ID}_{TIMESTAMP}.csv")

def main_m():
    #marker_outlet = StreamOutlet(marker_info)
    baseline_data = play_baseline_block_mini_trial(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO, marker_outlet = marker_outlet)
    save_trials_to_csv(baseline_data, filename = f"MINI_TRIAL_{PARTICIPANT_ID}_{TIMESTAMP}.csv")

def main_pp(): 
    run_pre_phase(win, fixation, marker_outlet = marker_outlet, fr = fr, mode = "pre")
    baseline_data = play_baseline_block_mini_trial(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO, marker_outlet = marker_outlet)
    save_trials_to_csv(baseline_data, filename = f"MINI_TRIAL_{PARTICIPANT_ID}_{TIMESTAMP}.csv")
    run_pre_phase(win, fixation, marker_outlet = marker_outlet, fr = fr, mode = "post")
    
def main_complete(): 
    run_pre_phase(win, fixation, marker_outlet = marker_outlet, fr = fr, mode = "pre")
    baseline_data = play_baseline_block(win,fr,go_stim, no_go_stim,fixation,FS, fc, fb, dc=DC, duration_seconds=duration_block, its_ratio=ITS_RATIO, marker_outlet = marker_outlet)
    save_trials_to_csv(baseline_data, filename = f"ONLY_BASELINE_{PARTICIPANT_ID}_{TIMESTAMP}.csv")
    run_pre_phase(win, fixation, marker_outlet = marker_outlet, fr = fr, mode = "post")
    
if __name__ == "__main__": 
    if EXPERIMENT_MODE == "X":
        
        main_x()

    elif EXPERIMENT_MODE == "Y":
        main_y()

    elif EXPERIMENT_MODE == "Z":
        main_z()
        
    elif EXPERIMENT_MODE == "W": 
        main_w()
        
    elif EXPERIMENT_MODE == "M":
        main_m()
        
    elif EXPERIMENT_MODE == "PP":
        main_pp()
    
    elif EXPERIMENT_MODE == "COMPLETE":
        main_complete()
        
    else:
        raise ValueError("Unknown experimental scenario")

win.close()
core.quit()