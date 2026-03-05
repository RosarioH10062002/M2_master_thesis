from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv 
from announcement import announcement 

stim_on = 0.250      # image display duration
stim_off = 1 - stim_on
trials = 90 
target_probability = 0.2

trials_data = []

def get_no_go(n): 
    idx = random.randint(0,n-1) # 4 stimulus 
    return idx

def get_random_stimulus(): # 0 = no_go, 1 = go 
    n_go = round(trials * target_probability)
    n_no_go = trials - n_go
    trial_types = np.array([1]*n_go + [0]*n_no_go)
    np.random.shuffle(trial_types)
    return trial_types

def go_no_go(win,fr, stim, fixation, stimulus):
    correct_var = None 
    frames_req_on = int(stim_on * fr) + 1
    frames_req_off = int(stim_off * fr) + 1
    event.clearEvents(eventType='keyboard') # to clear the buffer 
    rt_clock = core.Clock() # my stopwatch 
    resp_key = None
    rt = None 
    trial_onset = [None]
    for frame in range(frames_req_on): 
        stim.draw()
        if frame == 0:
            win.callOnFlip(event.clearEvents, eventType='keyboard') # scheduled right when it flips the window
            win.callOnFlip(rt_clock.reset)
            win.callOnFlip(lambda: trial_onset.__setitem__(0, core.getTime())) #SUPER IMPORTANT! Because allow me to extract the time exaclty when the stimulus appear. it takes the gloabl clock time (starts right before the time)

        win.flip()
        if resp_key is None:
            keys= event.getKeys(keyList = ["space"], timeStamped = rt_clock)
            if keys: 
                resp_key, rt = keys[0]

    for frame in range(frames_req_off): 
        fixation.draw()
        win.flip()
        if resp_key is None:
            keys= event.getKeys(keyList = ["space"], timeStamped = rt_clock)
            if keys: 
                resp_key, rt = keys[0]
    if stimulus == 1 and resp_key == "space": 
        correct_var = "Correct"
    elif stimulus == 0 and resp_key is None: 
        correct_var = "Correct"
    else:
        correct_var = "Incorrect"
    return resp_key, rt, correct_var, trial_onset[0]
    
def block_stimulus(win,fr, go_stim,no_go_stim,fixation,audio_array,FS, block_index, fc, fb): 
    correct_counter = 0 
    list_stimulus = get_random_stimulus()
    audio = sound.Sound(value = audio_array,sampleRate=FS)
    audio.play()
    for index, stimulus in enumerate(list_stimulus): #stimulus = 0 means NOGO, #stimulus = 1 means GO
        if stimulus == 0: 
            n = get_no_go(len(no_go_stim))
            stim = no_go_stim[n]
            resp_key, rt, correct_var, trial_onset = go_no_go(win,fr, stim, fixation, stimulus)
            trials_data.append({
            "block_index": block_index,
            "trial": index,
            "trial_onset(s)": trial_onset,
            "stimulus_type": "No_go",
            "stimulus_code": 0,
            "resp_key": resp_key,
            "rt(s)": rt,
            "fc(hz)": fc, 
            "fb(hz)": fb, 
            "correct": correct_var
            })
        elif stimulus == 1: 
            stim = go_stim
            resp_key, rt, correct_var, trial_onset = go_no_go(win,fr, stim, fixation, stimulus)
            trials_data.append({
            "block_index": block_index,
            "trial": index,
            "trial_onset(s)": trial_onset,
            "stimulus_type": "Go",
            "stimulus_code": 1,
            "resp_key": resp_key,
            "rt(s)": rt,
            "fc(hz)": fc, 
            "fb(hz)": fb,
            "correct": correct_var
            })
        
        if correct_var == "Correct": 
            correct_counter = correct_counter + 1
            
    audio.stop()
    return trials_data, correct_counter
    

            
    #event.waitKeys(keyList = ["space"])
    #idx = get_no_go(win)
    #off_stimulus = visual.ImageStim(win, image = no_go_task[idx], size = (200,200), units = "pix")

    #event.waitKeys(keyList = ["space"])
    
    #event.waitKeys(keyList = ["space"])
    

