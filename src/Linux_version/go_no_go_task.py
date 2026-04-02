from psychopy import visual, event, core, data, gui
import pandas as pd
import numpy as np
import random
import csv
from announcement import announcement

wrong_sound = "wrong.wav"
right_sound = "right.wav"
FS = 44100

stim_on = 0.250
stim_off = 1 - stim_on

stim_on_trial = 0.250
stim_off_trial = 1 - stim_on_trial
extra_window_trial = 0.6

trials = 90
trials_trial = 45
target_probability = 0.2

trials_data = []


def load_feedback_sounds(fs=FS, right_file=right_sound, wrong_file=wrong_sound):
    right_audio = sound.Sound(value=right_file, sampleRate=fs)
    wrong_audio = sound.Sound(value=wrong_file, sampleRate=fs)
    return right_audio, wrong_audio

def get_no_go(n): 
    idx = random.randint(0,n-1) # 4 stimulus 
    return idx

def get_random_stimulus(trials): # 0 = no_go, 1 = go 
    n_go = round(trials * target_probability)
    n_no_go = trials - n_go
    trial_types = np.array([1]*n_go + [0]*n_no_go)
    np.random.shuffle(trial_types)
    return trial_types

def go_no_go(win,fr, stim, fixation, stimulus, marker_outlet):
    correct_var = None 
    frames_req_on = int(stim_on * fr)
    frames_req_off = int(stim_off * fr)
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
            #win.callOnFlip(marker_outlet.push_sample, [0])  # trial start
            if stimulus == 1: #NO GO
                win.callOnFlip(marker_outlet.push_sample, [1])
            elif stimulus == 0: # GO
                win.callOnFlip(marker_outlet.push_sample, [2])           
        
        win.flip()
            
        if resp_key is None:
            keys= event.getKeys(keyList = ["space"], timeStamped = rt_clock)
            if keys: 
                resp_key, rt = keys[0]
                marker_outlet.push_sample([3]) #PRESS A KEY 

    for frame in range(frames_req_off): 
        fixation.draw()
        win.flip()
        if resp_key is None:
            keys= event.getKeys(keyList = ["space"], timeStamped = rt_clock)
            if keys: 
                resp_key, rt = keys[0]
                marker_outlet.push_sample([3]) #PRESS A KEY    
    fixation.draw()
    #win.callOnFlip(marker_outlet.push_sample, [4])
    win.flip()
    
    if stimulus == 0 and resp_key == "space": 
        correct_var = "Correct"
    elif stimulus == 1 and resp_key is None: 
        correct_var = "Correct"
    else:
        correct_var = "Incorrect"
    return resp_key, rt, correct_var, trial_onset[0]
    
def block_stimulus(win,fr, go_stim,no_go_stim,fixation,audio_array,FS, block_index, fc, fb, marker_outlet): 
    correct_counter = 0 
    list_stimulus = get_random_stimulus(trials)
    #audio = sound.Sound(value = audio_array,sampleRate=FS)
    #audio.play()
    marker_outlet.push_sample([1000]) ###
    for index, stimulus in enumerate(list_stimulus): #stimulus = 0 means NOGO, #stimulus = 1 means GO, 90 stimulus 
        if stimulus == 0: 
            n = get_no_go(len(no_go_stim))
            stim = no_go_stim[n]
            resp_key, rt, correct_var, trial_onset = go_no_go(win,fr, stim, fixation, stimulus, marker_outlet)
            trials_data.append({
            "block_index": block_index,
            "trial": index,
            "trial_onset(s)": trial_onset,
            "stimulus_type": "Go",
            "stimulus_code": 0,
            "resp_key": resp_key,
            "rt(s)": rt,
            "fc(hz)": fc, 
            "fb(hz)": fb, 
            "correct": correct_var
            })
        elif stimulus == 1: 
            stim = go_stim
            resp_key, rt, correct_var, trial_onset = go_no_go(win,fr, stim, fixation, stimulus, marker_outlet)
            trials_data.append({
            "block_index": block_index,
            "trial": index,
            "trial_onset(s)": trial_onset,
            "stimulus_type": "No_Go",
            "stimulus_code": 1,
            "resp_key": resp_key,
            "rt(s)": rt,
            "fc(hz)": fc, 
            "fb(hz)": fb,
            "correct": correct_var
            })
        
        if correct_var == "Correct": 
            correct_counter = correct_counter + 1
            
    #audio.stop()
    #marker_outlet.push_sample([1001]) ###
    return trials_data, correct_counter
 
def display_sound(win, resp_key, stimulus, right_audio, wrong_audio, FS, right_img, wrong_img):
    if (stimulus == 0 and resp_key == "space") or (stimulus == 1 and resp_key is None): 
        #correct_var = "Correct"
        #audio = sound.Sound(value = right_sound, sampleRate = FS)
        right_audio.play()
        right_img.draw()
    else:
        #correct_var = "Incorrect"
        #audio = sound.Sound(value = wrong_sound, sampleRate = FS)
        wrong_audio.play()
        wrong_img.draw()
    win.flip()
    
def go_no_go_trial(win,fr, stim, fixation, stimulus, right_sound, wrong_sound, fs, right_img, wrong_img):
    correct_var = None 
    frames_req_on = int(stim_on_trial * fr)
    frames_req_off = int(stim_off_trial * fr)
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
                display_sound(win, resp_key, stimulus, right_audio, wrong_audio, FS, right_img, wrong_img)
                #audio = display_sound(resp_key, stimulus, right_sound, wrong_sound, fs)
                #audio.play()
        #display_sound(resp_key, stimulus, right_audio, wrong_audio, FS)
    for frame in range(frames_req_off): 
        fixation.draw()
        win.flip()
        if resp_key is None:
            keys= event.getKeys(keyList = ["space"], timeStamped = rt_clock)
            if keys:
                resp_key, rt = keys[0]
                display_sound(win, resp_key, stimulus, right_audio, wrong_audio, FS, right_img, wrong_img)
                #audio = display_sound(resp_key, stimulus, right_sound, wrong_sound, fs)
                #audio.play()
        #display_sound(resp_key, stimulus, right_audio, wrong_audio, FS)
    #display_sound(resp_key, stimulus, right_audio, wrong_audio, FS)
    if resp_key is None: 
        win.flip()
        display_sound(win, resp_key, stimulus, right_audio, wrong_audio, FS, right_img, wrong_img)
        core.wait(extra_window_trial) # 0.6
        win.flip()
        core.wait(0.4)
        
    if (stimulus == 0 and resp_key == "space") or (stimulus == 1 and resp_key is None): 
        correct_var = "Correct"
        #right_audio.play()
    else:
        correct_var = "Incorrect"
        #wrong_audio.play()
    return resp_key, rt, correct_var, trial_onset[0]


def block_stimulus_trial(win,fr, go_stim,no_go_stim,fixation,FS, block_index, right_sound, wrong_sound, right_img, wrong_img): 
    correct_counter = 0 
    list_stimulus = get_random_stimulus(trials_trial)
    #audio = sound.Sound(value = audio_array,sampleRate=FS)
    #audio.play()
    for index, stimulus in enumerate(list_stimulus): #stimulus = 0 means NOGO, #stimulus = 1 means GO, 90 stimulus 
        if stimulus == 0: 
            n = get_no_go(len(no_go_stim))
            stim = no_go_stim[n]
            resp_key, rt, correct_var, trial_onset = go_no_go_trial(win,fr, stim, fixation, stimulus, right_sound, wrong_sound, FS, right_img, wrong_img)
            #if correct_var == "Correct": 
            #    audio = sound.Sound(value = right_sound, sampleRate = FS)
            #elif correct_var == "Incorrect": 
            #    audio = sound.Sound(value = wrong_sound, sampleRate = FS)
            #audio.play()
            #core.wait()
            #stop.play()
            trials_data.append({
            "block_index": block_index,
            "trial": index,
            "trial_onset(s)": trial_onset,
            "stimulus_type": "No_go",
            "stimulus_code": 0,
            "resp_key": resp_key,
            "rt(s)": rt,
            "fc(hz)": None,
            "fb(hz)": None,
            "correct": correct_var
            })
        elif stimulus == 1: 
            stim = go_stim
            resp_key, rt, correct_var, trial_onset = go_no_go_trial(win,fr, stim, fixation, stimulus, right_sound, wrong_sound, FS, right_img, wrong_img)
            #if correct_var == "Correct": 
            #    audio = sound.Sound(value = right_sound, sampleRate = FS)
            #elif correct_var == "Incorrect": 
            #    audio = sound.Sound(value = wrong_sound, sampleRate = FS)
            #audio.play()
            trials_data.append({
            "block_index": block_index,
            "trial": index,
            "trial_onset(s)": trial_onset,
            "stimulus_type": "Go",
            "stimulus_code": 1,
            "resp_key": resp_key,
            "rt(s)": rt,
            "fc(hz)": None,
            "fb(hz)": None,
            "correct": correct_var
            })
        
        if correct_var == "Correct": 
            correct_counter = correct_counter + 1
            
    #audio.stop()
    return trials_data, correct_counter
    #event.waitKeys(keyList = ["space"])
    #idx = get_no_go(win)
    #off_stimulus = visual.ImageStim(win, image = no_go_task[idx], size = (200,200), units = "pix")

    #event.waitKeys(keyList = ["space"])
    
    #event.waitKeys(keyList = ["space"])
    

