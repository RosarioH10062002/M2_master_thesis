from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv

from announcement import announcement
from generate_sound import generate_pink_noise, generate_ITS_withnoise

def get_A_or_B():  # if its 0 will be Noise and if its 1 will be Noise + ITS
    list_order = [0,1]
    random.shuffle(list_order)
    #print(list_order)
    return list_order
    
def A_B(list_order, duration_seconds, fs, fc, fb, dc, its_ratio):
    array_sound = []
    for item in list_order: 
        if item == 0: 
            # Pink Noise 
            sound = generate_pink_noise(duration_seconds, fs)
        else: 
            # Pink plus ITS
            t, sound = generate_ITS_withnoise(fc, fb, dc, duration_seconds,
                           fs, its_ratio)
        array_sound.append(sound)
        #print(array_sound)
    return array_sound


def run_two_afc(win, fr,  fs, fc, A_announ, B_announ, fb, dc, duration_seconds, its_ratio): 
    flag = None 
    frames_req = int(duration_seconds * fr)
    list_order = get_A_or_B()
    index_its = list_order.index(1) # 1 contains the beat
    array_sound = A_B(list_order, duration_seconds, fs, fc, fb, dc, its_ratio)
    event.clearEvents(eventType='keyboard')
    for index, sound_audio in enumerate(array_sound): 
        audio = sound.Sound(value = sound_audio, sampleRate=fs)
        audio.play()
        
        for frame in range(frames_req): 
            if index == 0: 
                A_announ.draw()
            else: 
                B_announ.draw()
            win.flip()
        audio.stop()
        win.flip()
        core.wait(0.25)
    text = "Which interval had the beat?\n\nPress 'A' or 'B'."
    question = visual.TextStim(win, text = text, color = "White")
    question.draw()
    win.flip()
    event.clearEvents(eventType="keyboard")
    key = event.waitKeys(keyList = ["a", "b"])[0]
    if (key == "a" and index_its == 0) or (key == "b" and index_its == 1):
        flag = True
    else:
        flag = False 
    return flag 

def run_5_trials_twoafc(win, fr,  fs, fc, A_announ, B_announ, fb, dc, duration_seconds, its_ratio, n_trials): # if the user can distinguish 3 out of 5 times then we can say that he distinguished the diference 
    trials_flag = None
    correct_answers = 0
    announcement(win, text = "In each trial, you will hear two sounds: Interval A and Interval B.\n"
         "One of them contains a faint rhythmic beat.\n\n"
         "Your task: decide which interval had the beat.\n\n"
         "Press 'A' for the first sound or 'B' for the second.\n\n"
         "Press space to start.") # should I put the 5 times? like to clarify the participants 
    for n_trial in range(n_trials): 
        flag = run_two_afc(win, fr,  fs, fc, A_announ, B_announ, fb, dc, duration_seconds, its_ratio)
        if flag == True: 
            correct_answers = correct_answers + 1
    if correct_answers >= 3:
        trials_flag = True 
    else: 
        trials_flag = False 
    return trials_flag, correct_answers 
        


def main_twoafc(win, fr,  fs, fc, A_announ, B_announ, fb, dc, duration_seconds, its_ratio, n_trials): 
    trials_flag = None 
    while (trials_flag != True) and (its_ratio <= 1.0): 
        trials_flag, correct_answers = run_5_trials_twoafc(win, fr,  fs, fc, A_announ, B_announ, fb, dc, duration_seconds, its_ratio, n_trials)
        if trials_flag == False or trials_flag == None:
            its_ratio = its_ratio + 0.05 # I want to add 5% of IT/Noise Ratio
        else: 
            break 
    # if they found the correct ITS ratio for themselves so we reduce one step before (-5%)
    its_ratio = its_ratio - 0.05 
    print(f"Correct_answers: {correct_answers}/5, ITS ratio:{its_ratio*100:.1f} %")
    return its_ratio