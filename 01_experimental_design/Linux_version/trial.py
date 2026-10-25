from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv

from announcement import announcement
from generate_sound import generate_pink_noise, generate_ITS_withnoise
from go_no_go_task import go_no_go, block_stimulus, block_stimulus_trial

target_probability = 0.2

def compute_accuracy(correct_counter): 
    return correct_counter/(45) # Higher probability to change since if they let the trial without pressing a buttom, thye will get 80% of accuracy 
    
def play_trial_block(win,fr,go_stim, no_go_stim,fixation,fs, fc, fb, dc, duration_seconds, right_sound, wrong_sound, right_img, wrong_img):
    announcement(win, text = "INSTRUCTIONS:\n\n"
        "Do NOT press any key when you see the pain au chocolat.\n\n"
        "Press the SPACEBAR only for any other figure.\n"
        "Keep your eyes on the center of the screen.\n\n"
        "Press SPACE to begin.")
 
    #audio_array = generate_pink_noise(duration = duration_seconds, fs = fs)
    trials_data, correct_counter = block_stimulus_trial(win,fr,go_stim, no_go_stim,fixation,fs, block_index = None, right_sound = right_sound, wrong_sound = wrong_sound, right_img = right_img, wrong_img = wrong_img)
    trial_acc = compute_accuracy(correct_counter)
    # I want to test if they a) can distinguished the go task = pan_au_chocolat b) they can perform at least 50% perfom well in the go/no-go task
    return trials_data, trial_acc
def main_trial(win,fr,go_stim, no_go_stim,fixation,fs, fc, fb, dc, duration_seconds, right_sound, wrong_sound, right_img, wrong_img): 
    trials = 0 
    # The user will have 3 trials to reach the minimum accuracy before being discarted as a possible subject of study 
    for trial in range(3): 
        trials_data, trial_acc = play_trial_block(win,fr,go_stim, no_go_stim,fixation,fs, fc, fb, dc, duration_seconds, right_sound, wrong_sound, right_img, wrong_img)
        if trial_acc > 0.5:
            event.clearEvents(eventType='keyboard')
            announcement(win, text = f"Trial Session Sucessful. \n Overall accuracy: {trial_acc*100:.1f}%. \n Press SPACE to continue.")
            break
        else:
            if trial != 2:
                event.clearEvents(eventType='keyboard')
                announcement(win, text = f"Trial Session Unsucessful. \n Overall accuracy: {trial_acc*100:.1f}%. \n You have {3-(trial+1)} trials left. \n Press SPACE to try again.")
            else: 
                event.clearEvents(eventType='keyboard')
                announcement(win, text = "The experiment has finished.\n Please call the experimenter.")
                core.quit()
    return trials_data # the sucessful trial data 