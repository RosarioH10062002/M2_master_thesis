from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv 

from announcement import announcement
from generate_sound import generate_pink_noise, generate_ITS_withnoise
from go_no_go_task import go_no_go, block_stimulus

def get_list_fb(fb_list): 
    random.shuffle(fb_list)
    print(fb_list)
    return fb_list
def play_its_block(win,fr,go_stim, no_go_stim,fixation,fs, fc, fb_list, dc, duration_seconds, its_ratio):
    announcement(win, text = "INSTRUCTIONS:\n\n"
        "Do NOT press any key when you see the pain au chocolat.\n\n"
        "Press the SPACEBAR only for any other figure.\n"
        "Keep your eyes on the center of the screen.\n\n"
        "Press SPACE to begin.")
    fb_list_random = get_list_fb(fb_list)
    for block_index,fb in enumerate(fb_list_random): 
        t, audio_array = generate_ITS_withnoise(fc, fb, dc, duration_seconds,fs, its_ratio)
        trials_data, correct_counter = block_stimulus(win,fr,go_stim, no_go_stim,fixation,audio_array,fs, block_index+1, fc, fb)
        if block_index != len(fb_list_random)-1:
            event.clearEvents(eventType='keyboard')
            announcement(win, text=f"Block {block_index+1}/{len(fb_list_random)} complete.\n\nTake a short rest.\n\nPress space to continue.")
        else:
            event.clearEvents(eventType='keyboard')
            announcement(win, text=f"Block {block_index+1}/{len(fb_list_random)} complete.\nPress space to continue.")
    announcement(win, text="All blocks complete.\n Thank you!")
    return trials_data

