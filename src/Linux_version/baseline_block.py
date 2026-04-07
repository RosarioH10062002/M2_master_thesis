from psychopy import visual, event, core, data, gui
import pandas as pd 
import numpy as np 
import random 
import csv 

from announcement import announcement
from generate_sound import generate_pink_noise, generate_ITS_withnoise
from go_no_go_task import go_no_go, block_stimulus

n_blocks = 6; # I would like to have 6 blocks of 90 seconds 

time_break = 20 #seconds
def play_baseline_block(win,fr,go_stim, no_go_stim,fixation,fs, fc, fb, dc, duration_seconds, its_ratio, marker_outlet):
    frames_req_break = int(time_break * fr)
    announcement(win, text = "INSTRUCTIONS:\n\n"
        "Do NOT press any key when you see the pain au chocolat.\n\n"
        "Press the SPACEBAR only for any other figure.\n"
        "Keep your eyes on the center of the screen.\n\n"
        "Press SPACE to begin.")
    for block_index in range(n_blocks): 
        audio_array = generate_pink_noise(duration = duration_seconds, fs = fs)
        trials_data, correct_counter = block_stimulus(win,fr,go_stim, no_go_stim,fixation,audio_array,fs, block_index+1,fc = None,fb = None, marker_outlet = marker_outlet)
        if block_index != (n_blocks-1):
            for t in range(time_break, 0, -1):  
                for frame in range(int(fr)):  
                    event.clearEvents(eventType='keyboard')

                    break_announ = visual.TextStim(
                        win,
                        text=f"Block {block_index+1}/{n_blocks} complete.\n\n"
                             f"Take a short rest\n\n"
                             f"{t} seconds left",
                        color="white"
                    )
                    break_announ.draw()
                    win.flip()
            announcement(win, text=f"Press space to continue.")
        else:
            event.clearEvents(eventType='keyboard')
            announcement(win, text=f"Block {block_index+1}/{n_blocks} complete.\nPress space to continue.")
    announcement(win, text="All blocks complete.\n Thank you!")
    return trials_data

