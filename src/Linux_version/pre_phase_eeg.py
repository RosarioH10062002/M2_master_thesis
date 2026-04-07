from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv 
from announcement import announcement 
#ARTIFACT TRIGGERS --------------------------------------
# Keys = V B N M  
#V: SPEECH / VOCALIZATION [100]
#B: BLINK/EYE MOVEMENT [101]
#N: EXTERNAL NOISE / INTERRUPTION [102]
#M: MOVEMENT (HEAD MOVEMENT, POSTURE CHANGE, MUSCLE TENSION) [103]
#[90-91] : Start/ End Prephase EEG
#[92-93] : Start/ End Postphase EEG
duration = 120 # seconds (300 seconds = 5 minutes)  
def run_pre_phase(win, fixation, marker_outlet, fr, mode): 
    
    frames_req = int(duration * fr)

    announcement(
        win,
        text="EEG PHASE:\n\n"
             "Please fixate on the cross at the center of the screen.\n"
             "When you hear a sound, gently close your eyes.\n\n"
             "Remain still and relaxed."
    )
    if mode == "pre":
        start_marker = 90
        end_marker = 91
    elif mode == "post":
        start_marker = 92
        end_marker = 93
    
    ding_marker = 1003

    event.clearEvents()

    for frame in range(frames_req):

        fixation.draw()

        if frame == 0: 
            win.callOnFlip(marker_outlet.push_sample, [start_marker])

        win.flip()
        
        keys = event.getKeys()

        for key in keys: 
            if key == "v": 
                marker_outlet.push_sample([100])
            elif key == "b": 
                marker_outlet.push_sample([101])
            elif key == "n": 
                marker_outlet.push_sample([102])
            elif key == "m": 
                marker_outlet.push_sample([103])
            elif key == "escape": 
                core.quit()
        if frame == int(frames_req/2): 
            #message_close_eyes = visual.TextStim(win, text = "Please Close your eyes, you will hear a sound once you can open again. ", color = "white")
            #message_close_eyes.draw()
            #win.flip()
            marker_outlet.push_sample([ding_marker])
    marker_outlet.push_sample([end_marker])
    marker_outlet.push_sample([ding_marker])

    if mode == "pre": 
        announcement(win, text="Thank you, you will continue with the next phase.")
    elif mode == "post":
        announcement(win, text="Thank you, the experiment is completed.")
    

        
