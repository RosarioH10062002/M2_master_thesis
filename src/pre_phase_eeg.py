from psychopy import visual, event, core, data, gui, sound 
import pandas as pd 
import numpy as np 
import random 
import csv 
from announcement import announcement 
#ARTIFACT TRIGGERS --------------------------------------
# Keys = V B N M 
#V: SPEECH / VOCALIZATION
#B: BLINK/EYE MOVEMENT 
#N: EXTERNAL NOISE / INTERRUPTION
#M: MOVEMENT (HEAD MOVEMENT, POSTURE CHANGE, MUSCLE TENSION)
duration = 60 # seconds (300 seconds = 5 minutes)  
def run_pre_phase(win, fixation, marker_outlet, fr, mode): 
    
    frames_req = int(duration * fr)

    announcement(win, text="Please look carefully at the cross +. Stay still and relax.")

    if mode == "pre":
        start_marker = 90
        end_marker = 91
    elif mode == "post":
        start_marker = 92
        end_marker = 93

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

    marker_outlet.push_sample([end_marker])

    if mode == "pre": 
        announcement(win, text="Thank you, you will continue with the next phase.")
    elif mode == "post":
        announcement(win, text="Thank you, the experiment is completed.")
        
