from psychopy import visual, event, core, data, gui, sound 
def announcement(win, text):
    message = visual.TextStim(win, text = text, color = "White")
    message.draw()
    win.flip()
    key = event.waitKeys(keyList = ["space", "escape"])[0]
    if key == "escape":  
        #marker_outlet.push_sample([1002])
        core.quit()
    #win.close()
