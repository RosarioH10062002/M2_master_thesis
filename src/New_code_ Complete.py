# LIBRARIES
import numpy as np
from psychopy import visual, sound, event, core  
import csv
#----------------------------------------------------------------------------
#FIXED VARS 
FS = 44100  
fc = 400 # fixed based on the literature 
fb = 10 # We are gonna change this 
FB_LIST = [10, 20, 12, 30, 15, 40]  
FB_TRIAL = [10]
IMG_GO = "white_go_task.jpg"      
IMG_NOGO_FILES = [
    "white_nogo_task1.jpg",
    "white_nogo_task3.jpg",
    "white_nogo_task4.jpg", 
    "white_nogo_task2.jpg"
]

#----------------------------------------------------------------------------
#FUNCTIONS 
def compute_go_accuracy(trials):
    go_trials = [t for t in trials if t.get('is_nogo') == 0]  # <-- FIXED

    if not go_trials:
        return 0.0

    go_correct = sum(t['correct'] for t in go_trials)
    return go_correct / len(go_trials)



def rms(x):
    return np.sqrt(np.mean(x**2))
#-
def set_rms(x, target_rms, eps=1e-12):
    r = rms(x)
    if r < eps:
        return x
    return x * (target_rms / r)
#-
def generate_pink_noise(duration, fs=FS):
    N = int(duration * fs)

    x = np.random.randn(N)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # 1/sqrt(f) =  pink noise 
    shaping = np.ones_like(freqs)
    shaping[1:] = 1.0 / np.sqrt(freqs[1:])
    Xpink = X * shaping

    pink = np.fft.irfft(Xpink, n=N) # time domain

   
    pink = pink / (np.max(np.abs(pink)) + 1e-12)

    return pink.astype(np.float32)
#-
def generate_ITS_withnoise(fc, fb, dc=0.5, duration_seconds=90,
                           fs=FS, its_ratio=0.05):

    t = np.arange(0, duration_seconds, 1/fs)

    # ----- ITS -----
    carrier = np.sin(2 * np.pi * fc * t)
    envelope = (np.sign(np.sin(2 * np.pi * fb * t)) + 1) / 2
    envelope = (envelope > (1 - dc)).astype(float)

    its = carrier * envelope
    its = its / np.max(np.abs(its))

    # ----- Pink noise -----
    noise = generate_pink_noise(duration_seconds, fs)

    # ensure same length
    N = min(len(its), len(noise))
    its = its[:N]
    noise = noise[:N]

    # ----- Mix -----
    mix = its_ratio * its + (1 - its_ratio) * noise # its_ratio is the ratio between ITS/NOISE
    #mix = mix / np.max(np.abs(mix))

    return t[:N], mix.astype(np.float32)

# 2AFC = Two alternative Forced Choice 
# ---------- MAIN ----------
win = visual.Window([1366, 768], color='black', units='pix')

instr_text = visual.TextStim(
    win,
    text="In each trial, you will hear two sounds: Interval A and Interval B.\n"
         "One of them contains a faint rhythmic beat.\n\n"
         "Your task: decide which interval had the beat.\n\n"
         "Press 'A' for the first sound or 'B' for the second.\n\n"
         "Press any key to start.",
    color='white', height=24
)

question_text = visual.TextStim(
    win,
    text="Which interval had the beat?\n\nPress 'A' or 'B'.",
    color='white', height=30
)

label_A = visual.TextStim(
    win,
    text="Interval A",
    color='white', height=40
)

label_B = visual.TextStim(
    win,
    text="Interval B",
    color='white', height=40
)
#-
def wait_for_any_key():
    keys = event.waitKeys()
    if 'escape' in keys:
        win.close()
        core.quit()
#-
def play_interval(stim_array, duration):
    snd = sound.Sound(value=stim_array, sampleRate=FS)
    snd.play()
    core.wait(duration)  # hold script so sound can finish
#-
def run_afc_block(its_ratio, fc=400, fb=10,
                  n_trials=5, interval_duration=2.0,
                  show_instructions=True, dc=0.5,
                  TARGET_RMS=0.02):

    correct_count = 0
    afc_trials = []


    if show_instructions:
        instr_text.draw()
        win.flip()
        wait_for_any_key()

    for trial in range(n_trials):
        its_in_A = np.random.rand() < 0.5  # True = A, False = B

        # ------------------------------------------------------------
        # 1) Generate ONE noise sample for this trial (shared base)
        noise_only = generate_pink_noise(interval_duration, FS)

        # 2) Generate ITS ONLY (no noise inside)
        t = np.arange(0, interval_duration, 1/FS)
        carrier = np.sin(2 * np.pi * fc * t)

        envelope = (np.sign(np.sin(2 * np.pi * fb * t)) + 1) / 2
        envelope = (envelope > (1 - dc)).astype(float)

        its_only = carrier * envelope
        its_only = its_only / (np.max(np.abs(its_only)) + 1e-12)


        N = min(len(noise_only), len(its_only))
        noise_only = noise_only[:N]
        its_only = its_only[:N]

        stim_its = (1 - its_ratio) * noise_only + its_ratio * its_only

        noise_only = set_rms(noise_only, TARGET_RMS)
        stim_its   = set_rms(stim_its,   TARGET_RMS)

        noise_only = np.clip(noise_only, -1, 1).astype(np.float32)
        stim_its   = np.clip(stim_its,   -1, 1).astype(np.float32)
        # ------------------------------------------------------------

        # Interval A / B assignment
        if its_in_A:
            stim_A = stim_its
            stim_B = noise_only
        else:
            stim_A = noise_only
            stim_B = stim_its

        # Interval A 
        label_A.draw()
        win.flip()
        core.wait(0.3)
        play_interval(stim_A, interval_duration)
        win.flip()
        core.wait(0.3)

        #Interval B 
        label_B.draw()
        win.flip()
        core.wait(0.3)
        play_interval(stim_B, interval_duration)
        win.flip()

                # ----- Ask for response -----
        question_text.draw()
        win.flip()

        resp_clock = core.Clock()
        resp_key = event.waitKeys(keyList=['a', 'b', 'escape'], timeStamped=resp_clock)[0]
        key, rt = resp_key  # key is 'a'/'b', rt is seconds

        if key == 'escape':
            win.close()
            core.quit()

        # correctness
        correct_interval = 1 if its_in_A else 2
        response_interval = 1 if key == 'a' else 2
        correct = int(response_interval == correct_interval)

        if correct:
            correct_count += 1

        # LOG this trial
        afc_trials.append({
            "trial_index": trial,
            "its_ratio": its_ratio,
            "fc": fc,
            "fb": fb,
            "interval_duration": interval_duration,
            "its_in_A": int(its_in_A),
            "correct_interval": correct_interval,
            "response_interval": response_interval,
            "response_key": key,
            "rt": rt,
            "correct": correct
        })


    prop_correct = correct_count / n_trials
    return prop_correct, afc_trials


# ----- VISUAL STIMULI -----
fixation = visual.TextStim(win, text="+", color='white', height=40)


# GO 
standard_stim = visual.ImageStim(
    win,
    image=IMG_GO,
    size=(200, 200),
    units="pix"
)

# NO-GO
nogo_stims = [
    visual.ImageStim(
        win,
        image=img_file,
        size=(200, 200),
        units="pix"
    )
    for img_file in IMG_NOGO_FILES
]



block_instr = visual.TextStim(
    win,
    text=(
        "INSTRUCTIONS:\n\n"
        "Press the SPACEBAR only when you see the pan au chocolat.\n"
        "Do NOT press any key for any other figure.\n\n"
        "Keep your eyes on the center of the screen.\n\n"
        "Press SPACE to begin."
    ),
    color='white', height=24
)

def play_baseline_block(fs, duration_seconds, target_rms=0.02):
    
    #_, audio_block = generate_ITS_withnoise(
    #    fc=fc,
    #    fb=fb,
    #    duration_seconds=duration_seconds,
    #    its_ratio=its_ratio
    #)
    noise = generate_pink_noise(duration = duration_seconds, fs=fs)
    audio_block = noise 
    
    

    # Match loudness 
    audio_block = set_rms(audio_block, target_rms)
    audio_block = np.clip(audio_block, -1, 1).astype(np.float32)

    audio_stim = sound.Sound(value=audio_block, sampleRate=FS)
    audio_stim.play()
    return audio_stim



def play_its_block(fc, fb, its_ratio, duration_seconds, target_rms=0.02):
    _, audio_block = generate_ITS_withnoise(
        fc=fc,
        fb=fb,
        duration_seconds=duration_seconds,
        its_ratio=its_ratio
    )
    
    

    # Match loudness 
    audio_block = set_rms(audio_block, target_rms)
    audio_block = np.clip(audio_block, -1, 1).astype(np.float32)

    audio_stim = sound.Sound(value=audio_block, sampleRate=FS)
    audio_stim.play()
    return audio_stim


#----------------------------------------------------------------------------
def run_gonogo_block(fb, its_ratio, fc=400,
                     block_duration=90.0,
                     stim_on=0.250,
                     trial_period=1.0,
                     target_prob=0.2,
                     block_index=0, 
                     show_instructions=True,
                     show_end_screen=True):

    
   
    block_instr.text = (
        f"Instructions:\n\n"
        "Press the SPACEBAR only when you see the pan au chocolat.\n"
        "Do NOT press any key for any other figure.\n\n"
        "Keep your eyes on the center of the screen.\n\n"
        "Press SPACE to begin."
    )
    if show_instructions:
        block_instr.draw()
        win.flip()
        wait_for_any_key()


    # ITS + Noise 
    audio_stim = play_its_block(fc=fc, fb=fb, its_ratio=its_ratio,
                            duration_seconds=block_duration,
                            target_rms=0.02)
                            
    # BASELINE 
    #audio_stim = play_baseline_block(fs = FS, duration_seconds = block_duration, target_rms=0.02)


    # ---- Trial schedule ----
    N_TRIALS = int(block_duration / trial_period)
    n_targets = int(target_prob * N_TRIALS)

    trial_types = np.array([1]*n_targets + [0]*(N_TRIALS - n_targets))
    np.random.shuffle(trial_types)

    trials_data = []
    clock = core.Clock()
    block_start = clock.getTime()

    for k in range(N_TRIALS):

        trial_start_time = block_start + k * trial_period
        while clock.getTime() < trial_start_time:
            core.wait(0.001)

        is_nogo = (trial_types[k] == 1)

        if is_nogo:
            stim = np.random.choice(nogo_stims)
        else:
            stim = standard_stim

        rt_clock = core.Clock()

        # ---- STIM ON (250 ms) ----
        stim.draw()
        win.flip()
        on_time = clock.getTime()   # ✅ add this line

        # 🔴 THIS IS THE FIX — ADD THESE TWO LINES
        event.clearEvents(eventType='keyboard')
        rt_clock.reset()

        keys = event.waitKeys(
            maxWait=stim_on,
            keyList=['space', 'escape'],
            timeStamped=rt_clock
        )


        fixation.draw()
        win.flip()

        resp_key = None
        rt = None
        if keys is not None:
            key, ts = keys[0]
            if key == 'escape':
                audio_stim.stop()
                win.close()
                core.quit()
            resp_key = key
            rt = ts
        is_omission = 0
        is_commission = 0

        if is_nogo:
            correct = (resp_key is None)

            if resp_key is not None:
                is_commission = 1

        else:
            correct = (resp_key == 'space')

            if resp_key is None:
                is_omission = 1

        # -----------------------------------------------------------------

        # store everything
        trials_data.append({
            'block_index': block_index,
            'fb': fb,
            'trial_index': k,
            'is_nogo': int(is_nogo),           # 1 = No-Go, 0 = Go
            'stim_onset_time': on_time - block_start,
            'resp_key': resp_key,
            'rt': rt,                              # None si no respondió
            'correct': int(correct),
            'is_omission': is_omission,
            'is_commission': is_commission
        })

        # wait remaining time to complete 1 second
        now = clock.getTime()
        remaining = trial_start_time + trial_period - now
        if remaining > 0:
            core.wait(remaining)

    # ---- End of block ----
    audio_stim.stop()

    if show_end_screen:
        end_text = visual.TextStim(
            win,
            text=f"Block {block_index+1} complete.\n\nTake a short rest.\n\nPress any key to continue.",
            color='white', height=30
        )
        end_text.draw()
        win.flip()
        wait_for_any_key()

    return trials_data

#---------------------------------------------------------------
#2AFC ALTERNATIVE FORCED CHOICE 
def find_good_its_ratio(start_ratio=0.1, 
                        step=0.05,
                        max_ratio=0.4, 
                        target_perf=0.7):

    its_ratio = start_ratio
    first_block = True
    all_afc_trials = [] 

    while True:
        print(f"\nTesting ITS ratio = {its_ratio:.2f}")

        prop_correct, afc_trials = run_afc_block(
            its_ratio=its_ratio,
            fc=400, fb=10,
            n_trials=5,
            interval_duration=3.0,
            show_instructions=first_block
        )
        all_afc_trials.extend(afc_trials)

        first_block = False  # after first time, never show instructions again

        print(f"Performance: {prop_correct*100:.1f}%")

        if prop_correct >= target_perf:
            print("Reached target performance, keeping this ratio.")
            return its_ratio, all_afc_trials


        its_ratio += step
        if its_ratio > max_ratio:
            print("Max ITS ratio reached; could not hit target performance.")
            return its_ratio, all_afc_trials
            

#----------------------------------------------------------------------------
#TRIAL PARADIGM PIPELINE 
def run_trial_paradigm(final_ratio=0.1, fc=400):
    all_trials = []
    fb_order = np.random.permutation(FB_TRIAL)

    for b_idx, fb in enumerate(fb_order):
        block_data = run_gonogo_block(
            fb=fb,
            its_ratio=final_ratio,
            fc=fc,
            block_duration=90.0,
            stim_on=0.250,
            trial_period=1.0,
            target_prob=0.2,
            block_index=b_idx,
            show_instructions=True,
            show_end_screen=False 
        )
        all_trials.extend(block_data)

    return all_trials

#----------------------------------------------------------------------------
#FULL PARADIGM PIPELINE 
def run_full_paradigm(final_ratio = 0.1, fc=400):
    all_trials = []
    fb_order = np.random.permutation(FB_LIST)
    for b_idx, fb in enumerate(fb_order):
        block_data = run_gonogo_block(
            fb=fb,
            its_ratio=final_ratio,
            fc=fc,
            block_duration=90.0,
            stim_on=0.250,
            trial_period=1.0,
            target_prob=0.2, # for NO GO 
            block_index=b_idx, 
            show_instructions=(b_idx == 0),
            show_end_screen=True
        )
        all_trials.extend(block_data)

    # after 6 blocks, thank the participant
    goodbye = visual.TextStim(
        win,
        text="All blocks complete.\n\nThank you!",
        color='white', height=30
    )
    goodbye.draw()
    win.flip()
    event.waitKeys()

    #win.close()
    #core.quit()

    return all_trials




#----------------------------------------------------------------------------
#GET THE RESULTS 
def save_trials_to_csv(all_trials, filename):
    if not all_trials:
        print("No trials to save.")
        return

    fieldnames = list(all_trials[0].keys())

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_trials)

    print(f"Saved {len(all_trials)} trials to {filename}")
#---------------------------------------------------------------------------
def save_afc_to_csv(trials, filename):
    if not trials:
        return
    fieldnames = trials[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

#----------------------------------------------------------------------------
#MAIN
def main():
    
    # 2AFC = descomentar**************************************
    final_ratio, afc_trials = find_good_its_ratio(
        start_ratio=0.1, 
        step=0.02,
        max_ratio=0.4,
        target_perf=0.7
    )
    save_afc_to_csv(afc_trials, filename = "AFC_REYNA_03-01.csv") 

    # --- FULL PARADIGM ---
    all_trials = run_full_paradigm(final_ratio=0.1, fc=400)
    save_trials_to_csv(all_trials, filename="behavior_REYNA_03-01.csv")

    win.close()
    core.quit()



if __name__ == "__main__": 
    main()


