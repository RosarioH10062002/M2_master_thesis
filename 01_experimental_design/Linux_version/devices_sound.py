import psychtoolbox.audio as audio
devices = audio.get_devices()

for d in devices:
    print(d)
    