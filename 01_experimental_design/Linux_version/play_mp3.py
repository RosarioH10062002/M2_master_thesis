from scipy.io import wavfile
import sounddevice as sd 


wav_file = "/home/telecom/Bureau/Rosario/M2_master_thesis-main/ding_sound.wav" # wav file needs the whole path 
samplerate, data = wavfile.read(wav_file)

sd.play(data, samplerate)
sd.wait()