"""
import librosa

sound = librosa.load("res/Music_Audio.wav");
print("Sound loaded !")

frames = librosa.beat.beat_track(sound)

print(frames)
"""

# Beat tracking example

import librosa
import matplotlib.pyplot as plt
import numpy

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load("res/Theme-From-The-Pink-Panther_piano.mp3")

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {' + tempo.__str__() + '} beats per minute : ')

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print("beat_frames" + beat_frames.__str__())
print("")
print("beat_times" + beat_times.__str__())

'''
plt.figure("fig 1")

librosa.display.waveshow(y=y, sr=sr);
plt.show()
'''

stft = librosa.stft(y)
spectogram = numpy.abs(stft)
db = librosa.amplitude_to_db(spectogram)

db2 = librosa.reassigned_spectrogram(y=y, sr=sr)

print("test :" + db[10, 50].__str__())

plt.figure("fig 1")
librosa.display.specshow(data=db, sr=sr, y_axis='log', x_axis='time', cmap='inferno');
plt.xlabel("time")
plt.ylabel("Hz")
plt.show()


while True:
    continue