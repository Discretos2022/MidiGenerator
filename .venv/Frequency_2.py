import librosa
import numpy as np
import array

y, sr = librosa.load("res/Theme-From-The-Pink-Panther_piano.mp3")

stft = librosa.stft(y=y)

#print(stft.__abs__())

import matplotlib.pyplot as plt

S = np.abs(librosa.stft(y))

fig, ax = plt.subplots()

img = librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max), y_axis='log', x_axis='time', ax=ax)

ax.set_title('Power spectrogram')

fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.show()


while True:
    continue