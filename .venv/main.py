import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import scipy as sp

#Loading audio
audioSample = "res/do.wav"
HOP_LENGTH = 512
#Sample rate of 22050Hz 
music, sr = librosa.load(audioSample)


#1. Applying Fourier on the Audio (return two complexes numbers)
ShortFourier = librosa.stft(music)
#2. Calculating Spectogram
musicSpectrogram = np.abs(ShortFourier) ** 2 #Better visual
musicSpectrogram = librosa.power_to_db(musicSpectrogram) #Convert power in Db
#Result of spectrog. is a array of 2 dimension decibel level of a specific frequency bin for each time frame


#3. Display simple spectogram (axis : Time, Freq, Db)
def displaySpectrog(musicSpectrogram, sr, hop_length, y_axis="log"):
    plt.figure(figsize=(10,5))
    librosa.display.specshow(musicSpectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis
                             )
    plt.colorbar(format="%+2.f dB")
    plt.show()

#displaySpectog(musicSpectrogram, sr, HOPE_SIZE)


#4. Detection of the peaks
#Find peaks in the 2D Array Spectogram
spectrogram = musicSpectrogram
frequencies = librosa.fft_frequencies() #Array with all the frequences
#Variable to stock information

notes_data = []
#Go through the spectrogram to find peaks and put them inside an array
#Each row contains all the Db for one frequency
#Peaks contains the time frame and properties (heights,...)
for i, row in enumerate(spectrogram):
    peaks = sp.signal.find_peaks(row, height=20, threshold=0.28, distance=20, prominence=65)
    peak_times = librosa.frames_to_time(peaks[0]) #All the frames containing a peak
    if peaks[0].size != 0:
        notes_data.append([peak_times, [frequencies[i]] * len(peaks), peaks[1]])

#5. Display peaks on spectrogram
def displaySpectogPeaks(musicSpectrogram, sr, hop_length, notes_data, y_axis="log"):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(musicSpectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis
                             )
    for row in range(len(notes_data)): #go through the rows
        for freq in range(len(notes_data[row][0])): #go through all the diffferent times
            plt.plot(notes_data[row][0][freq], notes_data[row][1][0], 'x')
    plt.colorbar(format="%+2.f dB")
    plt.show()


##6 Midi Conversion
piano_Instrument = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(program=0)

for row in range(len(notes_data)):
        for freq in range(len(notes_data[row][0])):  # go through all the different times
            time = notes_data[row][0][freq]
            frequency = notes_data[row][1][0]
            note_number = pretty_midi.hz_to_note_number(float(frequency))
            note = pretty_midi.Note(velocity=100, pitch=int(note_number), start=(float(time)), end=float(time)+ 1)
            piano.notes.append(note)

piano_Instrument.instruments.append(piano)
piano_Instrument.write('pianoTEST.mid')

displaySpectogPeaks(spectrogram,sr, HOP_LENGTH, notes_data)