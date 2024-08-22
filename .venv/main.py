import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import scipy as sp

#Loading audio
audioSample = "res/Do_accord.wav"
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
    peaks = sp.signal.find_peaks(row, height=20, threshold=0, distance=1)
    peak_times = librosa.frames_to_time(peaks[0]) #All the frames containing a peak
    if peaks[0].size != 0:
        notes_data.append([peak_times, [frequencies[i]] * len(peaks), peaks[1]])


#Array of Tuple(Frequency, Time, PeakHeights)
dataSets = []
for row in range(len(notes_data)):
    frequency =  notes_data[row][1][0]
    for freq in range(len(notes_data[row][0])) :
        timeArray =  notes_data[row][0][freq]
        peaksProperties = notes_data[row][2]['peak_heights'][freq]
        dataSets.append((float(frequency),float(timeArray), float(peaksProperties)))
def displayDataSet():
    for row in range(len(dataSets)):
        print(f"Row : {row}")
        print(f"Frequency : {dataSets[row][0]}, Time : {dataSets[row][1]}, PeakHeight : {dataSets[row][2]}")
        print("--------")

displayDataSet()

#5. Display peaks on spectrogram
def displaySpectogPeaks(musicSpectrogram, sr, hop_length, dataSets, y_axis="log"):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(musicSpectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis
                             )
    for row in range(len(dataSets)): #go through the rows
         plt.plot(dataSets[row][1], dataSets[row][0], 'x')
    plt.colorbar(format="%+2.f dB")
    plt.show()

#Filtering

#Goal 0 : Create an array of Tuple (with freq and time)

#Goal 1 remove all the closest cross which are at the same frequency (or close)
 #Group all the peaks by time
#Goal 2 remove the harmonique bins
 # to proceed : Harmonics ares multiples of Fundamental (Control each frequency from each peaks and if the frequency is a multiple
 #take the lower

#threshold = 10
#filtered_data = []
#for row in range(len(notes_data) - 1):

#   #Previous Frequency and Peak
#    prevPeak = notes_data[row][2]['peak_heights'][0]
#    print(prevPeak)
#    print(f"Time : {notes_data[row][0]}")
#    prevFreq = notes_data[row][1][0]
#    print(prevFreq)
#    #Next Frequency and Peak
#   nextPeak = notes_data[row + 1][2]['peak_heights'][0]
#    print(nextPeak)
#    print(f"Time : {notes_data[row+1][0]}")
#    nextFreq = notes_data[row + 1][1][0]
#    print(nextFreq)
#    print("-----")
#
#    if np.abs(prevFreq - nextFreq) >= threshold:
#        filtered_data.append(notes_data[row])


##6 Midi Conversion
piano_Instrument = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(program=0)

for row in range(len(dataSets)):
            time = dataSets[row][1]
            frequency = dataSets[row][0]
            note_number = pretty_midi.hz_to_note_number(float(frequency))
            note = pretty_midi.Note(velocity=100, pitch=int(note_number), start=(float(time)), end=float(time)+ 1)
            piano.notes.append(note)

piano_Instrument.instruments.append(piano)
piano_Instrument.write('pianoTEST.mid')

displaySpectogPeaks(spectrogram,sr, HOP_LENGTH, dataSets)