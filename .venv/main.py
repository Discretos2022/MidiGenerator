import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import scipy as sp
#AUDIO LOADING AND SETTINGS
audioSample = "res/piano.wav"
HOP_LENGTH = 512
n_fft = 2048
#sr = 22050
music, sr = librosa.load(audioSample)
#APPLYING FOURIER AND CREATE SPECTROGRAM
shortFourier = librosa.stft(music)
musicSpectrogram = np.abs(shortFourier) ** 2
musicSpectrogram = librosa.power_to_db(musicSpectrogram)
#Result = 2D Array : decibel and frequency bin /// frequency bins = freq * n_fft / sr


#DISPLAY BASIC INFO SPECTROGRAM
def displayBasic(spectrogram, sr, hop_length, y_axis="log"):
    plt.figure(figsize=(10,5))
    librosa.display.specshow(spectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis
                             )
    plt.colorbar(format="%+2.f dB")

#DISPLAY PEAKS ON SPECTROGRAM
def displaySpectogPeaks(musicSpectrogram, sr, hop_length, dataSets, y_axis="log"):
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(musicSpectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis
                             )
    for row in range(len(dataSets)):
         plt.plot(dataSets[row][1], dataSets[row][0], 'x')
    plt.colorbar(format="%+2.f dB")

#DISPLAY LINES
def displaySpectogLines(musicSpectrogram, sr, hop_length, y_axis="log", hline = -1, pr = -1):
    onset_frames = librosa.onset.onset_detect(y=music, sr=sr)
    plt.figure(figsize=(10, 5))
    plt.vlines(librosa.frames_to_time(onset_frames), 0, 8192, color="b")
    if pr != -1:
        plt.vlines(librosa.frames_to_time(onset_frames) + librosa.frames_to_time(pr), 0, 8192, color="g")
        plt.vlines(librosa.frames_to_time(onset_frames) - librosa.frames_to_time(pr), 0, 8192, color="g")
    if hline != -1:
        plt.hlines(hline, 0, 12, color="g")
    librosa.display.specshow(musicSpectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis
                             )
    plt.colorbar(format="%+2.f dB")

def displayLinesNotes(musicSpectrogram, sr, hop_length, notes, y_axis="log", hline=-1, pr=-1):
    onset_frames = librosa.onset.onset_detect(y=music, sr=sr)
    plt.figure(figsize=(10, 5))
    plt.vlines(librosa.frames_to_time(onset_frames), 0, 8192, color="b")
    if pr != -1:
        plt.vlines(librosa.frames_to_time(onset_frames) + librosa.frames_to_time(pr), 0, 8192, color="g")
        plt.vlines(librosa.frames_to_time(onset_frames) - librosa.frames_to_time(pr), 0, 8192, color="g")
    if hline != -1:
        plt.hlines(hline, 0, 12, color="g")
    librosa.display.specshow(musicSpectrogram,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis
                             )
    for note in range(len(notes)):
        plt.plot(notes[note][0], notes[note][1], 'x')
    plt.colorbar(format="%+2.f dB")


#DISPLAY BASIC DATA
def displayDataSet(dataSets):
    for row in range(len(dataSets)):
        print(f"Row : {row}")
        print(f"Frequency : {dataSets[row][0]}, Time : {dataSets[row][1]}, PeakHeight : {dataSets[row][2]}")
        print("--------")



#4. DETECTION OF THE PEAKS
def detectionOfPeaks(spectrogram):
    frequencies = librosa.fft_frequencies()
    peaksData = [] #Peaks contains the time frame and properties (heights,...)
    for i, row in enumerate(spectrogram):
        peaks = sp.signal.find_peaks(row, height=5, threshold=0.3, distance=1, prominence=1)
        peak_times = librosa.frames_to_time(peaks[0]) #All the frames containing a peak
        if peaks[0].size != 0:
            peaksData.append([peak_times, [frequencies[i]] * len(peaks), peaks[1]])
    return(peaksData)
def creationOfPeaksDataSet(notes_data):
    dataSets = []
    for row in range(len(notes_data)):
        frequency = notes_data[row][1][0]
        for freq in range(len(notes_data[row][0])) :
            timeArray = notes_data[row][0][freq]
            peaksProperties = notes_data[row][2]['peak_heights'][freq]
            dataSets.append((float(frequency),float(timeArray),float(peaksProperties)))
    return(dataSets)

#DETECTING LINES ON THE SPECTROGRAM
def detectionOfLines(dataSets):
    filteringOne = []
    onset_frames = librosa.onset.onset_detect(y=music, sr=sr)
    for row in range(len(dataSets)):
        take = False
        for num in range(0, 20):
            if onset_frames.__contains__(librosa.time_to_frames(dataSets[row][1]) + num):
                #dataSets[row][1] = librosa.time_to_frames(dataSets[row][1]) + num
                filteringOne.append((float(dataSets[row][0]), float(librosa.frames_to_time(librosa.time_to_frames(dataSets[row][1]) + num)), float(dataSets[row][2])))
                take = True
            elif onset_frames.__contains__(librosa.time_to_frames(dataSets[row][1]) - num):
                #dataSets[row][1] = librosa.time_to_frames(dataSets[row][1]) - num
                filteringOne.append((float(dataSets[row][0]), float(librosa.frames_to_time(librosa.time_to_frames(dataSets[row][1]) - num)), float(dataSets[row][2])))
                take = True
                #print(onset_frames.__str__() + " / " + (librosa.time_to_frames(dataSets[row][1]) - num).__str__() + " \t : " + take.__str__())
    return(filteringOne)

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

tableOfNotes = [[0,261],[22,293],[44,329],[66,349],[88,392],[110,440],[132,493],[154,523]]
#tablesOfNotes = [frames,frequency]

def dataToDisplayNotes(tablesOfNotes):
    displayable = []
    for note in range(len(tableOfNotes)):
        time = librosa.frames_to_time(tablesOfNotes[note][0] + 1)
        frequency = tableOfNotes[note][1]
        displayable.append([time,frequency])
    return(displayable)

def getDuration(notes, spectrogram):
    finalnotes = []
    for note in range(len(notes)):
        frequency = notes[note][1]
        row = int(round(frequency * n_fft / sr))
        startFrame = notes[note][0] + 1
        minimumDecibel = 6

        #Going through all the frames of the spectogram
        for frame in range(startFrame + 1, len(spectrogram[0])):
            currentDecibel = spectrogram[row][frame]
            if  minimumDecibel  >= currentDecibel:
                finalnotes.append([frequency, startFrame, frame, frame-startFrame])
                #print(f"freq {frequency}, start {startFrame}, stop {frame}, total {frame-startFrame}")
                break
    return(finalnotes)

##6 Midi Conversion
def midiConversion(notes):
    piano_Instrument = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for note in range(len(notes)):
        frequency = notes[note][0]
        startFrame = notes[note][1]
        stopFrame = notes[note][2]
        note_number = round(pretty_midi.hz_to_note_number(float(frequency)))
        startTime = float(librosa.frames_to_time(startFrame))
        stopTime = float(librosa.frames_to_time(stopFrame))

        note = pretty_midi.Note(velocity=100, pitch=note_number, start=startTime, end=stopTime)
        piano.notes.append(note)

    piano_Instrument.instruments.append(piano)
    piano_Instrument.write('pianoTEST.mid')

displayLinesNotes(musicSpectrogram,sr,HOP_LENGTH, dataToDisplayNotes(tableOfNotes))
music = getDuration(tableOfNotes,musicSpectrogram)
midiConversion(music)
plt.show()