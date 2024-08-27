import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import scipy as sp
#AUDIO LOADING AND SETTINGS
audioSample = "res/piano.wav"
HOP_LENGTH = 512
#sr = 22050
music, sr = librosa.load(audioSample)
#APPLYING FOURIER AND CREATE SPECTROGRAM
FMIN = float(librosa.note_to_hz("C2"))
shortFourier = librosa.cqt(music, fmin=FMIN)
musicSpectrogram = librosa.amplitude_to_db(np.abs(shortFourier))


#DISPLAY LINES

def displayLinesNotes(musicSpectrogram, sr, hop_length, notes, hline=-1, pr=-1):
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
                             y_axis="cqt_hz",
                             bins_per_octave=12,
                             fmin=FMIN
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

def hz_to_fft(hz):
    frequencies = librosa.cqt_frequencies(84,fmin=FMIN, bins_per_octave=12).tolist()
    return frequencies.index(hz)

#4. DETECTION OF THE PEAKS
def detectionOfPeaks(spectrogram):
    frequencies = librosa.cqt_frequencies(84,fmin=FMIN, bins_per_octave=12)
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

def displaySpectogInTimeT(musicSpectrogram, framePeaks, num = -1):
    #All important peaks
    onset_frames = librosa.onset.onset_detect(y=music, sr=sr)
    SpectTransposed = np.transpose(musicSpectrogram)
    plt.plot(SpectTransposed[onset_frames[framePeaks]])
    if num != -1:
        plt.vlines(num, -50, 50, color="r")

def displaySpectogInTimeTWithFreq(musicSpectrogram,framePeaks, nums, reduceMaximums):
    onset_frames = librosa.onset.onset_detect(y=music, sr=sr)
    T = np.transpose(musicSpectrogram)
    for n in nums:
        if n[0] != 0:
            plt.vlines(hz_to_fft(n[0]), -50, 60, color="r") # 41
    plt.plot(T[onset_frames[framePeaks]])

    for i in range(len(reduceMaximums)):
        plt.vlines(hz_to_fft(reduceMaximums[i][0]), -50, 60, color="g") # 41
    plt.plot(T[onset_frames[framePeaks]])


def searchMaxValues(graph):
    threshold = -25
    frequencies = librosa.cqt_frequencies(84, fmin=FMIN, bins_per_octave=12)
    #going through all the value in one frame
    values = []
    for value in range(0,len(graph)-1):
        if value != 0 or value != len(graph)-2:
            #Check the maximum (between two lower values)
            if graph[value-1] < graph[value] and graph[value+1] < graph[value]:
                #decibels threshold
                if graph[value] > threshold:
                    values.append([frequencies[value],graph[value]])
    return values

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

def dataToDisplayNotes(tablesOfNotes):
    displayable = []
    for note in range(len(tableOfNotes)):
        time = librosa.frames_to_time(tablesOfNotes[note][0] + 1)
        frequency = tableOfNotes[note][1]
        displayable.append([time,frequency])
    return(displayable)

def SplitFondamentaleAndHarmoniqueWithoutOrdre(values):
    final = []
    if len(values) == 0:
        return final
    final.append(float(np.array(values).argmax()))
    for v in range(0, len(values) - 2):
        #if math.fabs(values[v] - values[v - 1]) > 50:
        if values[v] > 0:
            if values[v] > 12:
                #if isFondamentale(values, values[v]):
                    final.append(values[v])
    return final


def isFondamentale(value, hz):
    for v in value:
        print((v / hz), " / ", round(v / hz))

        if (v / hz) == round(v / hz):
            return False
    return True


def findMaximum(pics):
    maximum = pics[0][1]
    index = 0
    for note in range(1, len(pics)):
        if maximum < pics[note][1]:
            maximum = pics[note][1]
            index = note
    return index

def filterMaximum(pic):
    peaks = pic
    notes = []
    #remove the first maximum
    maximumIndex = findMaximum(peaks)
    notes.append(peaks[maximumIndex])
    peaks.pop(maximumIndex)
    return(notes)


#Find notes
def searchNote():
    notes = []
    #go through all the frames
    onset_frames = librosa.onset.onset_detect(y=music, sr=sr)
    for i,frame in enumerate(onset_frames):
        #change the orientation of the table
        tableTransposed = np.transpose(musicSpectrogram)
        frameTransposed = tableTransposed[frame]
        #look for all the maximal value of each frames
        pic = searchMaxValues(frameTransposed)
        filtered = filterMaximum(pic)
        #displaySpectogInTimeTWithFreq(musicSpectrogram,i,pic,reduceMaximums)


        #Adding all the found values and convert it into the right format (for duration)
        #tableOfNotes = [frame, frequency]
        for note in range(len(filtered)):
            notes.append([frame, filtered[note][0]])
        filtered = []
    return(notes)

def getDuration(notes, spectrogram):
    frequencies = (librosa.cqt_frequencies(84,fmin=FMIN, bins_per_octave=12)).tolist()
    finalnotes = []
    for note in range(len(notes)):
        frequency = notes[note][1]
        row = frequencies.index(frequency)

        startFrame = notes[note][0] + 1
        minimumDecibel = 6

        #Going through all the frames of the spectogram
        for frame in range(startFrame + 1, len(spectrogram[0])):
            currentDecibel = spectrogram[row][frame]
            if minimumDecibel >= currentDecibel:
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

#MAIN
tableOfNotes = searchNote()
displayLinesNotes(musicSpectrogram,sr,HOP_LENGTH, dataToDisplayNotes(tableOfNotes))
music = getDuration(tableOfNotes,musicSpectrogram)
midiConversion(music)
plt.show()