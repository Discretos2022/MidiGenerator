import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import scipy as sp


audioSample = "res/james_bond.mp3"
HOP_LENGTH = 512
bins_per_octave = 12
n_bins = 84
n_fft = 2048
sr = 22050
music, sr = librosa.load(audioSample, sr=sr)
onset_env = librosa.onset.onset_strength(y=music, sr=sr)

FMIN = float(librosa.note_to_hz("C1"))
shortFourier = librosa.cqt(music, sr=sr, hop_length=HOP_LENGTH, fmin=FMIN, bins_per_octave=bins_per_octave, n_bins=n_bins)
musicSpectrogram = librosa.amplitude_to_db(np.abs(shortFourier))
tempo, beat = librosa.beat.beat_track(y=music, sr=sr, onset_envelope=onset_env)
print(f"Tempo : {tempo}")
def displayLinesNotes(musicSpectrogram, sr, hop_length, notes, hline=-1, pr=-1):
    onset_frames = librosa.onset.onset_detect(y=music, onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    plt.figure(figsize=(10, 5))
    plt.vlines(librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft), 0, 15000, color="b")
    if pr != -1:
        plt.vlines(librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft) + librosa.frames_to_time(pr, sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft), 0, 15000, color="g")
        plt.vlines(librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft) - librosa.frames_to_time(pr, sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft), 0, 15000, color="g")
    if hline != -1:
        plt.hlines(hline, 0, 100, color="g")
    librosa.display.specshow(musicSpectrogram,
                             sr=sr,
                             hop_length=HOP_LENGTH,
                             n_fft=n_fft,
                             x_axis="time",
                             y_axis="cqt_hz",
                             bins_per_octave=bins_per_octave,
                             fmin=FMIN
                             )
    for note in range(len(notes)):
        plt.plot(notes[note][0], notes[note][1], 'x')
    plt.colorbar(format="%+2.f dB")

def hz_to_cqt(hz):
    frequencies = librosa.cqt_frequencies(n_bins,fmin=FMIN, bins_per_octave=bins_per_octave).tolist()
    return frequencies.index(hz)

def searchMaxValues(graph):
    threshold = 1
    frequencies = librosa.cqt_frequencies(n_bins, fmin=FMIN, bins_per_octave=bins_per_octave)
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

def dataToDisplayNotes(table):
    displayable = []
    for note in range(len(table)):
        time = librosa.frames_to_time(table[note][1], sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft)
        frequency = table[note][0]
        displayable.append([time,frequency])
    return(displayable)

def SplitFondamentaleAndHarmoniqueWithoutOrdre(values):
    #Going through each frames
    #There is only maximum 3 notes
    print(f"Values : {values}")
    final = []
    if len(values) == 0:
        return final
    #Considering that the fundamental is the loudest Note
    fundamental = getMaxiDb(values)
    frequencyFund = fundamental[0]
    dbFund = fundamental[1]
    final.append(fundamental)
    values.remove(fundamental)
    print(f"Fundamental {fundamental}")
    print(f"Values without maximum : {values}")
    if len(values) == 0:
        return final
    for note in range(0,len(values)):
        freq = values[note][0]
        db = values[note][1]
        #should test on 0.5, 2
        if freq / frequencyFund == 0.5 or round(freq,-1)/round(frequencyFund, -1) == 0.5 or int(freq) / int(frequencyFund) == 0.5:
            print(f"Results {round(freq / frequencyFund)}, {round(round(freq,-1)/round(frequencyFund,-1))}, {round(int(freq) / int(frequencyFund))}")
            index = final.index(fundamental)
            if(freq < frequencyFund):
                final[index] = [freq,db]
            else:
                final[index] = [frequencyFund,dbFund]

            print(f"Is an Harmonic {values[note]}")
        else:
            final.append(fundamental)
        print("-------------")

    return final

def isFondamentale(value, hz):
    for v in value:
        #print((v / hz), " / ", round(v / hz))
        if (v / hz) == round(v / hz):
            return False
    return True


#Find notes
def searchNote():
    notes = []
    #go through all the frames
    onset_frames = librosa.onset.onset_detect(y=music, onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    for i,frame in enumerate(onset_frames):
        #change the orientation of the table
        tableTransposed = np.transpose(musicSpectrogram)
        frameTransposed = tableTransposed[frame]
        #look for all the maximal value of each frames, could be negative
        pic = searchMaxValues(frameTransposed)
        pic2 = GetFiveMax(pic)
        #pic3 = SplitFondamentaleAndHarmoniqueWithoutOrdre(pic2)




        #Adding all the found values and convert it into the right format (for duration)
        #tableOfNotes = [frame, frequency]
        for note in range(len(pic2)):
            notes.append([frame, pic2[note][0]])
        pic2 = []
    return(notes)


def GetFiveMax(values):
    result = []
    valuesNeeded = 6
    valuesObtained = 0
    lenght = len(values)
    if lenght != 0:
        if lenght < valuesNeeded:
            valuesNeeded = lenght
        while valuesNeeded != valuesObtained:
            if len(values) != 0:
                result.append(getMaxiDb(values))
                values.remove(getMaxiDb(values))
            valuesObtained += 1
    return result

def getMaxiDb(values):
        maximDb = values[0][1]
        index = 0
        for note in range(1, len(values)):
            currentDb = values[note][1]
            if currentDb > maximDb:
                maximDb = currentDb
                index = note
        return([values[index][0],values[index][1]])



def getDuration(notes, spectrogram):
    #frame,frequencies
    frequencies = (librosa.cqt_frequencies(n_bins,fmin=FMIN, bins_per_octave=bins_per_octave)).tolist()
    finalnotes = []
    for note in range(len(notes)):
        frequency = notes[note][1]
        row = frequencies.index(frequency)
        startFrame = notes[note][0]
        startingDecibel = spectrogram[row][startFrame]
        #Going through all the frames of the spectogram
        for frame in range(startFrame + 1, len(spectrogram[0])):
            currentDecibel = spectrogram[row][frame]
            if startingDecibel > 0:
                if (startingDecibel - (startingDecibel*.5)) > currentDecibel:
                    if(frame-startFrame) > 8 and (frame-startFrame) < 600:
                        finalnotes.append([frequency, startFrame, frame, frame-startFrame])
                        print(f"freq {frequency}, start {startFrame}, stop {frame}, total {frame-(startFrame)}")
                        break
            else:
                if startingDecibel - abs(startingDecibel*.5) > currentDecibel:
                    if(frame-startFrame) > 8 and (frame-startFrame) < 600:
                        finalnotes.append([frequency, startFrame, frame, frame - startFrame])
                        print(f"freq {frequency}, start {startFrame}, stop {frame}, total {frame - (startFrame)}")
                        break
    return(finalnotes)

##6 Midi Conversion
def midiConversion(notes, instrument):
    instru = 0
    if instrument == "piano":
        instru = pretty_midi.Instrument(program=0)
        print("Song Converted Piano")
    else:
        instru = pretty_midi.Instrument(program=25)
        print("Song Converted Guitare")
    file = pretty_midi.PrettyMIDI(initial_tempo=round(tempo[0]))
    for note in range(len(notes)):
        frequency = notes[note][0]
        startFrame = notes[note][1]
        stopFrame = notes[note][2]
        note_number = round(pretty_midi.hz_to_note_number(float(frequency)))
        startTime = float(librosa.frames_to_time(startFrame, sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft))
        stopTime = float(librosa.frames_to_time(stopFrame, sr=sr, hop_length=HOP_LENGTH, n_fft=n_fft))

        note = pretty_midi.Note(velocity=45, pitch=note_number, start=startTime, end=stopTime)
        instru.notes.append(note)
    file.instruments.append(instru)
    file.write('FINAL_MUSIC.mid')


#MAIN
tableOfNotes = searchNote()
music = getDuration(tableOfNotes,musicSpectrogram)
displayLinesNotes(musicSpectrogram,sr,HOP_LENGTH, dataToDisplayNotes(music))
midiConversion(music, "piano")
plt.show()