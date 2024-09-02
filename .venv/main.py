import math
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import scipy as sp
#AUDIO LOADING AND SETTINGS
audioSample = "res/Interstellar.wav"
HOP_LENGTH = 512
bins_per_octave = 12
#sr = 22050
music, sr = librosa.load(audioSample)
onset_env = librosa.onset.onset_strength(y=music, sr=sr)

#APPLYING FOURIER AND CREATE SPECTROGRAM
FMIN = float(librosa.note_to_hz("C1"))
shortFourier = librosa.cqt(music, fmin=FMIN, bins_per_octave=bins_per_octave)
musicSpectrogram = librosa.amplitude_to_db(np.abs(shortFourier))
tempo, beat = librosa.beat.beat_track(y=music, sr=sr)
print(f"Tempo : {tempo}")
def displayLinesNotes(musicSpectrogram, sr, hop_length, notes, hline=-1, pr=-1):
    onset_frames = librosa.onset.onset_detect(y=music, onset_envelope=onset_env, sr=sr)
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
                             bins_per_octave=bins_per_octave,
                             fmin=FMIN
                             )
    for note in range(len(notes)):
        plt.plot(notes[note][0], notes[note][1], 'x')
    plt.colorbar(format="%+2.f dB")

def hz_to_cqt(hz):
    frequencies = librosa.cqt_frequencies(84,fmin=FMIN, bins_per_octave=bins_per_octave).tolist()
    return frequencies.index(hz)

def searchMaxValues(graph):
    threshold = -30
    frequencies = librosa.cqt_frequencies(84, fmin=FMIN, bins_per_octave=bins_per_octave)
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
        time = librosa.frames_to_time(table[note][1])
        frequency = table[note][0]
        displayable.append([time,frequency])
    return(displayable)

def SplitFondamentaleAndHarmoniqueWithoutOrdre(values):
    final = []
    if len(values) == 0:
        return final
    final.append(getMaxiFreq(values))
    for v in values:
        #print((v[0] / final[0][0]), " / ", int(v[0] / final[0][0]))
        if ((v[0] / final[0][0]) != int(v[0] / final[0][0])) or (((v[0] / final[0][0]) <= int(v[0] / final[0][0]) + 1) and ((v[0] / final[0][0]) >= int(v[0] / final[0][0]) - 1)):
            if v[0] > final[0][0]:
                if v[0] <= 1500:
                    final.append(v)
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
    onset_frames = librosa.onset.onset_detect(y=music, onset_envelope=onset_env, sr=sr)
    for i,frame in enumerate(onset_frames):
        #change the orientation of the table
        tableTransposed = np.transpose(musicSpectrogram)
        frameTransposed = tableTransposed[frame]
        #look for all the maximal value of each frames
        pic = searchMaxValues(frameTransposed)
        pic2 = GetFiveMax(pic)
        pic3 = SplitFondamentaleAndHarmoniqueWithoutOrdre(pic2)




        #Adding all the found values and convert it into the right format (for duration)
        #tableOfNotes = [frame, frequency]
        for note in range(len(pic3)):
            notes.append([frame, pic3[note][0]])
        pic3 = []
    return(notes)


def GetFiveMax(values):
    result = []
    for i in range(5):
        if i < len(values):
            result.append(getMaxiFreq(values))
            values.remove(getMaxiFreq(values))
    return result

def getMaxiFreq(values):

    result = values[0]

    for v in values:

        if v[1] > result[1]:

            result = v

    return result



def getDuration(notes, spectrogram):
    #frame,frequencies
    frequencies = (librosa.cqt_frequencies(84,fmin=FMIN, bins_per_octave=bins_per_octave)).tolist()
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
                    if(frame-startFrame) < 40:
                        finalnotes.append([frequency, startFrame, frame, frame-startFrame])
                        print(f"freq {frequency}, start {startFrame}, stop {frame}, total {frame-(startFrame)}")
                        break
            else:
                if startingDecibel - abs(startingDecibel*.5) > currentDecibel:
                    if(frame-startFrame) < 40:
                        finalnotes.append([frequency, startFrame, frame, frame - startFrame])
                        print(f"freq {frequency}, start {startFrame}, stop {frame}, total {frame - (startFrame)}")
                        break
    return(finalnotes)

##6 Midi Conversion
def midiConversion(notes):
    piano_Instrument = pretty_midi.PrettyMIDI(initial_tempo=round(tempo[0]))
    piano = pretty_midi.Instrument(program=0)
    for note in range(len(notes)):
        frequency = notes[note][0]
        startFrame = notes[note][1]
        stopFrame = notes[note][2]
        note_number = round(pretty_midi.hz_to_note_number(float(frequency)))
        startTime = float(librosa.frames_to_time(startFrame))
        stopTime = float(librosa.frames_to_time(stopFrame))

        note = pretty_midi.Note(velocity=45, pitch=note_number, start=startTime, end=stopTime)
        piano.notes.append(note)

    piano_Instrument.instruments.append(piano)
    piano_Instrument.write('pianoTEST.mid')

#MAIN
tableOfNotes = searchNote()
music = getDuration(tableOfNotes,musicSpectrogram)
displayLinesNotes(musicSpectrogram,sr,HOP_LENGTH, dataToDisplayNotes(music))
midiConversion(music)
plt.show()