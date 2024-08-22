import librosa
import numpy as np
import array
import pretty_midi


""" ****************** TIMING de début *****************"""


# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()

# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)


y, sr = librosa.load("res/Theme-From-The-Pink-Panther_piano.mp3")

onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
for onset in onset_frames:
    #print(str(onset) + "\t" + librosa.frames_to_time(onset).__str__())

    note = pretty_midi.Note(velocity=100, pitch=1, start=librosa.frames_to_time(onset), end=librosa.frames_to_time(onset) + .3)
    cello.notes.append(note)

cello_c_chord.instruments.append(cello)
cello_c_chord.write('4.mid')


S = librosa.stft(y)
pitches, magnitudes = librosa.piptrack(S=S, sr=sr)

for p in pitches:
    print(p.argmax())