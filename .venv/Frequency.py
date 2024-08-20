import math

import librosa
import pretty_midi

y, sr = librosa.load("res/Theme-From-The-Pink-Panther_piano.mp3")

#frequency = librosa.core.fft_frequencies(sr=sr, n_fft=16)

freqs = librosa.pyin(y=y, fmin=65, fmax=2093)

for fr in freqs[0]:
    if math.isnan(fr) == False:
        print(str(fr) + "\t" + librosa.hz_to_note(fr))


# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()

# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)


i = 0;
for fr in freqs[0]:
    if math.isnan(fr) == False:
        note_number = librosa.hz_to_midi(fr)

        # Create a Note instance, starting at 0s and ending at .5s
        note = pretty_midi.Note(velocity=100, pitch=note_number.__int__(), start=i, end=i+1)

        # Add it to our cello instrument
        cello.notes.append(note)
        i+=1


# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(cello)

# Write out the MIDI data
cello_c_chord.write('test.mid')