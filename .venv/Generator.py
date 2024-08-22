import librosa
import pretty_midi

def getNoteNameFromNumber(num: int):
    match (num):
        case 0:
            return "DO  "
        case 1:
            return "DO# "
        case 2:
            return "RE  "
        case 3:
            return "RE# "
        case 4:
            return "MI  "
        case 5:
            return "FA  "
        case 6:
            return "FA# "
        case 7:
            return "SOL "
        case 8:
            return "SOL#"
        case 9:
            return "LA  "
        case 10:
            return "LA# "
        case 11:
            return "SI  "
        case 12:
            return "DO  "

def getNoteFromNumber(num: int):
    match (num):
        case 0:
            return "C  "
        case 1:
            return "C# "
        case 2:
            return "D  "
        case 3:
            return "D# "
        case 4:
            return "E  "
        case 5:
            return "F  "
        case 6:
            return "F# "
        case 7:
            return "G  "
        case 8:
            return "G# "
        case 9:
            return "A  "
        case 10:
            return "A# "
        case 11:
            return "B  "
        case 12:
            return "C  "

    return "A  "

def getMidi(note:str):
    return librosa.note_to_midi(note.strip())

# Create a PrettyMIDI object
cello_c_chord = pretty_midi.PrettyMIDI()

# Create an Instrument instance for a cello instrument
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)

# Iterate over note names, which will be converted to note number later


y, sr = librosa.load("res/Theme-From-The-Pink-Panther_piano.mp3")

# Extracting the chroma features and onsets
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#chroma = librosa.feature.chroma_stft(y=y)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

first = True
notes = []
for onset in onset_frames:
  chroma_at_onset = chroma[:, onset]
  note_pitch = chroma_at_onset.argmax()
  # For all other notes
  if not first:
      note_duration = librosa.frames_to_time(onset, sr=sr)
      notes.append((note_pitch,onset, note_duration - prev_note_duration))
      prev_note_duration = note_duration
  # For the first note
  else:
      prev_note_duration = librosa.frames_to_time(onset, sr=sr)
      first = False
#print("Note pitch \t Note Name \t Note \t Onset frame \t Note duration \t\t\t MIDI")


for entry in notes:

    # Retrieve the MIDI note number for this note name
    note_number = 13-entry[0]#pretty_midi.note_name_to_number(pretty_midi.key_number_to_key_name(int(entry[0])))
    #print(note_number)
    note_number1 = getMidi((getNoteFromNumber(note_number).strip() + '4').__str__())

    # Create a Note instance, starting at 0s and ending at .5s
    note = pretty_midi.Note(velocity=100, pitch=note_number1, start=librosa.frames_to_time(entry[1]), end= librosa.frames_to_time(entry[1]) + entry[2].__float__())
    
    # Add it to our cello instrument
    cello.notes.append(note)



"""
note_number1 = getMidi("C5")
# Create a Note instance, starting at 0s and ending at .5s
note = pretty_midi.Note(velocity=100, pitch=note_number1, start=0, end=5)
cello.notes.append(note)

note_number1 = getMidi("D5")
# Create a Note instance, starting at 0s and ending at .5s
note = pretty_midi.Note(velocity=100, pitch=note_number1, start=5, end=10)
cello.notes.append(note)
"""



# Add the cello instrument to the PrettyMIDI object
cello_c_chord.instruments.append(cello)

# Write out the MIDI data
cello_c_chord.write('cello-C-chord.mid')