import librosa

y, sr = librosa.load("res/Theme-From-The-Pink-Panther_piano.mp3")


# Extracting the chroma features and onsets
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
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
print("Note pitch \t Onset frame \t Note duration")
for entry in notes:
  print(entry[0],'\t\t\t',librosa.frames_to_time(entry[1]),'\t\t\t',entry[2])
