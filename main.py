import os
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

# STEP 1: LOAD MIDI FILES
notes = []

for file in glob.glob("data/midi/*.mid"):
    midi = converter.parse(file)

    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print("Total notes:", len(notes))

# STEP 2: CREATE SEQUENCES
sequence_length = 100

pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(n_vocab)

network_output = to_categorical(network_output)

# STEP 3: BUILD MODEL
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(n_vocab))
model.add(Dense(n_vocab, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# STEP 4: TRAIN MODEL
model.fit(network_input, network_output, epochs=20, batch_size=64)

# STEP 5: GENERATE MUSIC
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]

int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

prediction_output = []

for i in range(200):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(prediction_input, verbose=0)

    index = np.argmax(prediction)
    result = int_to_note[index]

    prediction_output.append(result)

    pattern = np.append(pattern, index)
    pattern = pattern[1:]

# STEP 6: SAVE MIDI FILE
offset = 0
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_list = []
        for n in notes_in_chord:
            new_note = note.Note(int(n))
            new_note.offset = offset
            notes_list.append(new_note)
        new_chord = chord.Chord(notes_list)
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output/generated.mid')

print("✅ Music Generated! Check output/generated.mid")