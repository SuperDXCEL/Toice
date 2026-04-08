import time
from pyannote_onnx import PyannoteONNX

start = time.time()

pyannote = PyannoteONNX()

sample = "./short.wav"

turns = list(pyannote.itertracks(sample))

# Now, you can access the turns before further processing
print("Captured Turns:")
for turn in turns:
    print("speaker : ", turn['speaker'], "start : ", turn['start'], "end : ", turn['stop'])

print("TIME ELAPSED: " + str((time.time() - start)))
