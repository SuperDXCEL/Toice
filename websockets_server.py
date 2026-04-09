#!/usr/bin/env python

"""Echo server using the threading API."""

from websockets.sync.server import serve
import ssl
import time
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from heuristic_approach import heuristic_system

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
score = 0
full_text = ""
starting_timestamp_of_convo = time.time()

def handle_array(websocket):
    for message in websocket:
        # print("Byte String ", message)
        np_array = np.frombuffer(message, dtype=np.int16)
        # print(np_array)
        array_to_wav(np_array, websocket)

def array_to_wav(np_array, websocket):
    write("snippet.wav", 16000, np_array)
    global model
    segments, info = model.transcribe("snippet.wav")

    full_line = ""
    for segment in segments:
        full_line += segment.text
    print(full_line)

    global full_text 
    full_text += full_line

    global score, starting_timestamp_of_convo
    score = heuristic_system(full_text, full_line, score, starting_timestamp_of_convo, websocket)

def main():
    with serve(handle_array, "0.0.0.0", 8765, ssl=ssl_context) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
