#!/usr/bin/env python

"""Echo server using the threading API."""

from websockets.sync.server import serve
import ssl
import numpy as np

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

def handle_array(websocket):
    for message in websocket:
        print("Byte String ", message)
        np_array = np.frombuffer(message, dtype=np.int16)
        print(np_array)

def main():
    with serve(handle_array, "0.0.0.0", 8765, ssl=ssl_context) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
