# Toice, a vishing prevention and protection software

Tech Stack:
    Kotlin for Android app development
    Python websockets for sending voice call data to the server
    Fast-Whisper for turning .wav into .txt

Toice flags unknown calls, listens to them in real time, with a latency of (2-3 seconds) and searches for social engineering activity.

Using a heuristic system (heuristic_approach.py), we score different keywords and context, if a certain score is met, we sound the alarm, and notify the current user with the Text To Speech Android API.
