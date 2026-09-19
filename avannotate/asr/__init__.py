"""Transcription: which audio to recognise, in what language, and its words.

Split so that everything except the recogniser call itself is pure Python over
arrays and JSON -- the routing decision, the language vote, the word trimming
and the flag thresholds are all testable without the package installed.
"""
