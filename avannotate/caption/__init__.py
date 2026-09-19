"""Captioning: what a shot looks like, and the names it is allowed to use.

Two levels, one sentence for the video and one for each shot, both purely
visual.  The stage reads no audio, so it is independent of the whole
speech chain; the only thing it shares with it is the face identifiers, which
are handed to the model as a constraint and checked afterwards.
"""
