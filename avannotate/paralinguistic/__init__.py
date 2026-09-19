"""Paralinguistic tagging: one tag per utterance, from three models' opinions.

Split so the whole decision is pure and testable -- the vocabulary, the label
tables, and the precedence between dimensions are all data and arithmetic.  The
only part that needs a GPU is the three model calls themselves.
"""
