"""Offline annotation pipeline for multi-person video.

The package is layered so the deterministic core can be tested without a GPU:

* :mod:`avannotate.schema` -- the deliverable's data contract
* :mod:`avannotate.annotation` -- render and parse the two-level script
* :mod:`avannotate.associate` -- face/speaker assignment (stage S8)
* :mod:`avannotate.segment` -- silence-free utterance segmentation (stage S10)
* :mod:`avannotate.qa` -- hard gates and reported metrics
* :mod:`avannotate.stages` -- the model-backed stages, which need a GPU
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
