# Server setup

The stages before S4 need nothing but ffmpeg, numpy, and OpenCV, and run
anywhere. S4 needs DiariZen, which is not on PyPI and wants a GPU. This is the
part that only runs on the server.

## DiariZen

Confirmed from its README and model cards; **not yet executed here**, so treat
the first run as a smoke test rather than a formality.

```bash
conda create --name diarizen python=3.10
conda activate diarizen

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/BUTSpeechFIT/DiariZen
cd DiariZen
pip install -r requirements.txt && pip install -e .

# DiariZen vendors a modified pyannote-audio and uses dscore as a submodule.
cd pyannote-audio && pip install -e .[dev,testing] -c ../constraints.txt && cd ..
git submodule init && git submodule update
```

Weights come from Hugging Face on first use:

| model | overlapping speakers | licence |
| --- | --- | --- |
| `BUT-FIT/diarizen-wavlm-large-s80-md-v2` | up to 4 | CC BY-NC 4.0 |
| `BUT-FIT/diarizen-wavlm-large-s80-md` | arrival-order slots | CC BY-NC 4.0 |

**Non-commercial only.** The project has confirmed that is acceptable; if that
ever changes, this stage is the one to replace.

The default is v2 because simultaneous speech is what this pipeline exists to
handle, and v1 collapses extra speakers into its arrival-order slots.

## What to verify on the first run

The adapter in `avannotate/audio/diarize.py` is written against DiariZen's
documented interface, which is as far as this machine could take it. Two things
need confirming:

1. **`DiariZenPipeline.from_pretrained(...)` returns an object that accepts
   `.to(torch.device(...))`.** If it does not, `DiariZenDiarizer._place_on`
   records `"backend default"` rather than claiming a device it did not get --
   check the `backend.device` field in `s4-diarize/summary.json` and make sure it
   says what you expect.

2. **`result.itertracks(yield_label=True)` yields `(turn, _, speaker)`** with
   `turn.start` and `turn.end` in seconds. If the shape differs, `diarize()`
   raises a `DiarizerError` naming the type it got rather than failing obscurely
   further down.

A quick check against a known file, taken from DiariZen's own README:

```python
from diarizen.pipelines.inference import DiariZenPipeline
pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
result = pipeline("example/EN2002a_30s.wav")
for turn, _, speaker in result.itertracks(yield_label=True):
    print(f"{turn.start:.1f}-{turn.end:.1f} {speaker}")
```

Their documented output for that file has overlapping turns
(`0.0-2.7 speaker_0` under `0.8-13.6 speaker_3`), which is the property the rest
of the pipeline depends on.

## Running the pipeline

```bash
avannotate run --stage s0-preprocess --input videos.txt --output ./outputs
avannotate run --stage s1-faces     --input videos.txt --output ./outputs \
    --config configs/s1.insightface.json
avannotate run --stage s2-tracks    --input videos.txt --output ./outputs \
    --config configs/s2.default.json
avannotate run --stage s3-cluster   --input videos.txt --output ./outputs \
    --config configs/s3.default.json
avannotate run --stage s4-diarize   --input videos.txt --output ./outputs \
    --config configs/s4.diarizen.json
```

Every stage skips itself when its outputs are present and unchanged, so a rerun
after a crash costs only the video that was in flight. `--force` overrides that.

## Throughput

Measured here, on CPU, over 26 seconds of video across three clips:

| stage | time | note |
| --- | --- | --- |
| S0 | ~2s | ffmpeg |
| S1 (insightface, stride 3) | 63s | ~2.4x realtime; a GPU is far faster |
| S2 | <1s | pure Python |
| S3 | <1s | pure Python |

S1 dominates and is what to profile first on real hardware. `embedding_interval_seconds`
changes only disk, not runtime: insightface computes the vector as part of its
own pipeline whether or not it is written. `allowed_modules=["detection",
"recognition"]` skips the two bundled models this pipeline never uses and cut
that measurement by about 22%.
