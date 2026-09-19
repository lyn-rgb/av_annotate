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

## LoCoNet

Not a package either. Clone the repository and take the AVA weights from the
Google Drive link in its README:

```bash
git clone https://github.com/SJTUwxz/LoCoNet_ASD
# download loconet_AVA.model from the link in that README
```

The environment follows the repository's own `requirements.yml`; the model is
PyTorch-only and does not need the CUDA build DiariZen pins.

### What to verify on the first run

Two things in `avannotate/asd/` were derived rather than read off the
repository, and both are domain shifts applied to every frame if wrong:

1. **The crop margin.** `TalkNet`'s `cropScale` is 0.40 and LoCoNet inherits it,
   but the exact arithmetic -- whether the margin is a fraction of the width on
   each side, and whether the result is squared before resizing -- is not
   something this machine could check. It is isolated in
   `avannotate/asd/crop.py::crop_box`.

2. **The audio frontend.** The loader wants `[4T, 128]`: four feature frames per
   video frame, 128 bins. A log-mel at 16 kHz with a 10 ms hop gives exactly
   that at 25 fps, which is where `log_mel`'s parameters come from -- derived
   from the shape, not read from the code. It is isolated in
   `avannotate/asd/model.py`.

Both produce ordinary arrays, so a corrected version can be compared against a
saved sample without re-running anything else. The cheapest check is to run S5
on a clip where one person speaks and another is visibly silent, and look at
whether the two traces separate.

## ClearerVoice

```bash
pip install clearvoice
```

No clone and no manual weight download: importing it fetches
`AV_MossFormer2_TSE_16K` from Hugging Face on first use, so the first run needs
network access. Point `HF_HOME` at a shared cache if several jobs run at once.

### What to verify on the first run

The adapter in `avannotate/tse/model.py` is written against the call shown in
ClearerVoice's README:

```python
model = ClearVoice(task="target_speaker_extraction",
                   model_names=["AV_MossFormer2_TSE_16K"])
model(input_path="clip.mp4", online_write=True, output_path="out_dir")
```

Two details of that call could not be checked here, and they are why the adapter
keeps its call in a layer of its own:

1. **The produced file's name.** `ClearerVoiceExtractor.extract` lists the output
   directory before and after the call and takes what appeared, rather than
   predicting a name from the input. If the package derives that name from the
   input path, predicting it would have meant writing to a path this code had
   guessed.
2. **Whether `output_path` is a directory or a file prefix.** Both fit the
   signature; only one puts a file in the directory that was just scanned. If
   `extract` raises saying nothing was written, this is why.

There is also a **reported audio-offset bug** (their issue #160): when the
speaker appears later than the video's start, the output is said to begin at
frame 0. Every crop this pipeline writes starts with the target already on
screen, so it should not apply — but it would present as a constant shift, which
is why each segment's own `start` and `duration` are recorded next to its file.

A cheap end-to-end check before trusting a batch: run S7 with `keep_crops` set in
the config, open one `crops/F001_0000.mp4`, and confirm it holds exactly one face
and that the face is the right person's. That is the entire workaround, so it is
the one thing worth looking at with your own eyes.

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
avannotate run --stage s5-asd       --input videos.txt --output ./outputs \
    --config configs/s5.loconet.json
avannotate run --stage s6-associate --input videos.txt --output ./outputs \
    --config configs/s6.associate.json
avannotate run --stage s7-tse       --input videos.txt --output ./outputs \
    --config configs/s7.clearvoice.json
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
| S4 (DiariZen) | not measured here | GPU; the WavLM front end is the cost |
| S5 (LoCoNet) | not measured here | GPU; one pass per target per window |
| S7 (ClearerVoice) | not measured here | GPU; scales with speaking time, not screen time |

S1 dominates and is what to profile first on real hardware. `embedding_interval_seconds`
changes only disk, not runtime: insightface computes the vector as part of its
own pipeline whether or not it is written. `allowed_modules=["detection",
"recognition"]` skips the two bundled models this pipeline never uses and cut
that measurement by about 22%.
