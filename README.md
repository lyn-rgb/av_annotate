# avannotate

Offline annotation pipeline for multi-person video. From one video it produces
face tracks, per-person speech separated by target-speaker extraction, per-segment
transcripts, paralinguistic tags, and a two-level visual description — rendered
as a structured script.

The output feeds [SyncEdit](../SyncEdit), an LTX-2 based instruction-driven
audio-video editor, but nothing here imports from it.

## The deliverable

```
[GLOBAL]
Two people are discussing in a spacious living room.

[SHOT 1 0.0s-12.4s]
A bright living room with a sofa and a coffee table.
<F001> whispering: <S>I was late for work today</S>
<F002> surprised: <S>What's going on?</S>

[SHOT 2 12.4s-31.0s]
The camera moves closer to the window.
<F001> <S>I forgot my phone</S>
```

Grammar: ``<F(\d+)>\s*([\w-]+)?:?\s*<S>(.*?)</S>``. The tag is optional; the
colon is required when it is present. `F000` is reserved for off-screen speech
(narration, a phone call), because the grammar requires a face tag and an
unattributed utterance therefore needs an id rather than no tag.

`annotation.json` is the machine-readable form and the only artifact downstream
code reads. Every intermediate the pipeline writes is private to its stage, so a
change to the script format re-renders and never re-runs a model.

## Layout

| Module | Role |
| --- | --- |
| `avannotate/schema.py` | the deliverable's data contract |
| `avannotate/annotation.py` | render and parse the script; round-trips exactly |
| `avannotate/interval.py` | half-open time intervals and set operations |
| `avannotate/associate.py` | S8 — which diarization speaker is which face |
| `avannotate/segment.py` | S10 — silence-free utterance spans |
| `avannotate/qa.py` | hard gates and reported metrics |
| `avannotate/ffmpeg.py` | ffprobe, audio demux, and the shot detector |
| `avannotate/faces/` | detection, tracking, clustering, Kalman filter |
| `avannotate/matching.py` | assignment and box-overlap primitives |
| `avannotate/audio/` | diarization turns and their geometry |
| `avannotate/asd/` | windowing, face crops, and prediction stitching |
| `avannotate/stages/base.py` | contexts, artifacts, and the resume record |
| `avannotate/stages/s0_preprocess.py` | S0 — probe, demux, shot boundaries |
| `avannotate/stages/s1_faces.py` | S1 — face detection and identity vectors |
| `avannotate/stages/s2_tracks.py` | S2 — detection-to-tracklet association |
| `avannotate/stages/s3_cluster.py` | S3 — tracklets to numbered people |
| `avannotate/stages/s4_diarize.py` | S4 — who spoke when |
| `avannotate/stages/s5_asd.py` | S5 — which face is talking |
| `avannotate/cli.py` | `avannotate run --stage … --input … --output …` |

Everything except the detector call itself is pure Python over JSON, which is
what makes it testable without a model or a GPU.

## Status

Implemented and tested: **S0 (probe, audio, shots)**, **S1 (detection +
embeddings)**, **S2 (tracking)**, **S3 (identity clustering)**, **S4
(diarization)** and **S5 (active speaker detection)**, plus the deliverable
format, face/speaker assignment, segmentation, and the QA gates. 388 tests.

Not yet implemented: S6–S11 — association, target-speaker extraction, ASR,
paralinguistic tagging, captioning, and the compose step — plus the batch
driver. See the plan for the stage DAG and model choices.

**S4 and S5 need GPUs and packages that are not installed here.** Their model
adapters are written against documented interfaces and could not be run; each
names in its own docstring what to verify first. Everything that decides what
goes into a model pass, and what its output means, is separate, pure, and
tested.

### Running it

```bash
avannotate run --stage s0-preprocess --input data/examples.txt --output ./outputs
avannotate run --stage s1-faces  --input data/examples.txt --output ./outputs \
    --config configs/s1.insightface.json
avannotate run --stage s2-tracks --input data/examples.txt --output ./outputs \
    --config configs/s2.default.json
avannotate run --stage s3-cluster --input data/examples.txt --output ./outputs \
    --config configs/s3.default.json
avannotate run --stage s4-diarize --input data/examples.txt --output ./outputs \
    --config configs/s4.diarizen.json
avannotate run --stage s5-asd     --input data/examples.txt --output ./outputs \
    --config configs/s5.loconet.json
```

Every stage skips itself when its outputs are present and unchanged, so a rerun
after a crash costs only the video that was in flight.

**S3 needs insightface.** YuNet detects but produces no identity vectors, and S3
refuses to guess rather than fragmenting every identity into a separate person.
Use `configs/s1.insightface.json`.

### The sampling stride decides how much S3 has to repair

S2 associates by bounding-box overlap, and overlap stops working as soon as the
camera moves far enough between two sampled frames. On the sample corpus a fast
pan moved one face 176 px in 0.125 s — more than its own width — and the track
broke. S3 rejoins those fragments by appearance: the two halves of that person
score 0.78 cosine against each other and 0.08–0.12 against everyone else.

But stride 3 shatters that clip into ten tracklets, six of them a single frame,
and a one-frame tracklet has no appearance to match on. Measured:

| clip | stride 3 | stride 1 |
| --- | --- | --- |
| two profile faces | 2 tracklets → 2 people | — |
| couch conversation | 2 tracklets → 2 people | — |
| camera pan | 10 tracklets → 3 people, 6 dropped | 3 tracklets → 2 people, 0 dropped |

Two people is the right answer for all three. So on a corpus with camera motion,
prefer `configs/s1.insightface.dense.json` (stride 1); the extra detection cost
buys tracklets long enough to cluster.

### What S1 will not do

It does not threshold detections away. A face detector finds faces in framed
photographs and wall art — a gold record on a wall scored 0.64 across most of a
sample clip under YuNet — and tracking will not filter those either, because a
wall object is static and forms a long, stable, spurious track. The discriminators
are that a real face moves and a wall object does not (S2's `motion`), and that
insightface's detector does not fall for the gold record at all: with it, the
couch clip yields exactly two faces in every frame where YuNet yielded three.
So the stages record the evidence and the filtering happens with it.

### What S0 settles

The video's duration is authoritative, not the audio track's. AAC codes in
fixed-size frames, so a demuxed track runs 23–51 ms longer than the video on the
sample corpus. Slicing audio by a video-derived timestamp is then off by up to a
frame and a half — enough to cut a word in half before target-speaker extraction
sees it. `Timeline.audio_delta` records the surplus rather than hiding it, and
`load_timeline` is how later stages learn to clamp to the video's end.

## Development

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check avannotate tests
.venv/bin/python -m mypy avannotate
```

Test fixtures are generated with ffmpeg at test time, so the suite runs anywhere
ffmpeg exists; tests over the real sample corpus skip when `data/examples/` is
absent.

Model dependencies are grouped by stage in `pyproject.toml` (`faces`,
`diarization`, `asd`, `tse`, `asr`, `caption`) so a server installs only what it
runs. The core needs none of them.
