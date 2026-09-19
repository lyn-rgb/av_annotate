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
| `avannotate/faces/` | detection, frame decoding, box geometry |
| `avannotate/stages/base.py` | contexts, artifacts, and the resume record |
| `avannotate/stages/s0_preprocess.py` | S0 — probe, demux, shot boundaries |
| `avannotate/stages/s1_faces.py` | S1 — face detection |
| `avannotate/cli.py` | `avannotate run --stage … --input … --output …` |

Everything except the detector call itself is pure Python over JSON, which is
what makes it testable without a model or a GPU.

## Status

Implemented and tested: **S0 (probe, audio, shots)** and **S1 (face detection)**,
plus the deliverable format, face/speaker assignment, segmentation, and the QA
gates. 190 tests.

Not yet implemented: S2–S11 — tracking, identity clustering, diarization, ASD,
target-speaker extraction, ASR, paralinguistic tagging, captioning, and the
compose step — plus the batch driver. See the plan for the stage DAG and model
choices.

### Running it

```bash
avannotate run --stage s0-preprocess --input data/examples.txt --output ./outputs
avannotate run --stage s1-faces --input data/examples.txt --output ./outputs \
    --config configs/s1.yunet.json
```

Every stage skips itself when its outputs are present and unchanged, so a rerun
after a crash costs only the video that was in flight.

### What S1 will not do

It does not threshold detections away. A face detector finds faces in framed
photographs and wall art — a gold record on a wall scored 0.64 across most of a
sample clip here — and tracking will not filter those either, because a wall
object is static and forms a long, stable, spurious track. The discriminator is
that a real face moves and a wall object does not, and that signal only exists
once there is a track. So S1 records the score and keeps everything, and the
filtering happens with evidence.

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
