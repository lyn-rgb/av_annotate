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
| `avannotate/tse/` | segment planning, the face-track crop video, the extractor |
| `avannotate/asr/` | source routing, the language vote, words, the recogniser |
| `avannotate/stages/base.py` | contexts, artifacts, and the resume record |
| `avannotate/stages/s0_preprocess.py` | S0 — probe, demux, shot boundaries |
| `avannotate/stages/s1_faces.py` | S1 — face detection and identity vectors |
| `avannotate/stages/s2_tracks.py` | S2 — detection-to-tracklet association |
| `avannotate/stages/s3_cluster.py` | S3 — tracklets to numbered people |
| `avannotate/stages/s4_diarize.py` | S4 — who spoke when |
| `avannotate/stages/s5_asd.py` | S5 — which face is talking |
| `avannotate/stages/s6_associate.py` | S6 — which face each speaker is |
| `avannotate/stages/s7_tse.py` | S7 — one person's voice, one file per segment |
| `avannotate/stages/s8_asr.py` | S8 — what each person said, and when |
| `avannotate/cli.py` | `avannotate run --stage … --input … --output …` |

Everything except the detector call itself is pure Python over JSON, which is
what makes it testable without a model or a GPU.

## Status

Implemented and tested: **S0 (probe, audio, shots)**, **S1 (detection +
embeddings)**, **S2 (tracking)**, **S3 (identity clustering)**, **S4
(diarization)**, **S5 (active speaker detection)**, **S6 (association)**,
**S7 (target-speaker extraction)** and **S8 (ASR)**, plus the deliverable
format, segmentation, and the QA gates. 483 tests.

Not yet implemented: S9–S11 — paralinguistic tagging, captioning, and the
compose step — plus the batch driver. See the plan for the stage DAG and model
choices.

**S4, S5, S7 and S8 need GPUs and packages that are not installed here.** Their
model adapters are written against documented interfaces and could not be run;
each names in its own docstring what to verify first. Everything that decides
what goes into a model pass, and what its output means, is separate, pure, and
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
avannotate run --stage s6-associate --input data/examples.txt --output ./outputs \
    --config configs/s6.associate.json
avannotate run --stage s7-tse     --input data/examples.txt --output ./outputs \
    --config configs/s7.clearvoice.json
avannotate run --stage s8-asr     --input data/examples.txt --output ./outputs \
    --config configs/s8.whisper.json
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

### S6 is where the three signals meet

S3 produced people, S4 produced turn-taking, S5 produced per-frame evidence of
who was talking; S6 decides who is who, and every stage after it reads that
decision. The matching algorithm is in `avannotate/associate.py` and was written
and tested before any of the stages that feed it existed -- S6 is the adaptation
around it, plus the two things the rest of the pipeline needs from the result:
each identity's speaking intervals, and the confidence that the assignment is
right.

Three outcomes are first-class rather than errors:

- **A speaker can come out off-screen** (`F000`). A diarizer hears a voice and no
  face matches it: narration, a phone call, someone out of frame.
- **A speaker can share a face with another** (`merged`). That is diarization
  over-segmentation -- one person split into two clusters -- and sending the
  spare cluster off-screen would relabel real speech as narration.
- **An assignment can be ambiguous** (`ambiguous`). Two faces explain a speaker
  comparably well; the best guess is still reported, with its margin, because a
  reviewer wants the competition as well as the answer.

### S7 hands the extractor a video with one face in it

`AV_MossFormer2_TSE_16K` is face-conditioned, which is why it was chosen: a
voice-conditioned extractor needs a clean enrolment clip, and producing those is
what S7 is for. But its public API takes a video path and then runs *its own*
face detection and lip-motion scoring to pick the speaker. There is no bbox
argument. Asking it for F001 and getting F002 is a real possibility.

So the choice is made before the model sees anything: cut a video containing only
F001's face, following the track S2 built and S3 clustered, and hand that over.
With one candidate it cannot pick wrong. The crop is square at 224 px with 40%
margin, the way the model's own training preprocessing expands a face, and a
frame where the track has no detection becomes a black tile rather than being
dropped — the sequence has to stay aligned with the audio.

Three consequences worth stating:

- **Cost tracks speaking time, not screen time.** Extraction runs per segment, so
  a person on screen for a minute and talking for five seconds costs five
  seconds.
- **Each segment gets 0.5 s of context on each side**, then has it trimmed back
  off. A crop starting exactly at the first phoneme starts with a mouth already
  moving, and the model has no baseline to compare against.
- **The crop videos are deleted unless `keep_crops` is set.** A thousand segments
  is a hundred thousand files, and they are a debugging aid rather than an
  artifact.

An identity S3 dropped but S6 still assigned speech to is reported under
`skipped` with its reason, not silently omitted — the count of people with audio
and the count of people in the annotation have to be reconcilable by whoever
reads `summary.json`.

### S8 asks for the mix, except when it cannot

There are two recordings of every segment: the original mix, which is what the
microphone heard, and S7's extraction, which holds one person. The mix is better
audio and the extraction is better *evidence*, and which one a segment gets is
decided by interval arithmetic before any model runs.

The rule is the overlap: the mix while this person is the only one talking, the
extraction when they are not. Both halves matter. A recogniser handed two
overlapping voices returns one fluent transcript containing both people's words,
and nothing in its output says which words were whose — so a segment read from
the mix during simultaneous speech is not slightly worse, it is silently
attributed to the wrong person. Run the other way, sending everything to the
extraction would put a separation model between the speaker and the recogniser
for the majority of segments that never needed it.

Overlap is measured against every *other* identity's speech, and a person
overlapping themselves is not overlap — S6 merges one person's turns, so a
merge boundary would otherwise read as two competing voices.

### S8 decides the video's language once, then tells the short segments

Whisper detects a language from a single 30-second window. A two-second
utterance is a small fraction of that evidence, and its guess is not a weak
version of the right answer — it is frequently a confident wrong one. A wrong
language does not produce a bad transcript of the right words; it makes the
recogniser *translate*, which reads perfectly and is worth nothing.

So the segments long enough to be evidence are transcribed first, with detection
left on, and the video's language is the duration-weighted winner among them —
weighted by speech time, not by how many segments each language appeared in, so
twenty two-second clips do not outvote two minutes. Segments too short to vote
are then transcribed a second time with that language forced.

A segment that confidently detects a *different* language from the video's is
left as it is and counted. A five-second English sentence inside a Chinese video
is either a real code-switch, which should stay English, or a detection error,
which forcing Chinese would turn into nonsense — and the two are not
distinguishable from here, so the rate is reported instead of guessed at.

Three consequences worth stating:

- **The cost follows speech, not video.** Each segment is transcribed once. The
  only extra work is one detection pass, and only for a video with no segment
  long enough to vote.
- **The extraction carries no context and the mix does** — 0.25 s per side, so
  the recogniser does not open on a clipped phoneme. S7 trimmed its output to
  the segment before writing it, so that path has none. Words in the padding are
  dropped by a coverage rule, which is why the text is assembled from words
  rather than taken from the recogniser's own segment text.
- **Hallucination flags are recorded, never gated.** `empty`, `no_speech`,
  `low_confidence`, `repetition` are the published heuristics for a recogniser
  inventing text. A video where somebody whispers trips them throughout and is
  still correct, so they exist to rank videos for review, not to reject any.

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
