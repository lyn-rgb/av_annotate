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
| `avannotate/associate.py` | S6's matching algorithm, written before its inputs |
| `avannotate/segment.py` | silence-free utterance spans, shared by S7–S9 |
| `avannotate/qa.py` | hard gates and reported metrics |
| `avannotate/ffmpeg.py` | ffprobe, audio demux, and the shot detector |
| `avannotate/faces/` | detection, tracking, clustering, Kalman filter |
| `avannotate/matching.py` | assignment and box-overlap primitives |
| `avannotate/audio/` | diarization turns and their geometry |
| `avannotate/asd/` | windowing, face crops, and prediction stitching |
| `avannotate/tse/` | segment planning, the face-track crop video, the extractor |
| `avannotate/asr/` | source routing, the language vote, words, the recogniser |
| `avannotate/paralinguistic/` | the tag vocabulary, three models, one tag |
| `avannotate/caption/` | frame planning, prompts, and the name check |
| `avannotate/compose/` | ten stages' records into one document |
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
| `avannotate/stages/s9_paralinguistic.py` | S9 — how each line was said |
| `avannotate/stages/s10_caption.py` | S10 — what the video and each shot look like |
| `avannotate/stages/s11_compose.py` | S11 — the deliverable and its quality report |
| `avannotate/requirements.py` | what each stage needs from the machine |
| `avannotate/cli.py` | `avannotate run --stage … --input … --output …` |

Everything except the detector call itself is pure Python over JSON, which is
what makes it testable without a model or a GPU.

## Status

All eleven stages are implemented and tested: **S0** (probe, audio, shots),
**S1** (detection + embeddings), **S2** (tracking), **S3** (identity
clustering), **S4** (diarization), **S5** (active speaker detection), **S6**
(association), **S7** (target-speaker extraction), **S8** (ASR), **S9**
(paralinguistic tagging), **S10** (captioning) and **S11** (compose) — plus the
deliverable format, segmentation, and the QA gates. 611 tests.

Not yet built: the **batch driver** that walks a corpus and isolates failures.
Every stage is resumable on its own, so the driver is a loop around the CLI
rather than new machinery — see the plan's M1.

**S4, S5, S7, S8, S9 and S10 need GPUs and packages that are not installed
here.** Their model adapters are written against documented interfaces and could
not be run; each names in its own docstring what to verify first. Everything
that decides what goes into a model pass, and what its output means, is
separate, pure, and tested.

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
avannotate run --stage s9-paralinguistic --input data/examples.txt --output ./outputs \
    --config configs/s9.paralinguistic.json
avannotate run --stage s10-caption --input data/examples.txt --output ./outputs \
    --config configs/s10.caption.json
avannotate run --stage s11-compose --input data/examples.txt --output ./outputs \
    --config configs/s11.compose.json
```

Every stage skips itself when its outputs are present and unchanged, so a rerun
after a crash costs only the video that was in flight.

### Before a batch: ask the machine what it has

```bash
avannotate doctor
```

Reports, per stage, whether this machine can run it — the packages, the
checkouts and the weight files — and for anything missing, the command that
fixes it. It resolves model paths the way the stages do, against the directory
of the config that names them, so it cannot report a model missing that a run
would have found.

Two of the six models are research repositories rather than packages, and two
need weights that no package fetches. `scripts/setup_server.sh` does the parts
that can be automated and prints the exact commands for the two that cannot — a
Google Drive link and a Zenodo download. Following the script and re-running
`doctor` is the whole setup.

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

### S9 reduces three models to one tag, by precedence rather than by score

No single model covers what the format asks for. An emotion classifier has no
notion of *whispering*, which is how a line was produced rather than how the
speaker felt; a voice-tagging model covers delivery but not the sounds that
replace words entirely; a general audio tagger covers those but knows nothing
about delivery. So all three run on every segment and one tag survives.

**The order is event, then delivery, then emotion**, and it is precedence rather
than the highest score, because the three scores are not comparable — one is a
softmax over nine classes and the others are sigmoids over hundreds of
independent labels. Letting the largest number win would make the priority
between dimensions an artefact of how each model happens to be calibrated.

That has a visible consequence: a dimension can win while scoring *lower*, so
`TagChoice.margin` goes negative, and a negative margin is the case worth
looking at — right by the rule and wrong by the numbers. The reason string says
so explicitly.

Three more things the reduction does:

- **The vocabulary is closed**, and every tag is proved against the script
  format's own pattern. A tag containing a space or a colon would not fail
  loudly; it would be parsed as part of the spoken text, on every line it
  appeared on. The test renders and re-parses each tag to check the claim.
- **`neutral` is never emitted.** It is a result, not a description, and the
  format makes the tag optional precisely so an utterance with nothing to say
  about it renders without one. A rendered `neutral:` looks like a finding.
- **A confident unmapped label is reported, not guessed at.** That signal is
  meant to mean *the model is sure of something we have no word for*, so it is
  gated on the score and excludes labels known and deliberately unused —
  otherwise AudioSet's `Speech`, which fires on every speech segment, would make
  the report a constant.

### S9 reads the extraction; S8 reads the mix

Opposite routing, for the opposite reason. S8 wants the best *fidelity*, so it
reads the mix whenever one voice is on it. S9 wants the best *isolation*,
because a general audio tagger listening to a mix reports what is audible rather
than what this person did — a cough from across the room lands in the segment of
whoever happens to be speaking, and `cough` on a line that was not coughed is a
wrong tag nothing downstream can detect.

The cost is real and is not hidden: the extraction has been through a separation
model, and a separated voice is not a clean one. The taggers are classifiers
rather than transcribers and tolerate that better than a recogniser, but this
trades one error for another, so all three models' full score lists are kept in
`tags.json` and the QA report surfaces both the count of tags per dimension and
the close calls.

### S10 tells the model who is in the shot, then checks its answer

Nothing in a still frame says which face is F001. Asking a captioning model to
*identify* people would be asking it to guess, and a confident guess about who
someone is cannot be told from a correct one once it is rendered. So the roster
is computed from tracking and handed over as a constraint — "these frames
contain tracked people F001, F002; refer to a person by their identifier and use
no other" — which turns an unanswerable question into an answerable one.

That alone would be a hope rather than a guarantee, so the answer is checked. A
name the model uses that was not on the roster it was given is removed before the
caption is written, and the removal is recorded. Nothing downstream reads a
caption's names, so an unchecked one would ship.

Two kinds of wrong name are told apart, because they mean different things about
the model: a name that exists nowhere in the video is a hallucination, while a
real person named in a shot they are not in is a grounding failure. Both are
stripped; a run with many of one and none of the other is telling you which.

The check is deliberately narrow. It does not judge the description — the model
may be right that somebody is in the room — it only removes what nothing in this
pipeline can support.

**Names in captions are bare (`F001`), not bracketed (`<F001>`).** Angle
brackets in this format belong to the utterance grammar, so a parser that sees
one should be able to assume a speech line follows. Captions are not utterances.

### What S10 will not do

It does not caption from video. A shot is described from a handful of stills,
spread across the shot and taken from the middle of each interval rather than
its edges — the first frame of a shot is the one most likely to be a transition,
and the last is the one most likely to be a cut. The count follows the shot's
length up to a cap of eight, because request cost grows with the number of
images and the tenth frame of a shot adds less than the third.

The video-level caption takes one frame from the start of each shot rather than
an even spread over the timeline: an even spread lands wherever the cuts happen
to fall, while one frame per shot covers every distinct scene the video contains.

It also does not caption from the audio. Both levels are visual only — the
format's own example has no dialogue in either — which is what makes this stage
independent of S4 through S9 and free to run on any schedule.

### S11 joins on the segment name and nothing else

Nine stages left records behind; S11 reads them all and writes `annotation.txt`,
`annotation.json` and `qa/report.json`. It runs no model, which is the point:
this is where a mistake becomes permanent, and it is also the stage with nothing
in it that can fail unpredictably.

Every join is on S7's segment name. S7 decided which spans exist and wrote one
audio file for each; S8 transcribed those files, S9 tagged them, and S11 puts the
three back together by name. Joining on a timestamp, an index or a sort order
would be a second definition of what a segment is, and two definitions disagree
on exactly the videos nobody checks.

`annotation.json` is written by one function and read by another, so the writer
checks its own output through the reader before returning it — a document this
pipeline's own reader rejects would otherwise fail at the far end of a batch, on
the one video nobody is watching.

### A word can start before the line it is in

Measured on the sample corpus: an utterance beginning at 2.832 s whose first word
the recogniser places at 2.582 s. Nothing is wrong — the segment span is trimmed
from a VAD that shaves onsets, and S8 reads 0.25 s of context around every
segment, so a word straddling the boundary is kept if most of it falls inside.
The word's own start is reported rather than clamped to the line, because
clamping it would replace what the recogniser said with what the pipeline
decided, and the pipeline's boundary is the less reliable of the two.

The consequence for a consumer is that word timestamps are not guaranteed to
nest inside their utterance. Nothing downstream should assume they do.

### The one real gap: speech from off camera

**Nothing in this pipeline produces an utterance for a speaker who is not on
screen.** An off-screen speaker has no face track by definition, S7 skips any
identity without tracklets, and so no audio is extracted, nothing is transcribed,
and no `F000` line reaches the deliverable. `F000` exists in the schema and the
grammar reserves it; nothing currently writes it.

This is not hidden: the speech-accounting gate compares what was attributed to
what the diarizer heard, and for a video with off-camera speech that gate fails,
with the size of the gap in the report. That is the gate doing its job — it is
the measurement the plan asked for before deciding whether off-screen support is
worth building, and it is far better read from a report than assumed either way.

The one-line version: **run a small batch first and look at
`qa/report.json`'s `speech_accounting` detail.** If the gaps are seconds, the
corpus has narration and the off-screen branch is needed. If they are
milliseconds, it does not.

### What S11 will not hide

The QA report is written whether the gates pass or fail, and the failing gate
names are in the stage's own summary so a batch driver can route on them without
parsing the report. A video that fails is flagged for review rather than dropped
or passed quietly.

### Where the deliverable's files actually are

`annotation.json` sits beside the stage directories, and its `audio_path` fields
point at the real extraction — `s7-tse/audio/F001/F001_0000.wav` — rather than at
a flattened `audio/` tree. That is a deviation from the layout sketched in the
plan, made because the alternative is either a second copy of every audio file or
a set of symlinks, and because the path is *in* the JSON: a consumer following
`audio_path` cannot be wrong about where a file is. The stage directories are the
private working area, not a leak.

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
