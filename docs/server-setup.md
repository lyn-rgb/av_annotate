# Server setup

```bash
scripts/setup_venv.sh        # the virtualenv, and every package in it
scripts/setup_server.sh      # the two repository checkouts, and what they need
scripts/download_models.sh   # every model, into ./models
avannotate doctor            # says what is still missing, and how to fix it
```

Those four are the setup, in that order.  `setup_venv.sh` is the one to run on a
fresh GPU box: it creates `./.venv` -- which is the path `run_batch.sh` and
`download_models.sh` already look for -- installs torch ahead of everything that
depends on it, installs the pipeline and every runtime extra, installs DiariZen
if its checkout is present, and then checks that the environment can actually
see the cards rather than just that pip exited zero:

```bash
scripts/setup_venv.sh --gpus 4
```

The checks are the point.  Two failures on a GPU box are silent rather than
loud -- a CPU-only torch wheel, and `onnxruntime-gpu` installed without a
matching cuDNN, which falls back to the CPU provider and makes S1/S3 about ten
times slower with nothing saying so.  Both are reported explicitly, along with
each card's compute capability and the architectures the wheel has kernels for.
It also skips torch entirely when nothing in the requested extras needs it --
`--extras asr` is CTranslate2 only, and 2.5 GB of CUDA libraries for a stage
that never calls them is a waste worth avoiding.

**`download_models.sh` tries ModelScope before the
Hugging Face mirror**, and writes what it fetches into the layout
`huggingface_hub` reads, so the repo ids in `configs/` keep resolving unchanged.

Why that order: `hf-mirror.com` is not a way around a blocked network. It is
itself abroad — it resolves to `160.16.86.14`, a Sakura address in Japan — so it
is blocked by exactly the things that block `huggingface.co`. ModelScope is
domestic, carries most of these checkpoints under the same repo ids, and is
reachable on precisely the networks where the mirror is not. Where ModelScope
has no copy, the mirror is still tried.

**Every checkpoint now comes from ModelScope**, including the three whose home
is abroad and which used to need a route out: insightface's `buffalo_l` (a
GitHub release), DiariZen's checkpoint (Hugging Face only), and PANNs' `Cnn14`
(Zenodo). The last two arrive from community uploads, which is worth being
precise about — see the notes on each below. The GitHub URLs are still tried
first where they are the canonical source, so a machine that can reach GitHub
is unaffected.

`HF_ENDPOINT` still matters when the mirror *is* used, because it is honoured by
every library using `huggingface_hub` — including ClearerVoice and
faster-whisper, whose checkpoints are fetched by their own code rather than by
anything here. Set these for a run:

```bash
export HF_HOME=/path/to/models/hf
export HF_ENDPOINT=https://hf-mirror.com
```

**`run_batch.sh` sets `HF_HOME`, `MODELSCOPE_CACHE`, `TORCH_HOME` and
`HF_ENDPOINT` itself**, pointing at the checkout's `models/`, so a run finds
what `download_models.sh` put there without any exporting. All of them still
respect an existing value. That matters more than it looks: the configs name
their checkpoints by repo id, so with the cache in the wrong place
`huggingface_hub` looks in `~/.cache/huggingface`, finds nothing, and either
fails or fetches all 26 GB a second time.

**None of the four defaults to `$HOME` or `/tmp`**, and that is deliberate. On
a cluster the home directory is usually on a quota and `/tmp` is on the node's
own disk, so a cache placed there is either too small for a 64 GB model set or
gone at the next reboot. Everything the pipeline chooses for itself lives under
the checkout. The one path it does not choose is the output directory — which
is why `--output` has no default and is required: name one on the same
persistent volume as the checkout.

`TORCH_HOME` is the newest of the four and the easiest to get wrong, because a
mismatch there fails rather than degrades. It holds the seeded VGGish weights
(`scripts/seed_vggish.py`). If `setup_venv.sh` and `run_batch.sh` disagreed
about where that is, S5 would not fall back to anything — the `torchvggish`
constructor asks `torch.hub` to download 275 MB from a GitHub release, which a
server with no route there cannot do. Those weights are ones LoCoNet's own
checkpoint overwrites immediately, which is the entire reason the seed exists.

**On a machine with no route to huggingface.co, one more is needed:**

```bash
export HF_HUB_OFFLINE=1
```

This is the one that cannot be defaulted, and it is the difference between a
cached checkpoint resolving and the same checkpoint being reported missing.
Without it, `from_pretrained` asks the network to resolve `main` before it
consults the cache at all — so a machine with no route out fails on a model
that is sitting right there on disk. It is not set by default because it also
turns every miss into a hard failure instead of a download, which is a decision
rather than a default. Set it once `download_models.sh` has reported `ok`.

`MODELSCOPE_CACHE` is where `funasr` looks for emotion2vec, and it has to match
the path `download_models.sh` wrote into. Ordering matters for that one
checkpoint: it is fetched with the `modelscope` package rather than over plain
HTTP, so `pip install modelscope` has to come first, and the fetch is skipped —
not failed — if it has not. Install the S9 packages and re-run the download if
you want it carried rather than fetched on first use:

```bash
pip install funasr modelscope
scripts/download_models.sh --only s9-paralinguistic
```

### S7 reads its checkpoint relative to the working directory

Worth knowing before it bites: `clearvoice` hardcodes its checkpoint path as
`checkpoint_dir/AV_MossFormer2_TSE_16K`, relative to wherever the process was
started, and its public API takes no `checkpoint_dir` argument. So the run has
to be pointed at what was downloaded:

```bash
mkdir -p checkpoint_dir
ln -s /path/to/models/clearvoice/AV_MossFormer2_TSE_16K \
      checkpoint_dir/AV_MossFormer2_TSE_16K
```

It skips the download entirely when `checkpoint_dir/last_best_checkpoint` is
present, which is why nothing needs to reach Hugging Face once that is in place.

### The three community uploads, and what was checked

`buffalo_l`, DiariZen's checkpoint and PANNs' `Cnn14` are not published on
ModelScope by their authors — they arrive from user uploads. So each was checked
against something outside the upload rather than trusted because it downloaded
without complaint.

**`buffalo_l`** (`SiYuan044/buffalo_l`, cross-checked against
`muse/insightface_model_buffalo_l`). Two unrelated uploads carry the same five
files at the same byte sizes. Then it was loaded into insightface and run on a
frame from `data/examples/`: 2 faces detected, 512-d embeddings, working
`gender`/`age` attributes. That exercises the SCRFD detector, the ArcFace
recognition model and the genderage model, so all three are real.

**PANNs `Cnn14`** (`pengzhendong/panns`, whose repo holds sixteen PANNs
checkpoints — which is why the one file is taken by name, not by repo). Loaded
with `panns_inference` and run on three seconds of an example video: 2048-d
embedding, top labels `Speech 0.87`, `Music 0.82`, `Male speech, man speaking`.
Coherent AudioSet output, so these are trained weights rather than a renamed
file.

**DiariZen** (`lvzepeng4youcash/DiariZen`). This one could not be run — running
it means executing code from the same upload. What was checked instead is that
the checkpoint is consistent with the architecture its own `config.toml`
declares: 24 transformer layers matching `WAVLM_LARGE_S80_MD`,
`encoder_embed_dim` 1024, `weight_sum` of shape (1, 25) for
`wavlm_layer_num = 25`, `proj` (256, 1024), and `conformer_layer.0..3` for
`num_layer = 4`. Every one matches, and `config.toml` is DiariZen's own schema,
naming `wavlm_src = "wavlm_large_s80_md"` — the `s80-md` of the repo id the
configs ask for.

What that does **not** establish: the upload is labelled `DiariZen`, not
`...-v2`, and two training runs can share an architecture exactly. So it will
load and run, and it may be v1 rather than v2. If diarization quality looks
wrong against a known-good run, that is the first thing to suspect; the fix is
the Hugging Face original, once there is a route to it.

That upload also carries `DiariZen-main.zip`, which is the repository itself --
the thing `setup_server.sh` clones from GitHub, and therefore cannot get on the
network this whole path exists for.

### The two repository checkouts

The checkpoints are all reachable now; the *code* is a separate problem, because
these two are git clones rather than packages.

**LoCoNet** (`SJTUwxz/LoCoNet_ASD`) has no mirror anywhere — not on ModelScope
under any spelling, and `gitclone.com` serves an empty repository for it, while
`gitcode`'s and `gitee`'s GitHub mirrors 404/405. The CDNs that could serve a
repository's files (`jsDelivr`, `statically`, `githack`) are all abroad, like the
ghproxy services — `ghproxy.net` is OVH France, `gh-proxy.com` is Cloudflare,
`ghfast.top` is US, which is why none of them are a way around anything.

So a checkout has to be carried, the same way the weights are. Once one is in
place at `models/loconet/LoCoNet_ASD`, `setup_server.sh` skips the clone
entirely — `install_repo` treats the presence of `loconet.py` as "already
there" — and `make_offline_bundle.sh` packs it into the bundle instead of
downloading it. That is what the third argument to `fetch_repo` is for.

Verifying a carried checkout is one command, and it is worth running because
this is the stage where a wrong file would produce plausible nonsense rather
than fail:

```bash
python -c "
from avannotate.asd.model import LoCoNetAsd
m = LoCoNetAsd(checkpoint='models/loconet/loconet_AVA.model',
               repo='models/loconet/LoCoNet_ASD', device='cpu')
print(m.load_report)"
# -> {'missing': 0, 'unexpected': 0}
```

**DiariZen** comes from the same place, and the checkout is carried rather than
cloned for the same reason. It is the `DiariZen-main` tree out of the ModelScope
upload — which is the right one to use, and not merely the available one: the
checkpoint came from that same upload, so the two are the same revision by
construction. A separate `DiariZen` tree that is also on hand is a *different*
revision (it carries extra multi-channel modules and its
`model_wavlm_conformer.py` differs), and nothing has checked it against this
checkpoint.

Two things that look like gaps and are not:

* `dscore/` is an empty directory. It is a git submodule that a tarball cannot
  carry, and nothing needs it — it appears in neither `requirements.txt` nor
  `pyproject.toml`, and no module under `diarizen/` imports it. It is the
  scoring toolkit, used to *evaluate* diarization, not to run it.
* `constraints.txt` is absent, and that is deliberate. It pins
  `torch==2.1.1`/`torchaudio==2.1.1`, and `setup_server.sh` passes it as `-c` to
  the vendored `pyannote-audio` install — where pip, resolving pyannote's own
  torch dependency, would honour the pin and *downgrade* a working torch to get
  there. Absent, that step falls through to its `|| warn`. `setup_venv.sh` does
  not pass a constraints file at all.

DiariZen's `pyproject.toml` declares no runtime dependencies at all (flit
metadata and optional test/docs groups only), so `pip install -e .` alone
installs nothing but the package — `requirements.txt` is what actually brings
the dependencies, which is why both are installed.

**If the server cannot reach github.com, start with the bundle instead.** More
depends on GitHub than this repository's own three URLs: insightface fetches
`buffalo_l` from a GitHub release, and so does torchvggish's VGGish. Building the
pipeline's own environment is therefore not enough — the dependencies reach out
on their own, at first use, from inside code we do not control.

```bash
# on a machine that can reach the internet
scripts/make_offline_bundle.sh --out ./offline-bundle
scp offline-bundle.tar server:/tmp/

# on the server
tar xf /tmp/offline-bundle.tar -C /tmp
scripts/setup_server.sh --from-bundle /tmp/offline-bundle
```

The bundle holds a tarball of each repository, the model files whose home is a
GitHub release, `MANIFEST.json` with a sha256 per file, and `repos/SOURCES.txt`
recording the commit each tarball came from. Nothing in it is fetched again on
the server. Add `--with-wheels` if PyPI is out of reach too, and `--with-hf` to
cover the Hugging Face checkpoints.

### LoCoNet's weights: the one file that has to be carried

Everything else can be re-fetched from some mirror. **`loconet_AVA.model` is
only on Google Drive**, and a locked-down server usually has no route to it —
so it is a file you carry, not a file you download there. 131 MB.

`make_offline_bundle.sh` finds it in either of two ways, in order:

1. **Already on the bundle machine**, at `models/loconet/loconet_AVA.model` —
   the path `configs/s5.loconet.json` expects, so if you downloaded it there it
   is copied in with no argument.
2. **`gdown`** (`pip install gdown`), which fetches it by file id from the
   Drive link in the repository's README.

Either way what lands in the bundle is **checked before packing**: that it is a
torch state dict, and that its keys begin `model.module.` the way the release
does. That check earns its place — a Drive download that hits a quota or a
sign-in page returns *HTML with status 200*, and `gdown` saves it without
complaint. Without the check you get a 2 KB web page with a `.model` extension
that fails on the server instead of on the machine where you could still fix it.

If neither route works on the bundle machine, carry the file by hand:

```bash
# anywhere with a browser, then
scp loconet_AVA.model server:/path/to/models/loconet/
```

The server is then done with it: nothing at run time reaches Google Drive. The rest of this document is the detail behind them:
what each model needs, what could not be verified without a GPU, and what to
check on the first run of each.

`doctor` is the part worth knowing about. It reports per stage whether this
machine can run it, resolves the model paths exactly as the stages resolve them
— against the directory of the config file that names them — and never imports
a module to check it, so it answers in a second rather than loading 60 GB of
weights. Run it before a batch rather than discovering a missing package at
video four hundred.

The stages before S4 need nothing but ffmpeg, numpy, and OpenCV, and run
anywhere. S4 needs DiariZen, which is not on PyPI and wants a GPU.

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

**This one works, and a plain clone is enough — verified end to end with the
real repository, the real VGGish weights and the real release checkpoint, no
stubs anywhere.** It is not a package, so it needs a checkout and a couple of
things its own instructions omit:

```bash
git clone https://github.com/SJTUwxz/LoCoNet_ASD      # or: scripts/setup_server.sh
pip install resampy                                    # see below
# then download loconet_AVA.model from the Google Drive link in its README
```

### The three things nobody mentions

1. **`resampy` is missing from its requirements.** Its in-tree `torchvggish`
   imports it, so model construction fails with `ModuleNotFoundError: resampy`
   and nothing in the repository warns you. It is in this project's `asd` extra.

2. **Building the model downloads 275 MB of VGGish weights that are already in
   the checkpoint.** The constructor calls `torchvggish.VGGish(...)` with
   `pretrained=True`, so it fetches them from
   `github.com/harritaylor/torchvggish/releases` at construction time — and then
   LoCoNet's own checkpoint overwrites every one of them a moment later.

   **They can be taken from the checkpoint instead.** Its `audioEncoder`
   sub-state is key-for-key identical to VGGish's own state dict — verified: 18
   keys each, set equality both ways — so the hub cache can be filled from the
   131 MB file you must download anyway:

   ```bash
   python - <<'EOF'
   import os, torch
   ckpt = torch.load("models/loconet/loconet_AVA.model", map_location="cpu")
   inner = "model.module.model.audioEncoder."
   audio = {k[len(inner):]: v for k, v in ckpt.items() if k.startswith(inner)}
   target = os.path.expanduser("~/.cache/torch/hub/checkpoints/vggish-10086976.pth")
   os.makedirs(os.path.dirname(target), exist_ok=True)
   torch.save(audio, target)
   EOF
   ```

   That is 20 MB of local copying against a 275 MB download — which matters most
   where it is least available: this download failed four times on an ordinary
   connection while writing this, each time leaving a truncated file that
   `torch.load` reports as `unexpected EOF` rather than as an incomplete
   download. `scripts/setup_server.sh` does this automatically once the
   checkpoint is in place.

3. **`loconet.py` cannot be imported.** It does `from xxlib.utils.distributed
   import ...` and `xxlib` exists nowhere in the repository — a leftover from
   the authors' private training harness. That is the training script, not the
   model, and **nothing here imports it**: the adapter needs
   `model/loconet_encoder.py` and `loss_multi.py`, and both import cleanly from
   a plain checkout. No patching, no vendoring.

### What about it is not what it looks like

All read off the source and confirmed by running it. Each of the first four was
wrong when it was assumed, and each would have failed quietly:

- **There is no inference entry point.** `Loconet.forward(audioFeature,
  visualFeature, labels, masks)` dereferences `labels` and `masks` before any
  branch and returns a loss tuple — it cannot be called to score anything. The
  adapter drives the four frontends and the classifier head itself, which is
  what the repository's own `evaluate_network` does internally.
- **The class is `locoencoder`**, not `LoCoNet` — that capitalisation appears
  nowhere in the repository.
- **The head is `lossAV.FC`**, a `Linear(256, 2)`, and its weights live in the
  checkpoint rather than in the encoder, under `model.module.model.lossAV.*`.
  Scoring is `softmax(-1)[:, 1]`; there is no sigmoid in this model.
- **`cropScale` is not in this repository.** That constant is TalkNet's. LoCoNet
  crops the detected box directly and resizes to 112×112 square, flattening the
  aspect ratio, so the margin here is **0.0** and was wrong at 0.40.
- **The audio is `[4T, 64]`, not `[4T, 128]`.** 64 is VGGish's mel-band count;
  128 is the *output* width of the audio frontend.
- **Crops are 0..255.** The visual frontend normalises its own input,
  `(x / 255 - 0.4161) / 0.1688`, so dividing by 255 beforehand applies the shift
  twice.

### The speaker axis is exactly three

`ConvLayer` builds `Conv2d(256, 256*s, (s, 7))` with `s = NUM_SPEAKERS` — the
count is baked into the weight shapes — and it convolves *across* the speaker
axis. So a group narrower than three has to be padded (the adapter pads with
black tiles, the repository's own convention, and drops them from the result),
and a group wider than three cannot be scored in one pass at all. The window
planner caps groups at three for this reason; passing four raises rather than
being trimmed, because trimming would attribute one person's speech to another.

### Licence

**The weights carry no declared licence.** There is no `LICENSE` at the
repository root; the only one anywhere covers the vendored `dlhammer`, and the
README says nothing about the weights. Fine for the research use this project
was built for, but it is a compliance question rather than a technical one, and
it is worth an explicit decision rather than an assumption.

### What to verify on the first run

1. **That the weights loaded at all.** `load_report` on the adapter counts the
   keys the encoder recognised and the ones it did not. A checkpoint whose keys
   do not match shows up there as `missing` being most of the model, and it will
   `load_state_dict(strict=False)` happily without it. **Verified against the
   real release**: 272 keys, `missing: 0`, `unexpected: 0`.

   That check earned its place. The released file is a plain save of the
   training wrapper, so every key begins `model.module.` — and what is left
   after stripping that is two *siblings*, `model.<encoder>` and
   `lossAV.<head>`. An earlier version of this adapter looked for
   `model.lossAV.`, and it passed a test against a hand-built checkpoint —
   because that checkpoint had been written from the same wrong assumption.
   Only the real file settled it.
2. **That the traces separate.** Run S5 on a clip where one person speaks and
   another is visibly silent and look at whether the two traces do.

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

## faster-whisper

```bash
pip install faster-whisper
```

It pulls CTranslate2 and downloads the checkpoint from Hugging Face on first
use, so the first run needs network access. `large-v3` is about 3 GB in fp16.
Set `download_root` in the stage config (or `HF_HOME`) to a shared cache if
several jobs run at once.

### What is verified, and what is not

The interface was read off **faster-whisper 1.2.1** — the signature, the field
names, and the two properties below are all confirmed against that release's
source, not recalled. Nothing has been *run*, so behaviour on real audio is
open. The two properties that matter here:

- **`word.word` keeps its leading space**, and upstream's own test asserts
  `segment.text == "".join(word.word for word in segment.words)`. That is the
  same join `avannotate/asr/text.py` performs, and it is why the words are
  joined with no separator: a rule that inserted a space would break Chinese and
  a rule that stripped the words first would glue English together.
- **A numpy array is used as-is, at 16 kHz, with no resampling and no check.**
  `decode_audio` is only called for a path or a file object. So an 8 kHz file
  does not fail — it transcribes the wrong frequencies against the wrong time
  base and returns every timestamp at half scale, reading plausibly while doing
  it. `avannotate/asr/audio.py` checks the rate and refuses rather than
  resampling past it.

### Where the GPU time goes, and what is not worth trying

One decode per segment, so the cost tracks speaking time. `beam_size: 5` is the
default and is most of the cost; a batch run that wants speed can drop it to 1
and lose some accuracy on hard audio.

`BatchedInferencePipeline` is **not** usable here despite looking like the
obvious speedup: it batches overlapping *windows of one input*, not several
inputs, and its defaults differ from `WhisperModel.transcribe` in ways that
matter (`vad_filter=True`, `without_timestamps=True`, and it silently forces
`condition_on_previous_text=False`). Since this stage already isolates one
speaker per call, there are no windows to batch — the parallelism is across
videos, not within one.

### What to verify on the first run

1. **`detect_language` accepts the sample array.** The sample fallback passes a
   1-D float32 array and nothing else. Leave `language_detection_threshold` at
   its default: it is annotated `Optional[float]` but dereferenced unguarded, so
   passing `None` explicitly raises rather than defaulting.
2. **Timestamps against a known clip.** Transcribe one segment, listen to it,
   and check the words land where they are said. A wrong `SegmentSource.origin`
   shifts a whole segment by a constant, which no amount of reading the
   transcript will reveal.
3. **The language on a monolingual clip.** If `summary.json` reports something
   other than the language you know the video is in, the vote is working on
   segments too short to be evidence, and `min_detect_seconds` wants raising.

The cheapest end-to-end check is one video whose overlap you can hear: look at
`transcripts.json` and confirm the segments marked `"source": "extracted"` are
the ones where two people were talking at once.

## Paralinguistic tagging (S9)

Three packages, one per dimension, and each can be installed on its own:

```bash
pip install funasr modelscope          # emotion
pip install transformers torch         # delivery
pip install panns-inference            # events
```

### Three things about these models that are not what the docs suggest

Read off the released artifacts, not recalled — each of these would otherwise
produce a plausible wrong answer.

**`laion/voice-tagging-whisper` is a generator, not a classifier.** Its Hugging
Face tag says `audio-classification` and its `config.json` contains
`classifier_proj_size`, but it is a `WhisperForConditionalGeneration` with no
classification head and no `id2label`. Loading it via
`WhisperForAudioClassification.from_pretrained` **succeeds**, attaching a
randomly initialised head — every score it returns is noise, and nothing warns
you. It also has **no published label set**: its own card counts "194 unique
tags" in a sample of 570 outputs and lists only the most frequent, because a
generator has no closed vocabulary. The adapter therefore generates a
comma-separated string, splits on commas, and maps what it recognises through
`vocabulary.DELIVERY_LABELS`, dropping the rest. It returns no probabilities, so
delivery tags are recorded with a score of 1.0 (presence) and this dimension's
`delivery_min_score` admits everything the model named.

**PANNs takes 32 kHz and resamples nothing.** Its mel front end is fixed there,
so feeding it the pipeline's 16 kHz audio does not fail — every band lands in
the wrong place and it returns confident tags for a signal it never heard. The
adapter resamples with `librosa` (which `panns-inference` already depends on).
Its `inference()` returns a **2-tuple** `(clipwise, embedding)`, and the
clipwise output is **already sigmoid-activated** — one probability per class,
independent, summing to nothing in particular. Thresholding is this pipeline's
to choose, which is what `event_min_score` is.

**PANNs downloads its checkpoint with a shelled-out `wget`** when
`checkpoint_path=None`, which **does nothing at all where wget is missing** and
then fails obscurely in `torch.load`. Pre-place `Cnn14_mAP=0.431.pth` and set
`checkpoint` in the config:

```bash
mkdir -p ~/panns_data && cd ~/panns_data
curl -LO 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1' \
  && mv 'Cnn14_mAP%3D0.431.pth?download=1' 'Cnn14_mAP=0.431.pth'
```

### The labels are matched by slug, and two are easy to get wrong

Labels go through `vocabulary.slug` (lowercase, non-word runs to hyphens) before
being looked up, because the raw strings are not usable as keys: 75 of AudioSet's
527 labels contain a comma, and emotion2vec's are bilingual with a slash
(`生气/angry`).

Two entries were wrong in the first draft and are worth knowing about, because
both would have silently produced no tag:

- emotion2vec's neutral class is **中立**, not 中性.
- Its ninth class is the literal string **`<unk>`**, not `unknown`. The model
  card's prose calls it "unknown"; the runtime never says that.

### What to verify on the first run

1. **That the delivery model produces tags at all.** Its outputs are free text;
   run one segment and print the raw string before it goes through the table. If
   nothing maps, the vocabulary's spellings need extending, and
   `summary.json`'s `unmapped` count is where that shows up.
2. **That the three thresholds are where you want them.** They are starting
   points, not measurements. `emotion_min_score` gates a nine-way softmax and
   the other two gate sigmoids — different scales, so tune them separately.
3. **That the extraction is good enough to tag.** This stage reads S7's output,
   not the mix, so a bad separation shows up here as tags that describe the
   wrong person. Compare a tagged segment against the mix when something looks
   wrong.

### Licences, as found rather than as assumed

| component | licence | note |
| --- | --- | --- |
| emotion2vec+ large | Apache-2.0 **declared on ModelScope** | the HF mirror says `other` and the upstream repo has no LICENSE file; the two mirrors disagree |
| laion/voice-tagging-whisper | Apache-2.0 declared | its base model `laion/BUD-E-Whisper` is CC-BY-4.0, so attribution is owed down the chain |
| PANNs code | MIT | the repo, the GitHub API and the wheel metadata all agree |
| PANNs checkpoint | **CC-BY-4.0** | Zenodo record 3987831 — attribution required for the `.pth`, separately from the code |

All permissive and all fine for the non-commercial research use this project was
built for. The emotion2vec entry is the one to re-check if the licence ever
needs to be stated precisely, because no source reconciles the discrepancy.

## Captioning (S10)

```bash
pip install 'transformers>=4.57' torch
```

The checkpoint is fetched from Hugging Face on first use. `transformers` must be
**4.57 or newer** — `qwen3_vl` does not exist before it. Nothing else in this
stage touches the audio chain; it reads shots and tracks and writes text.

### The memory arithmetic does not leave an easy answer

Weight sizes published by Qwen, not estimated:

| checkpoint | size | loads in transformers? |
| --- | --- | --- |
| `Qwen3-VL-30B-A3B-Instruct` (bf16) | **62.1 GB** | yes, but not on one 48 GB card |
| `Qwen3-VL-30B-A3B-Instruct-FP8` | **32.3 GB** | **no** — the card says transformers cannot load it |
| `Qwen3-VL-30B-A3B-Instruct-GGUF` (Q4_K_M) | **18.6 GB** | no — that is a llama.cpp format |
| `Qwen3-VL-8B-Instruct` (bf16) | ~16 GB | yes, comfortably |

**The FP8 checkpoint is the trap.** It is the obvious fit — 32 GB on a 48 GB
card, described by Qwen as nearly identical to bf16 — and its own model card
says transformers cannot load those weights, directing you to vLLM or SGLang
instead. There is no AWQ or GPTQ release for Qwen3-VL at all.

So a single 48 GB card running this stage's adapter has two honest options:

- **an 8B dense checkpoint** (`Qwen/Qwen3-VL-8B-Instruct`), which fits and works
  with no extra machinery — the right default for a first batch;
- **the 30B-A3B with `device_map="auto"`**, which will offload to host RAM and
  run slowly. It works, and it is worth measuring before committing a corpus to
  it.

If the 30B's quality is needed at speed, that means serving it through vLLM and
changing this adapter, which is a bigger decision than a config change.

The stage's own footprint is negligible next to any of this: a handful of stills
at 448 px, per request.

### What to verify on the first run

1. **That the frames actually arrived.** Set `max_new_tokens` low and run one
   video, then read `captions.json`. A caption describing a generic room rather
   than the one on screen means the images were dropped rather than erroring,
   which is the failure this interface is most likely to have.
2. **The image token budget.** `max_edge` (448 by default) is the lever: Qwen3-VL
   spends one token per 32×32 pixels, so a 448 px frame is about 196 tokens and
   eight of them about 1,600 per request. The checkpoint's own default allows
   16,384 tokens *per image*, so leaving it in charge is expensive — the frame
   size is what actually governs the cost here.
3. **Whether the model obeys the identifier rule.** `summary.json`'s `dropped`
   count is how often it did not. A handful is expected; a large number means
   the prompt needs strengthening, not that the check is wrong.
4. **`device_map` versus `device`.** They are alternatives — setting both means
   `device_map` wins. `device_map="auto"` is what you want for a checkpoint that
   does not fit on one card.

Do **not** port `enable_thinking=False` from Qwen3 or Qwen2.5-VL code: Qwen3-VL
has no such variable, and it is silently ignored with a warning. Which behaviour
you get is decided by downloading `-Instruct` or `-Thinking`, and this stage
wants `-Instruct`. `add_vision_id=True` is a real variable that labels each image
("Picture 1: ...") and may help with many frames, but it also invites
frame-by-frame description, which is the opposite of what a shot caption wants.

## What has actually been run, and where

Most of this document is unverified interface. This is the part that is not. On
an Apple M4 laptop (16 GB, no CUDA), against the three sample clips:

| stage | status | note |
| --- | --- | --- |
| S0, S1, S2, S3 | **run** | insightface on CPU; 120 frames → 240 detections → 2 tracklets → 2 people |
| S8 | **run** | faster-whisper `base`, CPU int8; 7.6 s for a 12 s clip |
| S9 (event) | **run** | PANNs CNN14, CPU; 2.0 s including model load |
| S11 | **run** | all five gates passed on real output |
| S4, S5, S7, S9 (emotion, delivery), S10 | **not run** | need a GPU, a repo clone, or more RAM than this machine has |

Four specific things that run confirmed, which reading could not:

- **`"".join(word.word)` reproduces `segment.text` exactly**, leading spaces and
  all. That is the join `avannotate.asr.text` performs, so the multilingual
  spacing rule is real rather than inferred.
- **A numpy array is used as-is at 16 kHz.** `info.duration` matched the sample
  count exactly; nothing resampled.
- **PANNs' `clipwise` output is already sigmoid** — observed range [0, 0.84] on a
  speech segment, one independent probability per class, no softmax. The
  threshold is yours to pick, which is what `event_min_score` is.
- **`IGNORED_LABELS` earns its place.** On real speech the confident labels were
  `Speech` (0.84) and `Music` (0.71). Both are on the ignore list, so the
  `unmapped` report came back empty; without it every segment in a corpus would
  report those two and the signal would be gone.

### Two things that look like problems and are not

**numpy prints `divide by zero` / `overflow` / `invalid value encountered in
matmul` from inside faster-whisper.** These come from the same numpy-2 + Apple
Accelerate interaction already worked around in `faces/kalman.py`. The mel
output was checked directly: shape (80, 1197), no NaN, no inf, values in
[-0.91, 1.09]. It is cosmetic, it appears only on Apple silicon, and it is not a
reason to distrust a transcript.

**PANNs writes `class_labels_indices.csv` into `~/panns_data/` on import**,
regardless of the `checkpoint` path you pass. On a machine with no home
directory (a container running as `nobody`) that import will fail, and the fix
is to pre-place both files.

### Cost, measured on CPU

A 12-second clip, on a laptop: S1 is 34 s and dominates (a GPU is far faster),
S2 and S3 are under a second each, S8 is 8 s with `base`, S9's event dimension
is 2 s. Use these only to sanity-check that the GPU machine is faster than a
laptop — the numbers that decide sharding have to come from the server.

## Running the pipeline

`scripts/run_corpus.sh` runs one stage over the whole corpus, then the next:

```bash
scripts/run_corpus.sh --data /path/to/videos --list names.txt --output /results
```

Stage-major rather than video-major, because of the models: every stage builds
its own inside `run()`, `run()` is called once per video, and a video-major
corpus of a few thousand videos therefore builds the captioner a few thousand
times. Stage-major plus the per-process cache in `avannotate.model_cache` makes
that one load per stage per worker. `run_batch.sh` is the video-major one, and is
right for a handful of videos and the wrong thing for a thousand.

```bash
# resolve the list and report the plan; loads nothing
scripts/run_corpus.sh --data ... --list ... --output ... --dry-run

# a line per video per stage, instead of the progress bar
scripts/run_corpus.sh ... --verbose

# start at a stage, skipping the check on the ones before it
scripts/run_corpus.sh ... --from s7-tse
```

`--workers` defaults to one per card for a stage that uses one and
`min(16, cores)` for a stage that does not, and the README explains when to move
it in either direction.

S11 needs nothing installed — it runs no model, reads JSON and writes text. It
is also the stage to run on its own after any change to `annotation.py` or
`compose/`: re-rendering a corpus costs seconds, where re-running anything that
touches a GPU costs days.

**An entry in the list need not name a file.** These lists are usually dataset
indices that name a clip by its id and leave the extension to disk, so
`part_001/43/be/43bec54b…` is resolved by trying that name and then that name
with each media extension on it. One match or none; two candidates is an error
rather than a guess.

**S7 needs its working directory to be writable.** ClearerVoice reads its
checkpoint from `checkpoints/` *relative to the process's working directory*,
with no argument to override it, and `run_batch.sh` runs from the data root
because that is what the list's entries are relative to. So the checkpoint is
linked into the data directory — which on a shared cluster mount may be
read-only. There is a check for it that runs before anything else, because the
alternative is discovering it at S7 with every stage before it already done.

## What running it for real cost

One corpus: 8,716 clips, 8 GPUs, 128 cores, one stage at a time. These are the
numbers that were worth having, and every one of them was a surprise.

| what | what it was |
| --- | --- |
| S0 over 8,716 clips | about an hour. ffprobe, ffmpeg and PySceneDetect; no card involved |
| S1 at four workers | **3%** of a card, and slow. 20% after the thread fix, card still idle |
| S4 at four workers per card | each worker's usable batch was a quarter of what one per card leaves |
| S10, unconfigured | on the CPU, ETA **24 hours** |
| Log volume | the libraries produced more output than the pipeline did |

The S4 row is the one figure here that was not measured: it is what the
mechanism below implies, and `batch_size_used` is now recorded so that the next
run answers the question instead of implying the answer.

Every one of them had the same shape: **it worked, it was correct, and it was
silently several times slower than it should have been.** Nothing raised. The
six below are what came out of it, and most of them are properties of the
libraries rather than of this code — a new stage that calls a new model meets
them again.

**No thread count was set anywhere.** Every worker spawned its own OpenCV pool,
OpenMP pool, torch pool and onnxruntime pool, each defaulting to one thread per
core — sixteen workers on a 128-core machine asking for two thousand threads to
run sixteen videos. They do not sleep while they wait, they spin, and the shape
that produces is a worker burning seven cores while the card it feeds sits at
3%. `avannotate.threads` now divides the machine among the workers instead:
`cores // workers`, set in the worker initialiser because those libraries read
it once, at import, and a pool that already exists ignores the environment.

**onnxruntime falls back to the CPU without failing.** S1's models would have
run ten times slower with the card idle, and nothing in the output would have
said so. The doctor now checks for `CUDAExecutionProvider` and says it in those
words; `run_corpus.sh` runs the doctor before it starts.

**`device: null` meant the CPU, to one stage only.** Everywhere else it means
"pick one" — S8 hands `"auto"` to faster-whisper, S7 and S9 leave it to their
libraries. S10 took it literally: no `device_map` passed and no `.to()`
afterwards, so `from_pretrained` loaded an 8B vision model to the CPU and left
it there. It now means what it means everywhere else.

**A batch size that halves is worth a line.** S4's checkpoint asks for 32, which
does not fit a 24 GB card, so the adapter halves on every CUDA out-of-memory and
keeps what fitted. The message saying so went to stdout, which in a worker is a
block-buffered pipe, and the pool is terminated rather than joined — so an
unflushed buffer was discarded. A run where every worker had halved its way down
to one chunk at a time looked exactly like a run where nothing was wrong. It
goes to stderr, flushed, now, and the stage records `batch_size_used`.

**Two of the model libraries narrate.** ClearerVoice announces the model and
says something per file, and shells out to ffmpeg for the audio, whose output
inherits the process's. The taggers draw tqdm bars, print a timing dict per
call, and let `transformers` write its warnings. Both are called once per file,
so a corpus produced more of their output than of ours — over the top of the
progress bar, which is the stage's own report. `avannotate.quiet` redirects the
descriptors around the calls and keeps what was said for the error message.

**One input shape mangles the picture silently.** `probe_media` reads the coded
dimensions; ffmpeg applies the display matrix by default. For a clip carrying a
rotation they disagree, and because the pixel count of `w×h` and `h×w` is the
same, every read still lines up and nothing raises — the frame comes back
sideways and the detector finds no faces in it. All four decoders now pass
`scale=w:h`, which pins the output to the size the probe reported. A clip with a
rotation is still decoded sideways rather than upright; fixing that properly
means probing the display matrix and carrying the orientation through, and it is
not done.

## Throughput

Measured on a laptop, on CPU, over 26 seconds of video across three clips:

| stage | time | note |
| --- | --- | --- |
| S0 | ~2s | ffmpeg |
| S1 (insightface, stride 3) | 63s | ~2.4x realtime; a GPU is far faster |
| S2 | <1s | pure Python |
| S3 | <1s | pure Python |
| S4 … S10 | not measured here | need a GPU, a repo clone, or more RAM than this machine has |

S1 dominates and is what to profile first on real hardware.
`embedding_interval_seconds` changes only disk, not runtime: insightface
computes the vector as part of its own pipeline whether or not it is written.
`allowed_modules=["detection", "recognition"]` skips the two bundled models this
pipeline never uses and cut that measurement by about 22%.
