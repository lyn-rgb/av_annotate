"""The active speaker network behind an interface, with LoCoNet as the choice.

The plan's choice is LoCoNet: 95.2% mAP on AVA and, more to the point here,
68.4% on Ego4D against the challenge winner's 60.7% -- it holds up outside the
Hollywood footage AVA is drawn from, and it beats EASEE on small faces, which a
822x462 clip full of them needs.

The interface is deliberately small -- crops and features in, probabilities out.

Every number in the frontend below was read off the LoCoNet repository's source
rather than inferred, and four of them were wrong while they were inferred:

1. **The crop margin is zero.**  ``cropScale = 0.40`` is TalkNet's and LoCoNet
   does not inherit it -- that string does not appear anywhere in the
   repository.  See :mod:`avannotate.asd.crop`.
2. **The audio is ``[4T, 64]``.**  Sixty-four mel bands, because that is
   VGGish's band count and the width of the audio frontend's first convolution.
   The 128 that appears nearby is that frontend's *output* width, and reading a
   shape off the wrong layer is how this was wrong first.
3. **The crops are 0..255.**  LoCoNet normalises inside its own visual frontend,
   ``(x / 255 - 0.4161) / 0.1688``, so scaling beforehand applies the shift
   twice -- silently, and to every frame of every video.
4. **The audio features are VGGish's, not a generic log-mel.**  Nothing in the
   repository calls ``torchaudio``.  Every dataloader reaches the audio through
   ``torchvggish.vggish_input.waveform_to_examples``, and that fixes five
   constants this module used to get wrong: a 25 ms window and so a 512-point
   FFT (not 20 ms and 320), mel edges at 125 and 7500 Hz (not 0 and 8000), a
   *magnitude* spectrogram (not power), a periodic Hann rather than torchaudio's
   symmetric default, and ``log(mel + 0.01)`` rather than ``log(mel.clamp(1e-6))``.

   This one is worth dwelling on, because *the band count was right*.  The
   tensor was ``[4T, 64]`` and the model accepted it, so every shape check in
   this file passed while the values underneath were a different transform
   altogether.  A frontend is a convention, not a shape, and only one of those
   two was ever checked.  See :func:`log_mel`.

**The repository is usable, narrowly.**  Its ``loconet.py`` and ``train.py``
cannot be imported -- both need an ``xxlib`` module the repository does not
contain -- but that is the training harness, and neither is needed to score.
The model files import cleanly and were built and run to verify this adapter:
``model/loconet_encoder.py`` plus ``loss_multi.py`` and the in-tree
``torchvggish``.  Its own inference path is still unusable as a library, because
it wants ``dlhammer`` and an AVA data tree and its ``forward`` is a training
step that cannot be called without labels -- so the frontends are driven
directly here, which is what the repository's ``evaluate_network`` does
internally too.

Two things the constructor needs that are easy to miss: building it downloads
**275 MB of VGGish weights** from GitHub releases, and that download happens
whether or not it is useful -- which it is not, because the LoCoNet checkpoint
overwrites those weights immediately.  See ``docs/server-setup.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

#: Frames per forward pass before the model runs out of memory.  From LoCoNet's
#: own ablation, which puts 200 at the accuracy/memory balance and 400 over it.
LOCO_NET_MAX_FRAMES = 200

#: Faces per pass including the target, from AVA's 99th percentile.
LOCO_NET_MAX_SPEAKERS = 3

#: Video frame rate the audio features are derived against.
_ASSUMED_VIDEO_FPS = 25.0

#: Feature frames per video frame, from the loader's ``audio_t == video_t * 4``.
_AUDIO_FRAMES_PER_VIDEO_FRAME = 4

#: Mel bands, from VGGish's own ``NUM_MEL_BINS``.
#:
#: The mel-band count is VGGish's, and VGGish's is 64 -- not the 128 that the
#: *output* of the audio frontend happens to be wide.  Reading the shape off the
#: model's later layer rather than its input is how this was wrong first: 64 is
#: what ``vggish_params.NUM_MEL_BINS`` says, and feeding 128 bands to a frontend
#: whose first convolution is 64 wide does not fail, it convolves over nonsense.
_MEL_BINS = 64

#: 16 kHz is the pipeline's audio standard and what the features are for.
_SAMPLE_RATE = 16_000

#: VGGish's feature constants, from the checkout's
#: ``torchvggish/vggish_params.py``.  They are not defaults anywhere in
#: ``torchaudio``, and every one of them changes the features.
_VGGISH_WINDOW_SECONDS = 0.025
_VGGISH_HOP_SECONDS = 0.010
_VGGISH_MEL_MIN_HZ = 125.0
_VGGISH_MEL_MAX_HZ = 7500.0
_VGGISH_LOG_OFFSET = 0.01

#: Amplitude scale of the samples the features are computed from.
#:
#: The dataloaders read their WAVs with ``scipy.io.wavfile.read``, which returns
#: **int16** for this corpus's 16-bit PCM -- not the ``[-1, 1]`` floats that
#: ``vggish_input``'s docstring invites.  The features are then a logarithm, so
#: the scale does not divide out: at the loud end the two conventions differ by
#: ``ln(32768) = 10.4`` and at the quiet end by almost nothing, because the
#: ``+ 0.01`` offset dominates there.  That is a change in dynamic range, not a
#: constant gain, and a constant gain is the only thing the first convolution's
#: bias could have absorbed.
_VGGISH_SAMPLE_SCALE = 32768.0

#: HTK's mel scale, the one VGGish uses.
_MEL_BREAK_FREQUENCY_HZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


class AsdError(RuntimeError):
    """The model could not be built or run."""


class AsdModel(Protocol):
    """What the stage needs from a model, and nothing more."""

    name: str
    max_window_frames: int
    max_speakers: int

    def score(
        self, crops: NDArray[np.float32], audio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """``[S, T, H, W]`` crops and ``[4T, 64]`` features to ``[S, T]``.

        The returned probabilities are per speaker and per frame; the caller
        keeps only the row for the target it asked about, which is row 0 --
        the ordering every caller here depends on, and the reason padding a
        narrow group at the end is safe.
        """
        ...


def _periodic_hann(window_length: int) -> NDArray[np.float64]:
    """VGGish's Hann window: one full period of a length-*N* cosine.

    Not ``np.hanning``, which is the *symmetric* window -- a period-``N-1``
    cosine whose first and last samples both land on zero.  The two differ by
    exactly one sample of phase and VGGish is explicit about wanting this one.
    """

    return 0.5 - 0.5 * np.cos(2.0 * np.pi / window_length * np.arange(window_length))


def _hertz_to_mel(frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
    """HTK's mel scale.  Slaney's is the other one, and torchaudio's default."""

    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + frequencies / _MEL_BREAK_FREQUENCY_HZ)


def _mel_matrix(
    *,
    n_mels: int,
    num_spectrogram_bins: int,
    sample_rate: int,
    lower_edge_hz: float,
    upper_edge_hz: float,
) -> NDArray[np.float64]:
    """VGGish's triangular filterbank, as a ``[bins, n_mels]`` post-multiply.

    Transcribed rather than called: the checkout's own
    :func:`~torchvggish.mel_features.spectrogram_to_mel_matrix` would be the
    sturdier thing to lean on, but importing it drags in ``resampy`` and
    ``soundfile`` through the package's ``vggish_input`` and needs the LoCoNet
    checkout on ``sys.path`` -- which is configured per-stage and may not be
    there.  :mod:`tests.test_asd` pins this against that function when the
    checkout is present, which is the part that actually keeps them equal.
    """

    nyquist = sample_rate / 2.0
    if not 0.0 <= lower_edge_hz < upper_edge_hz <= nyquist:
        raise ValueError(
            f"mel edges {lower_edge_hz}..{upper_edge_hz} Hz are not inside 0..{nyquist}"
        )

    bins_hz = np.linspace(0.0, nyquist, num_spectrogram_bins)
    bins_mel = _hertz_to_mel(bins_hz)
    # One edge per band boundary, so the bands need two more edges than they
    # have interiors: edge i, centre i, edge i+1 for band i.
    edges = np.linspace(
        _hertz_to_mel(lower_edge_hz), _hertz_to_mel(upper_edge_hz), n_mels + 2
    )

    weights = np.empty((num_spectrogram_bins, n_mels), dtype=np.float64)
    for index in range(n_mels):
        lower, centre, upper = edges[index : index + 3]
        # Slopes are linear in mel, not in hertz -- that is the whole point of
        # the scale, and doing it in hertz is a filterbank that looks right and
        # is not.
        rising = (bins_mel - lower) / (centre - lower)
        falling = (upper - bins_mel) / (upper - centre)
        weights[:, index] = np.maximum(0.0, np.minimum(rising, falling))
    # The DC bin is excluded by HTK, explicitly.
    weights[0, :] = 0.0
    return weights


def _stft_magnitude(
    signal: NDArray[np.float64], *, fft_length: int, hop_length: int, window_length: int
) -> NDArray[np.float64]:
    """VGGish's framing and FFT: no padding, no centring, magnitude only.

    ``fft_length`` is deliberately larger than ``window_length`` -- 512 against
    400 -- so the spectrum is zero-padded to the next power of two, which is
    what VGGish does and what a 320-point no-padding FFT did not.
    """

    num_frames = 1 + int(np.floor((len(signal) - window_length) / hop_length))
    if num_frames <= 0:
        return np.zeros((0, fft_length // 2 + 1), dtype=np.float64)
    # A view, not a copy: an hour of 16 kHz audio is 230 MB and framing it by
    # copying would be most of a gigabyte for no reason.  VGGish does the same
    # with stride tricks.
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, window_length),
        strides=(signal.strides[0] * hop_length, signal.strides[0]),
    )
    return np.abs(np.fft.rfft(frames * _periodic_hann(window_length), int(fft_length), axis=1))


def log_mel(
    samples: NDArray[np.float32],
    *,
    sample_rate: int = _SAMPLE_RATE,
    n_mels: int = _MEL_BINS,
    fps: float = _ASSUMED_VIDEO_FPS,
) -> NDArray[np.float32]:
    """The log-mel features LoCoNet was trained on, i.e. VGGish's.

    ``fps`` is not decoration.  The repository scales both the analysis window
    and the hop by ``25 / fps``, which keeps four feature frames under every
    video frame whatever the video's rate is; at 25 fps that is the textbook
    VGGish 25 ms / 10 ms pair and the two agree, but a 30 fps video would get a
    400-sample window and a 160-sample hop here against the reference's 333 and
    133.  Passing the timeline's fps is what keeps the two from drifting apart
    on a corpus that is not all 25 fps.

    ``samples`` are the pipeline's ``[-1, 1]`` floats and are rescaled to the
    int16 range internally -- see :data:`_VGGISH_SAMPLE_SCALE` for why that is
    not optional.
    """

    window_seconds = _VGGISH_WINDOW_SECONDS * _ASSUMED_VIDEO_FPS / fps
    hop_seconds = _VGGISH_HOP_SECONDS * _ASSUMED_VIDEO_FPS / fps
    window_length = int(round(sample_rate * window_seconds))
    hop_length = int(round(sample_rate * hop_seconds))
    if window_length <= 0 or hop_length <= 0:
        raise ValueError(f"fps {fps} gives a degenerate window or hop")
    # The next power of two at or above the window, as VGGish computes it.
    fft_length = 2 ** int(np.ceil(np.log2(window_length)))

    signal = np.asarray(samples, dtype=np.float64) * _VGGISH_SAMPLE_SCALE
    spectrogram = _stft_magnitude(
        signal, fft_length=fft_length, hop_length=hop_length, window_length=window_length
    )
    matrix = _mel_matrix(
        n_mels=n_mels,
        num_spectrogram_bins=spectrogram.shape[1],
        sample_rate=sample_rate,
        lower_edge_hz=_VGGISH_MEL_MIN_HZ,
        upper_edge_hz=_VGGISH_MEL_MAX_HZ,
    )
    # The BLAS under this matmul leaks floating-point status flags on some
    # builds: an all-ones ``(98, 257) @ (257, 64)`` -- which cannot divide by
    # zero or overflow -- raises all three of numpy's warnings on the machine
    # this was written on, and the real output is finite throughout.  Silenced
    # rather than left to flood a run's log, because this is once per window per
    # track and a corpus is thousands of them.
    with np.errstate(all="ignore"):
        mel = spectrogram @ matrix
    return np.asarray(np.log(mel + _VGGISH_LOG_OFFSET), dtype=np.float32)


def features_for_window(
    samples: NDArray[np.float32], *, video_frames: int, fps: float = _ASSUMED_VIDEO_FPS
) -> NDArray[np.float32]:
    """Log-mel trimmed to exactly four feature frames per video frame.

    The trim is not cosmetic: the loader asserts ``audio_t == video_t * 4``, and
    a batch off by a frame would reject -- or worse, silently misalign the audio
    against the video.

    VGGish's own framing runs short rather than long.  With no padding and no
    centring, a window of *T* video frames yields ``4T - 2`` feature frames, not
    ``4T``, so this pads rather than trims in the common case.  The repository
    pads the same two frames with ``np.pad(..., 'wrap')``, and so does this:
    silence would be a claim about those 20 ms that the audio did not make.
    """

    if video_frames <= 0:
        raise ValueError(f"video_frames must be positive, got {video_frames}")

    features = log_mel(samples, fps=fps)
    needed = video_frames * _AUDIO_FRAMES_PER_VIDEO_FRAME
    if len(features) >= needed:
        return features[:needed]
    if len(features) == 0:
        return np.zeros((needed, _MEL_BINS), dtype=np.float32)
    # Wrapped from the start of the window, matching the repository's pad.
    shortage = needed - len(features)
    wrapped = features[np.arange(shortage) % len(features)]
    return np.asarray(np.concatenate([features, wrapped], axis=0), dtype=np.float32)


#: What the released AVA checkpoint was trained with.  The speaker count is
#: **baked into the weight shapes**: ``convLayer`` builds
#: ``Conv2d(256, 256 * NUM_SPEAKERS, (NUM_SPEAKERS, 7))``, so a model
#: constructed with a different number cannot load these weights at all.
LOCO_NET_NUM_SPEAKERS = 3


#: The rest of the config the network reads, from ``configs/multi.yaml``.
LOCO_NET_AV_LAYERS = 3
LOCO_NET_ADJUST_ATTENTION = 0


def split_checkpoint(
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The released checkpoint into ``(encoder state, classifier head state)``.

    The file is a plain save of the training wrapper's ``state_dict``, so every
    key begins ``model.module.``.  What is left after stripping that is **two
    sibling groups**, not a nested one::

        model.module.model.visualFrontend.frontend3D.0.weight   -> encoder
        model.module.lossAV.FC.weight                           -> head
        model.module.lossA.FC.weight                            -> ignored
        model.module.lossV.FC.weight                            -> ignored

    The head is *not* under ``model.``, and assuming it was is how this was
    wrong first -- it stayed wrong through a test against a hand-built
    checkpoint, because that checkpoint had been written from the same wrong
    assumption.  The real file's 272 keys are what settled it, and they are what
    the test below is built from.

    ``lossA`` and ``lossV`` are the audio-only and visual-only heads, 128-d where
    the AV head is 256-d; nothing here uses them.
    """

    outer = "model.module."
    stripped = {
        (key[len(outer) :] if key.startswith(outer) else key): value
        for key, value in state.items()
    }
    encoder_state = {
        key[len("model.") :]: value
        for key, value in stripped.items()
        if key.startswith("model.")
    }
    head_state = {
        key[len("lossAV.") :]: value
        for key, value in stripped.items()
        if key.startswith("lossAV.")
    }
    return encoder_state, head_state


def pad_speakers(crops: NDArray[np.float32], *, width: int) -> NDArray[np.float32]:
    """Widen a group to the model's fixed speaker axis with black tiles.

    Fewer speakers than the model's axis is the common case -- a window with one
    person in it still has to be scored -- and black is not an arbitrary filler:
    the model normalises its own input, so a zero tile becomes the constant
    ``-2.47`` that the repository's own loader produces for a speaker with no
    face in frame.  That is what the network was trained to read as "nobody
    there", and a mid-grey tile would be a face-shaped lie.

    More than ``width`` is an error rather than a truncation: dropping a speaker
    would silently attribute whatever they did to somebody else.
    """

    speakers = crops.shape[0]
    if speakers > width:
        raise AsdError(
            f"LoCoNet's speaker axis holds {width} and this group has {speakers}. "
            "The planner is supposed to cap a group at that width; more than that "
            "has to be split into two passes rather than trimmed."
        )
    if speakers == width:
        return crops
    padding = np.zeros((width - speakers, *crops.shape[1:]), dtype=crops.dtype)
    return np.ascontiguousarray(np.concatenate([crops, padding], axis=0))


def _loconet_config() -> Any:
    """The three keys the encoder reads, in the shape it reads them.

    The repository's network takes a ``dlhammer`` config object and looks up
    ``cfg.MODEL.NUM_SPEAKERS``, ``cfg.MODEL.AV_layers`` and
    ``cfg.MODEL.ADJUST_ATTENTION``.  Building one by hand means this adapter
    does not need dlhammer, which is reachable only through the repository's
    own PYTHONPATH arrangement.
    """

    class _Section(dict):  # type: ignore[type-arg]
        __getattr__ = dict.__getitem__

    model = _Section()
    model["NUM_SPEAKERS"] = LOCO_NET_NUM_SPEAKERS
    model["AV_layers"] = LOCO_NET_AV_LAYERS
    model["ADJUST_ATTENTION"] = LOCO_NET_ADJUST_ATTENTION
    config = _Section()
    config["MODEL"] = model
    return config


class LoCoNetAsd:
    """LoCoNet behind :class:`AsdModel`.

    The checkpoint and the network come from the official repository
    (``SJTUwxz/LoCoNet_ASD``), which is not a package and cannot be installed.

    **It does not import as a whole, but the part that matters does.**
    ``loconet.py`` and ``train.py`` both open with ``from
    xxlib.utils.distributed import ...``, and ``xxlib`` exists nowhere in the
    repository -- a leftover from the authors' private harness.  Neither is
    imported here: this adapter needs ``model/loconet_encoder.py`` and
    ``loss_multi.py``, and those import cleanly apart from ``resampy`` and the
    in-tree ``torchvggish``.  ``loconet.py``'s only other use is the training
    wrapper, whose ``forward`` cannot be called to score anything anyway.

    Four things about that repository shape this adapter, each verified by
    building and running it rather than by reading alone.  The first three were
    wrong in the first draft, and each would have failed quietly:

    * **There is no inference entry point.**  ``Loconet.forward`` takes
      ``(audioFeature, visualFeature, labels, masks)`` and returns a loss tuple;
      both ``labels`` and ``masks`` are dereferenced before any branch, so it
      cannot be called to score anything.  The four frontends and the classifier
      head are driven separately here, which is what the repository's own
      ``evaluate_network`` does internally.
    * **The head is ``lossAV.FC``**, a ``Linear(256, 2)``, and its weights live
      in the checkpoint rather than in the encoder.  The repository scores with
      ``softmax(-1)[:, 1]``; there is no sigmoid anywhere in this model.
    * **The visual frontend normalises its own input** -- ``(x / 255 - 0.4161) /
      0.1688`` -- so crops arrive in 0..255 and must not be scaled first.

    And a fourth, which is a constraint rather than a correction: **the speaker
    axis is exactly three.**  ``ConvLayer`` convolves *across* speakers with a
    kernel as tall as ``NUM_SPEAKERS``, so a narrower group has to be padded --
    which this does with black tiles, the repository's own convention -- and a
    wider one cannot be scored in a single pass at all.
    """

    name = "loconet"
    max_window_frames = LOCO_NET_MAX_FRAMES
    max_speakers = LOCO_NET_MAX_SPEAKERS

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        repo: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise AsdError(
                f"LoCoNet checkpoint not found at {checkpoint_path}. The weights are "
                "distributed from the repository's README (SJTUwxz/LoCoNet_ASD) as a "
                "Google Drive link; download them once and point the config at the file."
            )

        if repo is not None:
            import sys

            repo_path = Path(repo).expanduser().resolve()
            if not repo_path.is_dir():
                raise AsdError(f"LoCoNet checkout not found at {repo_path}")
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))

        try:
            import torch
        except ModuleNotFoundError as error:
            raise AsdError("torch is required for the LoCoNet model") from error

        self._torch = torch
        self.checkpoint = str(checkpoint_path)
        self.device = self._place_on(device)
        #: How much of the checkpoint the encoder recognised.  Recorded rather
        #: than raised on: a wholesale mismatch means the weights do not belong
        #: to this network, and that shows up as this count being most of it.
        self.load_report: dict[str, int] = {}

        try:
            # ``locoencoder``, not ``Loconet`` and certainly not ``LoCoNet``:
            # neither of those capitalisations exists, and the wrapper that
            # shares the ``Loconet`` name exists only to compute training
            # losses.  The encoder is also the one that does not call ``.cuda()``
            # in its constructor, so it is usable on a CPU-only machine.
            from model.loconet_encoder import locoencoder  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise AsdError(
                "could not import LoCoNet's encoder. Its repository does not import "
                "as it stands -- loconet.py imports a module ('xxlib') the repository "
                "does not contain -- so either patch that line in the checkout or "
                "vendor the model files. See docs/server-setup.md."
            ) from error

        self._encoder = locoencoder(_loconet_config())
        self._head = self._build_head(checkpoint_path)
        self._encoder.to(self._resolve_device())
        self._encoder.eval()

    def _build_head(self, checkpoint_path: Path) -> Any:
        """Load the encoder's weights and lift the classifier head out.

        ``lossAV.FC`` belongs to the training wrapper, so its weights are saved
        under those names and have to be separated before the encoder's own
        state can be loaded.  The other two losses and any optimizer state are
        ignored.
        """

        torch = self._torch
        state = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise AsdError(
                f"{checkpoint_path} is not a state dict but a {type(state).__name__}. "
                "The Google Drive file is a plain torch.save of the wrapper's state."
            )

        encoder_state, head_state = split_checkpoint(state)
        if not head_state:
            raise AsdError(
                f"{checkpoint_path} holds no lossAV weights, and this adapter needs "
                "the classifier head -- without it the encoder's 256-d features mean "
                "nothing on their own. Keys seen: "
                + ", ".join(sorted(state)[:8])
            )

        try:
            from loss_multi import lossAV  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise AsdError(
                "could not import lossAV, which holds the classifier head. It lives "
                "in the repository's loss_multi.py, so 'repo' has to point at a "
                "directory containing it alongside the model files."
            ) from error

        head = lossAV()
        head.load_state_dict(head_state)
        head.to(self._resolve_device())
        head.eval()

        missing, unexpected = self._encoder.load_state_dict(encoder_state, strict=False)
        self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}
        return head

    def _resolve_device(self) -> Any:
        return self._torch.device(self.device)

    def _place_on(self, device: str | None) -> str:
        if device is None:
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        return device

    def score(
        self, crops: NDArray[np.float32], audio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        torch = self._torch
        if crops.ndim != 4:
            raise AsdError(f"crops must be [S, T, H, W], got shape {crops.shape}")
        speakers, frames = int(crops.shape[0]), int(crops.shape[1])
        if frames > self.max_window_frames:
            raise AsdError(
                f"window of {frames} frames exceeds the {self.max_window_frames} "
                "the model holds in memory; the window planner should have split it"
            )
        expected_audio = (frames * _AUDIO_FRAMES_PER_VIDEO_FRAME, _MEL_BINS)
        if audio.shape != expected_audio:
            raise AsdError(
                f"audio features must be {list(expected_audio)} to match {frames} "
                f"video frames, got {list(audio.shape)}"
            )

        if speakers > LOCO_NET_NUM_SPEAKERS:
            # The speaker axis is fixed by the weights, not by preference:
            # ``ConvLayer`` builds ``Conv2d(256, 256*s, (s, 7))`` with
            # ``s = cfg.MODEL.NUM_SPEAKERS`` and convolves *across* the speaker
            # axis, so more speakers than it was built for cannot be scored in
            # one pass at all.  The window planner caps groups at this width;
            # this is where that cap stops being an assumption.
            raise AsdError(
                f"LoCoNet's speaker axis holds {LOCO_NET_NUM_SPEAKERS} and this "
                f"window has {speakers}. The planner is supposed to cap a group at "
                "that width; more than that has to be split into two passes."
            )

        device = self._resolve_device()
        # How many rows the caller asked about, which is not the width the model
        # runs at.  The padding happens here and is dropped again before
        # returning, so the caller never sees it.
        requested = speakers
        crops = pad_speakers(crops, width=LOCO_NET_NUM_SPEAKERS)
        speakers = LOCO_NET_NUM_SPEAKERS

        # Four dimensions, not five: the visual frontend unpacks
        # ``B, T, W, H = x.shape`` itself and the caller is expected to have
        # already folded any batch dimension into the speaker one.
        visual = torch.from_numpy(crops).to(device)
        # The audio, by contrast, is a single-channel image: [b, 1, 4T, mel].
        audio_feature = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            audio_embed = self._encoder.forward_audio_frontend(audio_feature)
            visual_embed = self._encoder.forward_visual_frontend(visual)
            # The audio frontend runs once for the clip while the visual one runs
            # per speaker, so its embedding is repeated across the speaker axis.
            audio_embed = audio_embed.repeat(speakers, 1, 1)
            audio_embed, visual_embed = self._encoder.forward_cross_attention(
                audio_embed, visual_embed
            )
            # ``b`` and ``s`` are passed rather than left to the defaults: the
            # backend hands them to every other ConvLayer, which reshapes by
            # them.  The defaults are 1 and 1, so omitting them would silently
            # reshape a three-speaker batch as if it were one.
            combined = self._encoder.forward_audio_visual_backend(
                audio_embed, visual_embed, 1, speakers
            )
            # [S*T, 256] -> [S, T, 2] -> the probability of the speaking class.
            # The flat axis is ordered speaker-major, which is the order the
            # repository's own ``view(b, s, t, -1)`` assumes.
            logits = self._head.FC(combined).view(speakers, frames, 2)
            probabilities = torch.softmax(logits, dim=-1)[..., 1]

        # The padding rows are dropped rather than returned: they are an
        # artefact of the model's fixed width, and the caller asked about
        # ``requested`` speakers.  Dropping from the end is safe because the
        # target is row 0 -- which is the ordering the whole stage depends on.
        answer = probabilities[:requested]
        return np.asarray(answer.detach().cpu().numpy(), dtype=np.float32)


def build_asd_model(config: Mapping[str, Any]) -> AsdModel:
    """Construct the model a stage's config asks for."""

    backend = str(config.get("backend", "loconet"))
    if backend != "loconet":
        raise AsdError(f"unknown asd backend {backend!r}; the plan's choice is 'loconet'")

    checkpoint = config.get("checkpoint")
    if checkpoint is None:
        raise AsdError("the loconet backend needs a 'checkpoint' path to the weights")
    repo = config.get("repo")
    device = config.get("device")
    return LoCoNetAsd(
        checkpoint=str(checkpoint),
        repo=str(repo) if repo is not None else None,
        device=str(device) if device is not None else None,
    )
