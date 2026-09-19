"""The active speaker network behind an interface, with LoCoNet as the choice.

The plan's choice is LoCoNet: 95.2% mAP on AVA and, more to the point here,
68.4% on Ego4D against the challenge winner's 60.7% -- it holds up outside the
Hollywood footage AVA is drawn from, and it beats EASEE on small faces, which a
822x462 clip full of them needs.

The interface is deliberately small -- crops and features in, probabilities out.
Two things in this file could not be checked from this machine and both are
isolated so they can be corrected in one place:

1. **The audio frontend.**  LoCoNet's loader wants ``[4T, 128]``: four feature
   frames per video frame, 128 bins.  That is exactly what a log-mel at 16 kHz
   with a 10 ms hop gives at 25 fps, which is where :func:`log_mel`'s parameters
   come from -- derived from the shape, not read off TalkNet's source.
2. **The crop margin.**  See :mod:`avannotate.asd.crop`.

Both are domain shifts applied to every frame if they are wrong, so both are
worth measuring against the checkpoint's own preprocessing before a full corpus
run.  Their outputs are ordinary arrays, so a corrected version can be checked
against a saved sample without re-running anything else.
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

#: Feature frames per video frame, from the loader shape ``[4T, 128]``.
_AUDIO_FRAMES_PER_VIDEO_FRAME = 4

#: Mel bins, from the same shape.
_MEL_BINS = 128

#: 16 kHz is the pipeline's audio standard and what the features are for.
_SAMPLE_RATE = 16_000


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
        """``[S, T, H, W]`` crops and ``[4T, 128]`` features to ``[S, T]``.

        The returned probabilities are per speaker and per frame; the caller
        keeps only the row for the target it asked about.
        """
        ...


def log_mel(
    samples: NDArray[np.float32],
    *,
    sample_rate: int = _SAMPLE_RATE,
    n_mels: int = _MEL_BINS,
    hop_length: int | None = None,
) -> NDArray[np.float32]:
    """Log-mel features shaped as the loader expects.

    ``hop_length`` defaults to whatever puts four feature frames under each
    video frame at 25 fps -- 160 samples at 16 kHz -- because that relationship
    is what the ``[4T, 128]`` shape encodes.  Pass it explicitly if a different
    output frame rate is wanted; the model will not agree, but a caller
    comparing frontends needs to be able to vary it.
    """

    try:
        import torch
        import torchaudio
    except ModuleNotFoundError as error:
        raise AsdError(
            "torch and torchaudio are required for the audio frontend: "
            "pip install torch torchaudio"
        ) from error

    if hop_length is None:
        hop_length = int(round(sample_rate / (_ASSUMED_VIDEO_FPS * _AUDIO_FRAMES_PER_VIDEO_FRAME)))

    waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=hop_length * 2,
        win_length=hop_length * 2,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel = transform(waveform)
    features = torch.log(mel.clamp(min=1e-6)).transpose(0, 1).contiguous()
    return np.asarray(features.numpy(), dtype=np.float32)


def features_for_window(
    samples: NDArray[np.float32], *, video_frames: int
) -> NDArray[np.float32]:
    """Log-mel trimmed to exactly four feature frames per video frame.

    The trim is not cosmetic.  ``MelSpectrogram`` centres its windows by
    default, so it returns ``len // hop + 1`` frames -- one more than the
    ``4T`` the loader expects -- and the model would reject the batch, or worse,
    silently misalign the audio against the video by one feature frame.
    """

    if video_frames <= 0:
        raise ValueError(f"video_frames must be positive, got {video_frames}")

    features = log_mel(samples)
    needed = video_frames * _AUDIO_FRAMES_PER_VIDEO_FRAME
    if len(features) >= needed:
        return features[:needed]
    # A short tail -- the end of the file -- is padded with silence rather than
    # dropped, so the batch keeps its shape.
    pad = np.zeros((needed - len(features), features.shape[1]), dtype=np.float32)
    return np.asarray(np.concatenate([features, pad], axis=0), dtype=np.float32)


class LoCoNetAsd:
    """LoCoNet behind :class:`AsdModel`.

    The checkpoint and the module that builds the network come from the
    official repository (``SJTUwxz/LoCoNet_ASD``), which is not a package and
    cannot be installed -- the caller supplies the path to a checkout, and this
    adapter imports from it.
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
                "Google Drive link; download them once and point --config at the file."
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

        try:
            from loconet import LoCoNet  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise AsdError(
                "could not import LoCoNet. Pass the repository checkout as 'repo' so "
                "its loconet.py is importable -- the project is not a package."
            ) from error

        self._model = LoCoNet()
        state = torch.load(str(checkpoint_path), map_location=self._resolve_device())
        # Checkpoints from training wrappers nest the weights; accept both so a
        # raw state dict and a saved module both load.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._model.load_state_dict(state)
        self._model.to(self._resolve_device())
        self._model.eval()

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
        expected = (len(crops), crops.shape[1] if crops.ndim > 1 else 0)
        if crops.ndim != 4:
            raise AsdError(f"crops must be [S, T, H, W], got shape {crops.shape}")
        if crops.shape[1] > self.max_window_frames:
            raise AsdError(
                f"window of {crops.shape[1]} frames exceeds the {self.max_window_frames} "
                "the model holds in memory; the window planner should have split it"
            )
        if audio.shape != (crops.shape[1] * _AUDIO_FRAMES_PER_VIDEO_FRAME, _MEL_BINS):
            raise AsdError(
                f"audio features must be [{crops.shape[1] * _AUDIO_FRAMES_PER_VIDEO_FRAME}, "
                f"{_MEL_BINS}] to match {expected[1]} video frames, got {audio.shape}"
            )

        video = torch.from_numpy(crops).unsqueeze(0).to(self._resolve_device())
        waveform = torch.from_numpy(audio).unsqueeze(0).to(self._resolve_device())
        with torch.no_grad():
            logits = self._model(waveform, video)

        # The head is trained with a binary objective, so a sigmoid is the
        # probability the caller wants.  Squeezed back to [S, T] and moved to
        # the host: everything downstream is numpy.
        probabilities = torch.sigmoid(logits).squeeze(0)
        return np.asarray(probabilities.detach().cpu().numpy(), dtype=np.float32)


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
