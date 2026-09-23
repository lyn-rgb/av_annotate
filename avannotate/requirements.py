"""What each stage needs from the machine it runs on.

Declared in one place rather than scattered through six adapters, because the
question this answers is asked before a run and not during one: *will this
machine get through a batch?*  Discovering a missing package at video four
hundred, after a day of GPU time, is the failure this exists to prevent.

Three kinds of requirement, and they fail differently.

**An importable module.**  Most of the stack is a pip install.  Checked with
``importlib.util.find_spec``, which resolves the module without importing it --
importing is what the adapters do when they load a checkpoint, and a doctor that
loaded a 60 GB model to check it was present would be worse than no doctor.

**A checkout.**  Two of the models are research repositories rather than
packages, so the module only exists after a ``git clone``.  These also take a
config path, because where somebody put the checkout is not something this code
can guess.

**A weights file.**  A few checkpoints are downloaded by hand: LoCoNet's is on
Google Drive, PANNs' is on Zenodo, and neither is fetched by its package.  A
config key names the path, and that path has to exist.

Nothing here is a guarantee that a stage will *work* -- a model can be present
and still fail on a shape or a version.  It is a guarantee that the things which
are tedious to obtain are obtainable, and that a machine missing one says so
before the batch starts rather than four hundred videos in.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from avannotate.stages.base import load_config_file, resolve_config_path

#: Where the two repositories and the by-hand checkpoints come from.  Kept here
#: so the setup script and the doctor cannot disagree about a URL.
REPOSITORIES: Mapping[str, str] = {
    "diarizen": "https://github.com/BUTSpeechFIT/DiariZen",
    "loconet": "https://github.com/SJTUwxz/LoCoNet_ASD",
}

CHECKPOINTS: Mapping[str, str] = {
    "loconet": "loconet_AVA.model -- the Google Drive link in that repository's README",
    "panns": "Cnn14_mAP=0.431.pth -- https://zenodo.org/record/3987831",
}


@dataclass(frozen=True)
class Requirement:
    """One stage's dependencies, as facts rather than as a paragraph.

    A checkout and a weight file are kept apart because they fail differently:
    one is a directory somebody cloned to and the other is a file somebody
    downloaded, and telling an operator to "download" a checkout is the kind of
    instruction that wastes an afternoon.
    """

    stage: str
    #: Importable module names to look for.
    modules: tuple[str, ...] = ()
    #: What to install when a module is missing.
    install: str = ""
    #: A ``git clone`` this stage needs, keyed into :data:`REPOSITORIES`.
    repo: str | None = None
    #: The config key naming where that checkout lives, when the checkout is
    #: not installed as a package and has to be pointed at.
    repo_key: str | None = None
    #: The config file this stage is normally run with, under ``configs/``.
    #: The doctor loads it so the paths it checks are the paths the stage will
    #: resolve -- including their relativity to the config's own directory.
    config_file: str = ""
    #: Config keys naming weights files that must exist.
    weights: tuple[str, ...] = ()
    #: Where those weights come from.
    weights_hint: str = ""
    #: Shown when the stage is ready, so a reader can see *what* was found.
    note: str = ""
    #: Whether this stage's onnxruntime models have to reach a card to be worth
    #: running.  See :meth:`onnx_without_cuda` for why this is its own check.
    needs_onnx_cuda: bool = False

    def onnx_without_cuda(self) -> str | None:
        """onnxruntime is installed and will run these models on the CPU.

        The most expensive thing that can be quietly wrong here.
        ``onnxruntime`` and ``onnxruntime-gpu`` are the same import, the same
        ``InferenceSession`` and the same version string; what differs is
        whether ``CUDAExecutionProvider`` is among the ones they offer.  A
        session asked for ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
        does not fail when it is not there -- it takes the second and says
        nothing.  The stage then runs about ten times slower with a card idle,
        and nothing in its outputs distinguishes that from the normal case.

        Checked here or nowhere, then.  A warning rather than a failure: a
        machine with no card at all is a legitimate place to run this, and it is
        not the doctor's business to refuse it -- only to say what it costs.
        """

        if not self.needs_onnx_cuda:
            return None
        try:
            import onnxruntime
        except ModuleNotFoundError:
            # Reported as a missing module; one problem, said once.
            return None
        available = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" in available:
            return None
        version = getattr(onnxruntime, "__version__", "?")
        return (
            f"onnxruntime {version} offers {', '.join(sorted(available))} and no "
            "CUDAExecutionProvider: this stage's models run on the CPU, about ten "
            "times slower, with any card idle. onnxruntime-gpu needs a cuDNN "
            "matching the CUDA it was built for, and installing it without that "
            "gives an install that works and does not say so."
        )

    def _resolved(self, config: Mapping[str, object], key: str) -> Path | None:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        # Resolved the way the stage resolves it, or the doctor would report a
        # file missing that the run would have found.
        return resolve_config_path(value, config)

    def absent_weights(self, config: Mapping[str, object]) -> tuple[str, ...]:
        """Weight paths that are configured but not there, or not configured.

        An unset path counts as absent: for these models, "not configured" and
        "not downloaded" both mean the stage cannot run, and the operator needs
        to hear about both.
        """

        absent: list[str] = []
        for key in self.weights:
            path = self._resolved(config, key)
            if path is None:
                absent.append(f"{key} is not set")
            elif not path.is_file():
                absent.append(f"{key} -> {path} is not a file")
        return tuple(absent)

    def checkout_missing(self, config: Mapping[str, object]) -> str | None:
        """A checkout this stage needs but cannot find."""

        if self.repo_key is None:
            return None
        path = self._resolved(config, self.repo_key)
        if path is None:
            return f"{self.repo_key} is not set"
        if not path.is_dir():
            return f"{self.repo_key} -> {path} is not a directory"
        return None


#: Every stage that needs something beyond this repository and ffmpeg.
#:
#: S0-S3 need ffmpeg and, for S1, insightface -- which is checked here too
#: because a run that gets through S0 and dies at S1 has still wasted the pass
#: over the corpus.  S11 needs nothing: it reads JSON and writes text.
REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        stage="s1-faces",
        modules=("insightface", "onnxruntime"),
        install="pip install 'avannotate[faces]'",
        note="insightface for identity vectors; S3 refuses to run without them",
        # SCRFD detection and ArcFace embeddings, once per sampled frame of
        # every video in the corpus -- the stage where a CPU fallback costs
        # days rather than minutes.
        needs_onnx_cuda=True,
    ),
    Requirement(
        stage="s4-diarize",
        modules=("diarizen", "torch", "torchaudio"),
        install="git clone <DiariZen> && pip install -r requirements.txt && pip install -e .",
        repo="diarizen",
        note="DiariZen also vendors a modified pyannote-audio; see the server notes",
    ),
    Requirement(
        stage="s5-asd",
        modules=("torch", "cv2"),
        install="pip install 'avannotate[asd]'  then point 'repo' at the checkout",
        repo="loconet",
        repo_key="repo",
        config_file="s5.loconet.json",
        weights=("checkpoint",),
        weights_hint=CHECKPOINTS["loconet"],
        # A plain checkout really is enough -- verified by building the network
        # from one.  The repository does contain an unimportable file
        # (loconet.py wants a missing 'xxlib'), but it is the training harness
        # and nothing here imports it.
        note=(
            "checkout and checkpoint present; also needs resampy, and building the "
            "model downloads 275 MB of VGGish weights it then overwrites"
        ),
    ),
    Requirement(
        stage="s7-tse",
        modules=("clearvoice", "torch"),
        install="pip install 'avannotate[tse]'",
        note="the checkpoint is fetched from Hugging Face on first use",
    ),
    Requirement(
        stage="s8-asr",
        modules=("faster_whisper",),
        install="pip install 'avannotate[asr]'",
        note="CTranslate2 only -- no torch needed",
    ),
    Requirement(
        stage="s9-paralinguistic",
        modules=("funasr", "transformers", "torch", "panns_inference", "librosa"),
        install="pip install 'avannotate[paralinguistic]'",
        config_file="s9.paralinguistic.json",
        weights=("checkpoint",),
        weights_hint=CHECKPOINTS["panns"],
        note="one package per tagger dimension; point 'checkpoint' at PANNs' .pth",
    ),
    Requirement(
        stage="s10-caption",
        modules=("transformers", "torch", "PIL"),
        install="pip install 'avannotate[caption]'  # transformers>=4.57",
        note="check the checkpoint's weight size against the card's memory",
    ),
)


def requirement_for(stage: str) -> Requirement | None:
    for item in REQUIREMENTS:
        if item.stage == stage:
            return item
    return None


def missing_modules(modules: tuple[str, ...]) -> tuple[str, ...]:
    """Which of these cannot be imported here, without importing any of them."""

    absent: list[str] = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            # A parent package that is itself missing raises rather than
            # returning None, and either way the answer is the same.
            found = None
        if found is None:
            absent.append(name)
    return tuple(absent)


@dataclass(frozen=True)
class Status:
    """One stage's verdict, ready to print."""

    stage: str
    ready: bool
    detail: str = ""
    #: What a person has to do, when something is missing.
    todo: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "ready": self.ready,
            "detail": self.detail,
            "todo": list(self.todo),
        }


def load_stage_config(
    requirement: Requirement, *, configs_dir: Path, override: Mapping[str, object] | None
) -> Mapping[str, object]:
    """The config a stage would actually run with.

    ``override`` wins when given, so a caller can check a config that is not in
    the standard directory.  Otherwise the stage's own file under ``configs/``
    is loaded, and a missing one leaves the paths unset -- which is the truth,
    and is reported as such rather than skipped.
    """

    if override is not None:
        return override
    if not requirement.config_file:
        return {}
    candidate = Path(configs_dir) / requirement.config_file
    if not candidate.is_file():
        return {}
    return load_config_file(candidate)


def check(
    stages: Sequence[str],
    *,
    configs_dir: Path | str = "configs",
    config: Mapping[str, object] | None = None,
) -> tuple[Status, ...]:
    """What each stage needs that this machine does not have.

    The config matters because two of the paths are configured rather than
    discovered, and they are relative to the config file that names them.
    Checking against a config other than the one a run will use is the
    difference between checking this machine and checking a machine somebody
    described in a document.
    """

    statuses: list[Status] = []
    for stage in stages:
        requirement = requirement_for(stage)
        if requirement is None:
            statuses.append(Status(stage=stage, ready=True, detail="no model needed"))
            continue

        active = load_stage_config(
            requirement, configs_dir=Path(configs_dir), override=config
        )
        absent = missing_modules(requirement.modules)
        unset_weights = requirement.absent_weights(active)
        no_checkout = requirement.checkout_missing(active)

        todo: list[str] = []
        # The clone comes first, because for these two the install is a step
        # *inside* the checkout rather than an alternative to it.
        if requirement.repo and (absent or no_checkout):
            todo.append(f"clone: {REPOSITORIES[requirement.repo]}")
        if absent:
            todo.append(f"install: {requirement.install}")
        if no_checkout:
            todo.append(
                f"configure: set '{requirement.repo_key}' in the stage config to "
                "the checkout"
            )
        if unset_weights:
            todo.append(f"provide: {requirement.weights_hint or requirement.weights}")

        # Ready, but on the CPU.  Reported in the detail rather than as a
        # failure, so that the stage runs and the operator is told what it will
        # cost -- and said here because the detail is printed for a ready stage
        # too, which is the only place this would be seen before a corpus is
        # half-processed.
        slow = requirement.onnx_without_cuda()

        if not absent and not unset_weights and not no_checkout:
            detail = "; ".join(part for part in (requirement.note, slow) if part)
            statuses.append(Status(stage=stage, ready=True, detail=detail))
            continue

        parts: list[str] = []
        if absent:
            parts.append("missing module(s): " + ", ".join(absent))
        if no_checkout:
            parts.append(no_checkout)
        if unset_weights:
            parts.append("weights: " + "; ".join(unset_weights))
        if slow:
            parts.append(slow)
        statuses.append(
            Status(stage=stage, ready=False, detail="; ".join(parts), todo=tuple(todo))
        )
    return tuple(statuses)
