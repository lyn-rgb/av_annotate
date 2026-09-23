"""The vision-language model behind an interface, with Qwen3-VL as the choice.

Chosen over the alternatives for the reason this stage exists: it takes several
images in one request and follows a formatting instruction, and the request here
is always "these N frames are one shot" -- a model that could only take one
image would need a shot described N times and reconciled, which is a different
problem with a worse answer.

The adapter takes decoded frames rather than a path.  The stage has already
chosen exactly which frames represent a shot, and handing over a video would
mean the model deciding where the shot starts -- the same division of labour as
the other adapters in this pipeline, and for the same reason.

Reading the call
----------------

The interface was read off transformers 4.57 and the Qwen3-VL cards rather than
recalled.  Three things about it are not what a reader would guess:

* **It is two architectures under one name.**  The dense checkpoints load
  through ``Qwen3VLForConditionalGeneration`` and the mixture-of-experts ones --
  including the 30B-A3B this config names -- through
  ``Qwen3VLMoeForConditionalGeneration``.  ``AutoModelForImageTextToText``
  dispatches on the checkpoint's own config, which is why the auto class is used
  here rather than either concrete one.
* **Thinking is not a switch.**  Qwen3-VL has no ``enable_thinking`` argument --
  that belongs to the text-only Qwen3 -- and passing it is a silent no-op with a
  warning.  Whether a checkpoint reasons is decided by choosing ``-Instruct`` or
  ``-Thinking`` when downloading, and this stage wants the former.
* **The shipped image budget is very large** (16,384 tokens per image), so it is
  the frame's own size that decides the cost here, not the processor's cap.  See
  ``max_edge`` in the stage.

**Not yet run against the real checkpoint.**  Anything behavioural -- whether it
obeys the identifier rule, whether the captions are any good -- is unverified;
see ``docs/server-setup.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

#: The card's example checkpoint.  Roughly 61 GB in bf16, which does not fit a
#: 48 GB accelerator -- see the server notes for the quantised alternatives.
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class CaptionError(RuntimeError):
    """The captioning model could not be built or run."""


class Captioner(Protocol):
    """What the stage needs from a vision-language model, and nothing more."""

    name: str

    def caption(self, frames: Sequence[NDArray[np.uint8]], *, prompt: str) -> str:
        """Describe ``frames`` (RGB, top-to-bottom) according to ``prompt``."""
        ...


class Qwen3VLCaptioner:
    """Qwen3-VL behind :class:`Captioner`."""

    name = "qwen3-vl"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
        max_new_tokens: int = 128,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except (ModuleNotFoundError, ImportError) as error:
            raise CaptionError(
                "the captioning stage needs transformers and torch: pip install "
                "'transformers>=4.57' torch. A checkpoint is fetched from Hugging "
                "Face on first use, so the first run needs network access."
            ) from error

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model)

        # The auto class, and not either concrete one, because Qwen3-VL is two
        # architectures behind one name: the dense checkpoints load through
        # ``Qwen3VLForConditionalGeneration`` and the mixture-of-experts ones --
        # including the 30B-A3B this config names -- through
        # ``Qwen3VLMoeForConditionalGeneration``.  Naming one of them means the
        # other cannot be swapped in without editing code, and naming the wrong
        # one is a load failure at best.
        #
        # ``dtype`` rather than the older ``torch_dtype``, which is deprecated
        # as of 4.57: it still works and warns, and passing both would let the
        # deprecated one lose silently.
        # Before the load, because the load is what needs to know: passing
        # neither a device_map nor a `.to()` afterwards is what leaves it on the
        # CPU.
        device = resolve_device(device, device_map, cuda_available=torch.cuda.is_available())

        kwargs: dict[str, Any] = {"dtype": dtype or "auto"}
        if device_map:
            kwargs["device_map"] = device_map
        self._model = AutoModelForImageTextToText.from_pretrained(model, **kwargs)
        if device and not device_map:
            self._model = self._model.to(torch.device(device))
        self._model.eval()
        self._seed = seed

    def caption(self, frames: Sequence[NDArray[np.uint8]], *, prompt: str) -> str:
        if not frames:
            raise CaptionError("refusing to caption an empty frame list")

        from PIL import Image

        content: list[dict[str, Any]] = [
            {"type": "image", "image": Image.fromarray(frame)} for frame in frames
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # The processor does the image preprocessing as part of applying the
        # template, which is why no separate vision-info step appears here: the
        # pixels are carried inside the message and turned into tensors by the
        # same call that lays out the conversation.
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        with self._torch.no_grad():
            # Greedy, so the same frames give the same caption on every run.
            # A sampled caption is a different dataset each time it is produced,
            # and this one is meant to be a record.
            generated = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        # The prompt is sliced off per example rather than by column, which is
        # the card's own form: these are decoders, and what comes back contains
        # the question as well as the answer.
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated, strict=True)
        ]
        # ``clean_up_tokenization_spaces`` is off because the card says so: the
        # cleanup rules assume English word spacing and mangle anything else,
        # and this pipeline is explicitly multilingual.
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return str(text).strip()


def resolve_device(
    requested: str | None, device_map: str | None, *, cuda_available: bool
) -> str | None:
    """Where to put the model, with ``None`` meaning "pick one".

    Which is what ``None`` means in every other stage's config: S8 hands
    ``"auto"`` to faster-whisper, S7 and S9 leave the choice to their libraries.
    Here it meant the CPU, and silently -- ``from_pretrained`` with neither a
    ``device_map`` nor a ``.to()`` afterwards loads to the CPU and stays there.

    What that looks like from outside is not an error.  It is a card at zero,
    nothing in its memory, and an eight-billion-parameter vision model reading
    every frame of every shot on the CPU: found on a corpus the day the run
    reached S10, with the estimated finish twenty-four hours away.

    An explicit device is honoured even when it says ``cpu``, and a
    ``device_map`` is a placement decision of its own that takes precedence.
    """

    if requested is not None or device_map is not None:
        return requested
    return "cuda" if cuda_available else None


def build_captioner(config: Mapping[str, Any]) -> Captioner:
    """Construct the captioner a stage's config asks for."""

    backend = str(config.get("backend", "qwen3-vl"))
    if backend not in {"qwen3-vl", "qwen3vl"}:
        raise CaptionError(
            f"unknown captioning backend {backend!r}; the plan's choice is 'qwen3-vl'"
        )
    device = config.get("device")
    dtype = config.get("dtype")
    device_map = config.get("device_map")
    return Qwen3VLCaptioner(
        model=str(config.get("model", DEFAULT_MODEL)),
        device=str(device) if device is not None else None,
        dtype=str(dtype) if dtype is not None else None,
        device_map=str(device_map) if device_map is not None else None,
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        seed=int(config.get("seed", 0)),
    )
