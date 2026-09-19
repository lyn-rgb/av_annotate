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

**Not yet run against the real checkpoint.**  The call below is written against
the model card's own example; see ``docs/server-setup.md`` for what to confirm
and for the memory arithmetic that decides which checkpoint fits a 48 GB card.
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
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
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

        # ``dtype`` rather than the older ``torch_dtype``: the argument was
        # renamed, and passing the old name is silently accepted by some
        # versions and ignored by others, which turns a quantisation choice into
        # an out-of-memory error an hour later.
        kwargs: dict[str, Any] = {"dtype": dtype or "auto"}
        if device_map:
            kwargs["device_map"] = device_map
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(model, **kwargs)
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
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Sliced from the prompt's length: these models are decoders, and the
        # generated tensor contains the question as well as the answer.
        answer = generated[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(answer, skip_special_tokens=True)[0]
        return str(text).strip()


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
