"""Local, open-source language model used by the chatbot and the agents.

Model choice
------------
``Qwen2.5-1.5B-Instruct`` (Apache-2.0) is small enough to run on a CPU yet
still follows instructions well enough to write a readable product review.
No hosted API and no API key is involved - everything runs locally.

Backends
--------
The same model is served through one of two interchangeable backends:

``openvino``
    Weights are compressed to INT8 and executed with Intel OpenVINO.  On the
    development machine this raised generation from 2 to roughly 10 tokens per
    second - a 5x speed-up - with no noticeable loss of answer quality.  The
    conversion happens once and is cached on disk.

``transformers``
    Plain PyTorch on CPU.  Slower, but depends only on packages that are
    already required, so the project still runs where OpenVINO is unavailable.

The backend is selected automatically: OpenVINO when it is importable,
otherwise PyTorch.  Setting ``LLM_BACKEND`` pins a specific one.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

#: Where converted OpenVINO models are cached between runs.
OV_CACHE_DIR = PROJECT_ROOT / ".cache" / "openvino"


def _openvino_available() -> bool:
    try:
        import optimum.intel  # noqa: F401
    except Exception:
        return False
    return True


class LocalLLM:
    """A locally hosted instruction-tuned model with a chat interface."""

    def __init__(
        self,
        model_name: str | None = None,
        backend: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.llm_model
        self.max_new_tokens = max_new_tokens or settings.llm_max_new_tokens
        self.backend = backend or os.getenv("LLM_BACKEND") or (
            "openvino" if _openvino_available() else "transformers"
        )

        self._model = None
        self._tokenizer = None
        # Generation is not thread safe, but FastAPI serves requests
        # concurrently - so calls are serialised behind this lock.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @property
    def _ov_model_path(self) -> Path:
        safe_name = self.model_name.replace("/", "__")
        return OV_CACHE_DIR / f"{safe_name}-int8"

    def _load_openvino(self):
        from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

        path = self._ov_model_path
        if path.exists():
            logger.info("Loading cached OpenVINO model from %s", path)
            return OVModelForCausalLM.from_pretrained(str(path))

        logger.info(
            "Converting %s to OpenVINO INT8 (one-off, takes a few minutes)...",
            self.model_name,
        )
        model = OVModelForCausalLM.from_pretrained(
            self.model_name,
            export=True,
            quantization_config=OVWeightQuantizationConfig(bits=8),
        )
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path))
        logger.info("OpenVINO model cached at %s", path)
        return model

    def _load_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM

        logger.info("Loading %s with PyTorch (CPU)", self.model_name)
        # Use every available core; generation on CPU is bandwidth bound and
        # the default thread count leaves performance on the table.
        torch.set_num_threads(os.cpu_count() or 4)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        return model

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.backend == "openvino":
            try:
                self._model = self._load_openvino()
                return
            except Exception as exc:
                logger.warning(
                    "OpenVINO backend unavailable (%s); falling back to PyTorch.", exc
                )
                self.backend = "transformers"

        self._model = self._load_transformers()

    def warm_up(self) -> None:
        """Load the model ahead of the first request."""
        self._ensure_loaded()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run one chat completion and return the assistant's reply.

        Sampling is disabled when the temperature is zero or very low, which
        makes factual answers reproducible - important for a system whose job
        is to report specifications accurately.
        """
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        temperature = (
            settings.llm_temperature if temperature is None else temperature
        )
        max_tokens = max_new_tokens or self.max_new_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
        }
        if temperature and temperature > 0.05:
            generation_kwargs.update(
                {"do_sample": True, "temperature": temperature, "top_p": 0.9}
            )
        else:
            generation_kwargs["do_sample"] = False

        with self._lock:
            output = self._model.generate(**inputs, **generation_kwargs)

        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True)
        return _tidy(answer)


def _tidy(text: str) -> str:
    """Clean up small artefacts small models tend to emit."""
    text = text.strip()
    # Drop a leading role label such as "Assistant:".
    text = re.sub(r"^(assistant|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def get_llm() -> LocalLLM:
    """Return the process-wide LLM singleton.

    Loading the model costs time and roughly a gigabyte of RAM, so the chatbot
    and every agent share one instance.
    """
    return LocalLLM()
