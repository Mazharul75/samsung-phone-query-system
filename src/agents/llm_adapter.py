"""Adapter exposing the local model through LangChain's LLM interface.

The agents are built with LangChain prompt templates and chains.  LangChain
needs a model object that implements its ``LLM`` contract, so this thin adapter
wraps :class:`~src.rag.llm.LocalLLM`.

Wrapping rather than reimplementing means the chatbot and every agent share a
single loaded copy of the model - important when it occupies roughly a
gigabyte of RAM and takes several seconds to start.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from src.rag.llm import LocalLLM, get_llm


class LocalChatLLM(LLM):
    """LangChain-compatible wrapper around the locally hosted model."""

    local_llm: LocalLLM
    system_prompt: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None

    #: LangChain models are pydantic objects; LocalLLM is a plain class.
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "local-transformers"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.local_llm.model_name,
            "backend": self.local_llm.backend,
        }

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        text = self.local_llm.generate(
            prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )

        # Honour LangChain's stop-sequence contract.
        if stop:
            for sequence in stop:
                index = text.find(sequence)
                if index != -1:
                    text = text[:index]
        return text


def build_langchain_llm(
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> LocalChatLLM:
    """Create a LangChain LLM bound to the shared local model instance."""
    return LocalChatLLM(
        local_llm=get_llm(),
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
