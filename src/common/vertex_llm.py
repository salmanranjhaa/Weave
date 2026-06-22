"""Vertex AI (Gemini) LLM for LlamaIndex.

A thin CustomLLM that talks to Vertex AI through the google-genai SDK using
Application Default Credentials (ADC) — the same auth path the VM's metadata
server provides, so no key file is needed. Supports model + region fallback,
mirroring how the loaded-out service is wired.

Used by both ERP-RAG (router) and STRATA (ReAct agent).
"""
import os

from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr


def _csv(raw: str) -> list[str]:
    return [v.strip() for v in (raw or "").split(",") if v.strip()]


class VertexGeminiLLM(CustomLLM):
    """LlamaIndex LLM backed by Vertex AI Gemini via ADC."""

    model: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4096)
    project: str = Field(default="")
    location: str = Field(default="us-central1")
    model_fallbacks: list[str] = Field(default_factory=list)
    region_fallbacks: list[str] = Field(default_factory=list)
    context_window: int = Field(default=1_000_000)
    num_output: int = Field(default=8192)

    _clients: dict = PrivateAttr(default_factory=dict)

    @classmethod
    def from_env(cls) -> "VertexGeminiLLM":
        return cls(
            model=os.getenv("VERTEX_AI_MODEL", "gemini-2.5-flash"),
            project=os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            location=os.getenv("GCP_REGION", "us-central1"),
            model_fallbacks=_csv(os.getenv("VERTEX_AI_MODEL_FALLBACKS", "gemini-2.5-flash-lite")),
            region_fallbacks=_csv(os.getenv("VERTEX_AI_FALLBACK_REGIONS", "")),
            temperature=float(os.getenv("VERTEX_AI_TEMPERATURE", "0.1")),
        )

    # ── internals ─────────────────────────────────────────────────────────────
    def _client(self, region: str):
        from google import genai

        if region not in self._clients:
            self._clients[region] = genai.Client(
                vertexai=True, project=self.project, location=region
            )
        return self._clients[region]

    def _candidates(self) -> list[tuple[str, str]]:
        models = [self.model, *self.model_fallbacks]
        regions = [self.location, *self.region_fallbacks]
        seen: set = set()
        out: list[tuple[str, str]] = []
        for r in regions:
            for m in models:
                if m and r and (m, r) not in seen:
                    seen.add((m, r))
                    out.append((m, r))
        return out

    def _generate(self, contents, system_instruction: str | None = None) -> str:
        from google.genai import types

        cfg_kwargs = dict(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
        )
        # Gemini 2.5 models "think" by default, which silently consumes the
        # output budget and can return empty text. Disable it for fast, direct
        # answers (and tool reasoning the ReAct loop drives explicitly).
        if hasattr(types, "ThinkingConfig"):
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(**cfg_kwargs)
        last_err: Exception | None = None
        for model, region in self._candidates():
            try:
                resp = self._client(region).models.generate_content(
                    model=model, contents=contents, config=config
                )
                # Stick to the model/region that worked for subsequent calls.
                self.model, self.location = model, region
                return resp.text or ""
            except Exception as e:  # noqa: BLE001 — try the next candidate
                last_err = e
                print(f"[Vertex] {model}@{region} failed: {e}")
                continue
        raise RuntimeError(f"All Vertex model/region candidates failed. Last error: {last_err}")

    # ── LlamaIndex interface ───────────────────────────────────────────────────
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        return CompletionResponse(text=self._generate(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponseGen:
        text = self._generate(prompt)

        def gen():
            yield CompletionResponse(text=text, delta=text)

        return gen()

    @llm_chat_callback()
    def chat(self, messages, **kwargs) -> ChatResponse:
        from google.genai import types

        system_txt: str | None = None
        contents = []
        for msg in messages:
            content = msg.content or ""
            if msg.role == MessageRole.SYSTEM:
                system_txt = f"{system_txt}\n{content}" if system_txt else content
                continue
            g_role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            contents.append(types.Content(role=g_role, parts=[types.Part(text=content)]))
        if not contents:
            contents = [types.Content(role="user", parts=[types.Part(text=system_txt or "")])]
        text = self._generate(contents, system_instruction=system_txt)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    @llm_chat_callback()
    def stream_chat(self, messages, **kwargs) -> ChatResponseGen:
        resp = self.chat(messages, **kwargs)

        def gen():
            yield resp

        return gen()
