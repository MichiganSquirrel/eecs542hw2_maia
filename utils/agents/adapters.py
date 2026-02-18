from typing import Any

import anthropic
import openai
from anthropic.types import Message
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - optional dependency path
    genai = None
    genai_types = None

from .messages import normalize_messages, to_antrophic, to_gemini_messages


class OpenAIAdapter:
    """
    Uses OpenAI's ChatCompletion (compatible with many proxies).
    Accepts OpenAI-style `messages`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = 'gpt-4o',
        organization: str | None = None,
        base_url: str | None = None,
    ):
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        if base_url:
            openai.api_base = base_url
        self.model = model
        self._allow_system = True  # OpenAI supports system

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        **kwargs,
    ) -> str:
        # Normalize but keep system messages for OpenAI
        norm_messages = normalize_messages(messages, allow_system=self._allow_system)
        if not norm_messages:
            raise ValueError(
                'After normalization, no valid user/assistant messages remain.'
            )
        params: dict[str, Any] = {
            'model': self.model,
            'messages': norm_messages,
            'max_tokens': max_output_tokens,
        }
        params.update(kwargs)
        resp: dict[str, Any] = openai.ChatCompletion.create(**params)
        return resp['choices'][0]['message']['content']


class LocalAdapter(OpenAIAdapter):
    """
    For OpenAI-compatible local servers (vLLM, Ollama bridges, etc.).
    WARNING: sets global openai.api_base; avoid mixing with real OpenAI concurrently.
    """

    def __init__(
        self,
        base_url: str = 'http://localhost:11434/v1',
        model: str = 'llama3',
        api_key: str = 'dummy',
        allow_system: bool = False,  # Gemma / many local backends don't support system
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self._allow_system = allow_system  # override

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        **kwargs,
    ) -> str:
        # Strip/handle system according to local model capability
        norm_messages = normalize_messages(
            messages, merge_system_into_user=True, keep_system=self._allow_system
        )
        if not norm_messages:
            raise ValueError(
                'After normalization, no valid user/assistant messages remain for local model.'
            )

        params: dict[str, Any] = {
            'model': self.model,
            'messages': norm_messages,
            'max_tokens': max_output_tokens,
        }
        params.update(kwargs)
        resp: dict[str, Any] = openai.ChatCompletion.create(**params)
        return resp['choices'][0]['message']['content']


# Anhtropic (Claude Sonnet 4)
class AnthropicAdapter:
    """
    Adapts OpenAI-style messages to Anthropic's.
    """

    def __init__(self, api_key: str, model: str = 'claude-4-sonnet-20250514'):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _split(self, messages: list[dict[str, Any]]):
        system = None
        history: list[dict[str, Any]] = []
        for m in messages:
            role = m.get('role')
            blocks = to_antrophic(m)
            if role == 'system':
                system = blocks or [{'type': 'text', 'text': ''}]
            else:
                history.append(
                    {
                        'role': 'user' if role == 'user' else 'assistant',
                        'content': blocks or [{'type': 'text', 'text': ''}],
                    }
                )
        return system, history

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        **kwargs,
    ) -> str:
        system, messages = self._split(messages)
        params = {
            'model': self.model,
            'system': system,
            'messages': messages,
            'max_tokens': max_output_tokens,
        }
        params.update(kwargs)

        resp: Message = self.client.messages.create(**params)
        return ''.join(
            c.text for c in resp.content if getattr(c, 'type', None) == 'text'
        )


class GoogleGeminiAdapter:
    """
    Adapts OpenAI-style messages to Gemini's role/parts API.
    """

    def __init__(self, api_key: str, model: str = 'gemini-2.5-flash'):
        if genai is None or genai_types is None:
            raise ImportError(
                'google-genai is not installed. Run `uv pip install google-genai`.'
            )
        if not api_key:
            raise ValueError(
                'Missing Gemini API key. Set GEMINI_API_KEY or pass api_key explicitly.'
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        **kwargs,
    ) -> str:
        system_instruction, contents = to_gemini_messages(messages)
        if not contents:
            raise ValueError('After normalization, no valid messages remain for Gemini.')

        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            **kwargs,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        if getattr(resp, 'text', None):
            return resp.text

        texts: list[str] = []
        for candidate in getattr(resp, 'candidates', []) or []:
            content = getattr(candidate, 'content', None)
            for part in getattr(content, 'parts', []) or []:
                text = getattr(part, 'text', None)
                if text:
                    texts.append(text)
        return '\n'.join(texts)
