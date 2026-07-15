"""
judge_providers.py

Swappable "judge" backends. Every provider implements the same tiny
interface (`JudgeProvider.complete(system, user) -> str`), so the
evaluator doesn't care which LLM is actually doing the judging.

Included providers:
  - AnthropicJudge   (Claude via the Anthropic SDK)
  - OpenAIJudge      (GPT via the OpenAI SDK)
  - MockJudge        (deterministic, no API calls -- for testing the pipeline)
  - CallableJudge    (wrap literally any Python function: local model, another
                      vendor's SDK, an internal gateway, etc.)

To add a new vendor: subclass JudgeProvider, implement `complete()`, done.
Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Callable


class JudgeProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion for a given system+user prompt."""
        raise NotImplementedError


class AnthropicJudge(JudgeProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None, max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("pip install anthropic --break-system-packages") from e
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIJudge(JudgeProvider):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, max_tokens: int = 1024):
        try:
            import openai
        except ImportError as e:
            raise ImportError("pip install openai --break-system-packages") from e
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"openai:{model}"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content


class CallableJudge(JudgeProvider):
    """Wrap any function(system_prompt, user_prompt) -> str as a judge.
    Use this to plug in an internal gateway, a local model server, etc.
    """

    def __init__(self, fn: Callable[[str, str], str], name: str = "custom_callable"):
        self.fn = fn
        self.name = name

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.fn(system_prompt, user_prompt)


class MockJudge(JudgeProvider):
    """No network calls. Produces plausible-looking, randomized-but-seeded
    scores so you can test the full pipeline (loader -> evaluator -> report)
    before spending API credits or getting real credentials wired up.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.name = "mock"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Extract the criterion keys the evaluator asked about, if present,
        # so the mock output still round-trips through the same parser.
        criteria_keys = ["task_completion", "conversational_quality", "accuracy", "safety_compliance"]
        scores = {}
        for k in criteria_keys:
            if k in user_prompt:
                scores[k] = self.rng.choice([3, 3, 4, 4, 4, 5])
        result = {
            "scores": {k: {"score": v, "rationale": f"[mock] plausible rationale for {k}"} for k, v in scores.items()},
            "summary": "[mock] This is a placeholder evaluation for pipeline testing.",
            "flags": [],
        }
        return json.dumps(result)


PROVIDER_REGISTRY = {
    "anthropic": AnthropicJudge,
    "openai": OpenAIJudge,
    "mock": MockJudge,
}


def get_provider(name: str, **kwargs) -> JudgeProvider:
    """Convenience factory: get_provider('anthropic', model='claude-sonnet-4-6')"""
    if name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(PROVIDER_REGISTRY)}")
    return PROVIDER_REGISTRY[name](**kwargs)
