"""
config.py — Central configuration loader for agenticblocks examples.

Usage in any example:
    from config import get_model, get_litellm_kwargs

    agent = LLMAgentBlock(
        name="my_agent",
        model=get_model(),
        litellm_kwargs=get_litellm_kwargs(),
    )

Priority for model resolution:
    1. Environment variable  AGENTICBLOCKS_MODEL  (highest)
    2. examples/config.yaml  model: field
    3. fallback argument passed to get_model()    (lowest)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML — available as a transitive dependency of LiteLLM
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_DEFAULT_MODEL = "ollama/granite4:1b"


def _load_yaml() -> dict[str, Any]:
    """Load and return the raw config.yaml as a dict (empty dict on any error)."""
    if not _YAML_AVAILABLE:
        return {}
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_model(fallback: str = _DEFAULT_MODEL) -> str:
    """Return the LLM model to use.

    Resolution order:
      1. ``AGENTICBLOCKS_MODEL`` environment variable (highest priority).
      2. ``model:`` field in ``examples/config.yaml``.
      3. *fallback* argument (defaults to ``"ollama/granite4:1b"``).
    """
    env_model = os.getenv("AGENTICBLOCKS_MODEL")
    if env_model:
        return env_model
    return _load_yaml().get("model", fallback)


def get_litellm_kwargs() -> dict[str, Any]:
    """Return the ``litellm_kwargs`` dict from config.yaml (empty dict if absent)."""
    return _load_yaml().get("litellm_kwargs", {})
