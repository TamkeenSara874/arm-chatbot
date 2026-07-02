from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger()

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "files"


class PromptLoader:
    """Loads YAML prompt files and formats them with runtime variables.

    Files are parsed once and cached in memory for the lifetime of the process.
    """

    def __init__(self, prompts_dir: Path = _PROMPTS_DIR) -> None:
        self._dir = prompts_dir
        self._cache: dict[str, dict] = {}

    def _load(self, name: str) -> dict:
        if name not in self._cache:
            path = self._dir / f"{name}.yaml"
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if name not in data:
                raise KeyError(f"Prompt key '{name}' not found in {path}")
            self._cache[name] = data[name]
            logger.debug("prompt_loaded", name=name)
        return self._cache[name]

    def format(self, name: str, **kwargs: Any) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) with variables substituted.

        Variables in the template are written as {variable_name}. Literal
        curly braces in JSON examples must be escaped as {{...}} in the YAML.
        """
        prompt_def = self._load(name)
        system = prompt_def["system_prompt"].strip()
        user_template = prompt_def["user_prompt"].strip()
        try:
            user = user_template.format(**kwargs)
        except KeyError as exc:
            raise ValueError(f"Prompt '{name}' missing required variable: {exc}") from exc
        return system, user


@functools.lru_cache(maxsize=1)
def get_prompt_loader() -> PromptLoader:
    return PromptLoader()
