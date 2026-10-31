"""Runtime configuration (pydantic-settings, env prefix C1_; API keys also read without prefix)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SERVER_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="C1_",
        env_file=(str(REPO_DIR / ".env"), str(SERVER_DIR / ".env")),
        extra="ignore",
        populate_by_name=True,
    )

    mock_ai: bool = False
    replay: bool = False
    openai_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("C1_OPENAI_API_KEY", "OPENAI_API_KEY"))
    jev_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("C1_JEV_API_KEY", "JEV_API_KEY"))
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("C1_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"))
    typesafe_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("C1_TYPESAFE_API_KEY", "TYPESAFE_API_KEY"))
    typesafe_url: str = "https://api.typesafe.ai/v1/systemone"
    typesafe_model: str = "jev-latest"
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("C1_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"))
    openai_decisions_url: str = "https://api.openai.com/v1/decisions"
    openai_decisions_model: str = "gpt-6-luna"
    jev_url: str = "https://openrouter.ai/api/v1/chat/completions"
    jev_model: str = "typesafe/jev-router"
    jev_samples: int = 5
    llm_model: str = "claude-sonnet-5-5"
    llm_enabled: bool = True
    decisions_timeout_s: float = 5.0
    decisions_budget_s: float = 12.0
    llm_timeout_s: float = 8.0
    seed: int = 1729
    repo_dir: Path = REPO_DIR
    data_dir: Path | None = None
    db_path: Path | None = None
    cache_dir: Path | None = None
    strict_13: bool = False
    judge_mode: bool = False
    boot_scenario: str = "late-cdm"
    approval_hold_max_s: float = 45.0  # real seconds a pending approval may hold the SIM clock
    tick_hz: float = 4.0

    # ---- derived paths -------------------------------------------------
    @property
    def processed_dir(self) -> Path:
        return (self.data_dir or (self.repo_dir / "data")) / "processed"

    @property
    def raw_dir(self) -> Path:
        return (self.data_dir or (self.repo_dir / "data")) / "raw" / "Challenge 1 Airbus "

    @property
    def var_dir(self) -> Path:
        return self.repo_dir / "var"

    @property
    def sqlite_path(self) -> Path:
        return self.db_path or (self.var_dir / "orbitaldodge.sqlite")

    @property
    def cache_root(self) -> Path:
        return self.cache_dir or (self.var_dir / "cache")

    @property
    def scenarios_dir(self) -> Path:
        return self.repo_dir / "shared" / "scenarios"

    @property
    def schema_dir(self) -> Path:
        return self.repo_dir / "shared" / "schema"

    # ---- modes -----------------------------------------------------------
    @property
    def decisions_mode(self) -> str:
        if self.replay:
            return "replay"
        if self.mock_ai or not (self.typesafe_api_key or self.openai_api_key or self.jev_key):
            return "mock"
        return "live"

    @property
    def jev_key(self) -> SecretStr | None:
        """Jev is reached through OpenRouter's typesafe/jev-router; a dedicated JEV key is accepted too."""
        return self.openrouter_api_key or self.jev_api_key

    @property
    def llm_mode(self) -> str:
        if self.llm_enabled and self.anthropic_api_key and not self.mock_ai:
            return "live"
        if self.replay:
            return "replay"
        return "template"


@lru_cache
def get_settings() -> Settings:
    return Settings()
