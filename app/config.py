"""Application configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Paths and settings shared by application components."""

    sop_directory: Path = Path("data/sops")
