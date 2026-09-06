"""Response models for the public API."""

from dataclasses import dataclass, field


@dataclass
class ComplianceResponse:
    """A compliance decision with supporting SOP sources."""

    answer: str
    sources: list[str] = field(default_factory=list)
