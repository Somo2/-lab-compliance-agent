"""Request models for the public API."""

from dataclasses import dataclass


@dataclass
class ComplianceRequest:
    """A request to assess a laboratory compliance question."""

    query: str
