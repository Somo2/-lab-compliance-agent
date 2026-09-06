import re


def normalize_sop_id(value: str) -> str:
    """Normalize common SOP identifier formats to SOP-<number>."""
    normalized = value.strip().upper()

    match = re.fullmatch(r"(?:SOP[\s-]*)?(\d+)", normalized)

    if not match:
        return normalized

    return f"SOP-{match.group(1)}"


def normalize_query(query: str) -> str:
    """Normalize SOP identifiers embedded in a user query."""
    return re.sub(
        r"(?<!\w)(?:SOP[\s-]*)?(\d{3})(?!\w)",
        lambda match: f"SOP-{match.group(1)}",
        query,
        flags=re.IGNORECASE,
    )
