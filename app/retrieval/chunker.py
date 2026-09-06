import re

from app.models.schemas import DocumentChunk


class SOPChunker:
    def chunk(self, sop_id: str, content: str) -> list[DocumentChunk]:
        revision = self._extract_revision(content)
        effective_date = self._extract_effective_date(content)

        sections = []
        current_section = None
        current_lines = []

        for line in content.splitlines():
            if line.startswith("## ") or line.startswith("### "):
                self._add_section(
                    sections=sections,
                    sop_id=sop_id,
                    section=current_section,
                    lines=current_lines,
                    revision=revision,
                    effective_date=effective_date,
                )

                current_section = line.lstrip("# ").strip()
                current_lines = []
            else:
                current_lines.append(line)

        self._add_section(
            sections=sections,
            sop_id=sop_id,
            section=current_section,
            lines=current_lines,
            revision=revision,
            effective_date=effective_date,
        )

        return sections

    @staticmethod
    def _extract_revision(content: str) -> int | None:
        match = re.search(r"\*\*Revision:\*\*\s*(\d+)", content)

        if not match:
            return None

        return int(match.group(1))

    @staticmethod
    def _extract_effective_date(content: str) -> str | None:
        match = re.search(
            r"\*\*Revision:\*\*\s*\d+\s*\|\s*"
            r"\*\*Effective Date:\*\*\s*(\d{4}-\d{2}-\d{2})",
            content,
        )

        if not match:
            return None

        return match.group(1)

    @staticmethod
    def _add_section(
        sections: list[DocumentChunk],
        sop_id: str,
        section: str | None,
        lines: list[str],
        revision: int | None,
        effective_date: str | None,
    ) -> None:
        if section is None:
            return

        content = "\n".join(lines).strip()

        if not content:
            return

        sections.append(
            DocumentChunk(
                sop_id=sop_id,
                section=section,
                content=content,
                metadata={
                    "source": f"{sop_id}.md",
                    "revision": revision,
                    "effective_date": effective_date,
                },
            )
        )
